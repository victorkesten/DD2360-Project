#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nbodysim.h"

#define TPB 256

//dont forget renaming everything .cu	- or is that needed in VS?


/* PASTE THESE INTO THE TOP OF THE OPENGL-FILE*/
//	#include "nbodysim.cu"
//	bool started = false;
//	Particle *cuda_particles = 0;
//	Particle cpuparticles[NUM_PARTICLES];


/* PASTE THESE LINE WHEREVER PARTICLE INITIALIZATION IS APPROPRIATE*/
//	if (useGpu)printf("Calculating with CUDA.\n");
//	if (!useGpu)printf("Calculating with regular cpu.\n");
//	system("pause");
//	init_particles_planets(cpuparticles);
//	cudaMalloc(&cuda_particles, NUM_PARTICLES * sizeof(Particle));
//	cudaMemcpy(cuda_particles, cpuparticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);


/* PASTE THESE LINES INTO THE RENDER LOOP */
//	if (started) {
//		for (int i = 0; i < SIM_PER_RENDER; i++) {
//			if (useGpu) {
//				simulateStepGPU(NUM_PARTICLES, cpuparticles, cuda_particles);
//			}
//			else {
//				simulateStepCPU(NUM_PARTICLES, cpuparticles);
//			}
//		}
//	}


/*PASTE THIS INTO THE DECONSTRUCTOR, AFTER THE RENDER LOOP*/
//	cudaFree(cuda_particles);

/*
struct Particle {
	int type;
	glm::vec3 pos;
	glm::vec3 nPos;
	glm::vec3 vel;
	bool lock;
};*/


//**************************************************** SIMULATION VARIABLES ****************************************************
#ifdef __CUDA_ARCH__
#define CONST_VAR __constant__ static const
#else
#define CONST_VAR static const
#endif
static const bool useGpu = true;
CONST_VAR float pi = 3.14159265358979323846;	//Life of Pi

static int timesteps = 0;
CONST_VAR float epsilon = 47097.5;				//the "minimum" gravitational distance. If smaller, will threshhold at this value to avoid gravitational singularities
CONST_VAR float D = 376780.0f;					//particle diameter

																// [0] = silicate, [1] = iron
CONST_VAR float masses[2] = { 7.4161E+19f, 1.9549E+20f };		//element mass(kg)
CONST_VAR float rep[2] = { 2.9114E+11f, 5.8228E+11f };		//the repulsive factor
CONST_VAR float rep_redc[2] = { 0.01, 0.02 };				//the modifier when particles repulse eachother while moving away from eachother
CONST_VAR float shell_depth[2] = { 0.001, 0.002 };						//shell depth percentage

CONST_VAR float G = 6.674E-11f;				//gravitational constant
CONST_VAR float timestep = 5.8117f;//5.8117f;			//time step


static const int SIM_PER_RENDER = 1;
CONST_VAR int NUM_PARTICLES = 16384;//44295;			//currently takes 10ms for 10000 particles, 1s for 120000 particles. Goal is 131,072, or 16,384

															// Planet spawning variables
CONST_VAR float mass_ratio = 0.5;//0.5f;			//the mass distribution between the two planetary bodies (0.5 means equal distribution, 1.0 means one gets all)
CONST_VAR float rad = 6371000.0f;			//the radius of the planets (that is, the initial particle spawning radius)
CONST_VAR float collision_speed = -3241.6;		//the speed with which the planetoids approach eachother
CONST_VAR float rotational_speed = (2*3.1516)*1.0f/(8.0f*60.0f*60.0f);//1.0f/(24.0f*60.0f*60.0f);	//the speed with which the planetoids rotate, in revolutions per second multiplied by 2*pi
CONST_VAR float planet_offset = 6.0f;//2;			//the number times radius each planetoid is spawned from world origin
CONST_VAR float planet_offset_z = 0.6f;//number times radius one planet shifted sideways, ensures collision is not head-on

//****************************************************   SIMULATION DATA    ****************************************************
glm::vec3 *host_positions = 0;//major optimisation(?): store each particle member variable separate
glm::vec3 *host_velocities = 0;//this way exploits cache coherency much better
glm::vec3 *host_forces = 0;//it also allows us to pick and choose what data to retrieve from the gpu
uint8_t *host_types = 0;
static glm::vec3 *dev_positions;//no outside file should mess with these
static glm::vec3 *dev_velocities;
static glm::vec3 *dev_acceleration;
static glm::vec3 *dev_forces;
static uint8_t *dev_types;


int getParticleCount() {
	return NUM_PARTICLES;
}
//**************************************************** SIMULATION FUNCTIONS ****************************************************


/*simulate the next state of particle with index i*/
__host__ __device__ static void particleStep(int NUM_PARTICLES, int i, glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *forces, uint8_t *types) {

	glm::vec3 force(0, 0, 0);

	uint8_t iType = types[i];
	float Mi = masses[iType];
	float Ki = rep[iType];
	float KRPi = rep_redc[iType];
	float SDPi = shell_depth[iType];

	for (int j = 0; j < NUM_PARTICLES; j++) {
		if (j != i) {

			uint8_t jType = types[j];
			float Mj = masses[jType];
			float Kj = rep[jType];
			float KRPj = rep_redc[jType];
			float SDPj = shell_depth[jType];


			bool isMerging = (glm::dot((positions[j] - positions[i]), (velocities[j] - velocities[i])) <= 0);
			float r = glm::distance(positions[i], positions[j]);
			glm::vec3 unit_vector = glm::normalize(positions[j] - positions[i]);
			float gravForce = (G * Mi * Mj / (r*r));
			float repForce = 0.0f;

			if (r < epsilon) { r = epsilon; }
			//--------------------------------------------------------

			//If the two particles doesn't touch at all
			//NOTE: it might be a good idea to always just add the gravitational force to the final force
			//since this spares us an if statement, which are costly on the GPU
			if (D <= r) {
				force += (gravForce)* unit_vector;
			}

			else if (D - D*SDPi <= r && D - D*SDPj <= r) {
				repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
				force += (gravForce - repForce) * unit_vector;
			}

			//If the shell of one of the particles is penetrated, but not the other
			else if (D - D*SDPi <= r && r < D - D*SDPj) {
				if (isMerging) {
					repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = 0.5*(Ki + (Kj*KRPj))*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//If the shell of one of the particles is penetrated, but not the other(same as above, but if the ratios are the opposite)
			//else if (D - D*SDPi <= r && r < D - D*SDPj) {//old version...
			else if (D - D*SDPj <= r && r < D - D*SDPi) {//new version
				if (isMerging) {
					repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = 0.5*((Ki*KRPi) + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}


			//If both shells are penetrated
			else if (r < D - D*SDPj && r < D - D*SDPi) {
				if (isMerging) {
					repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = 0.5*((Ki*KRPi) + (Kj*KRPj))*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//--------------------------------------------------------
		}

	}
	forces[i] = force;
	//update nVel via force
	/*
	list[i].vel = list[i].vel + (timestep * force);
	list[i].nPos = list[i].pos + (timestep * list[i].vel);
	*/
}

/*frogleap every particles position from nextPos to pos*/
__global__ static void updateposCuda(int NUM_PARTICLES, glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *forces, uint8_t *types, glm::vec3 *acceleration) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < NUM_PARTICLES) {
		//kind of working...
		velocities[i] = velocities[i] + timestep*(forces[i]/masses[types[i]]); //F = ma, thus a = F/m
		//velocities[i] = velocities[i] + timestep*(forces[i]);
		positions[i] = positions[i] + timestep*velocities[i];

		/*
		//leap frog method from wikipedia
		//doesn't seem to work as well. Might have to do with timestep
		positions[i] = positions[i] + timestep*velocities[i] + 0.5f*timestep*timestep*acceleration[i];
		//NOTE: we need the forces from the precious timestep here
		velocities[i] = velocities[i] + 0.5f*((forces[i]/masses[types[i]])+acceleration[i])*timestep; //F = ma, thus a = F/m
		acceleration[i] = (forces[i]/masses[types[i]]);
		*/
	}
}




/*The CPU variant of the particle n-body simulation loop iteration*/
static void simulateStepCPU() {
	auto start = std::chrono::high_resolution_clock::now();

	//calculate their next positions
	for (int i = 0; i < NUM_PARTICLES; i++) {
		particleStep(NUM_PARTICLES, i, host_positions, host_velocities, host_forces, host_types);
	}
	//update their positions
	for (int i = 0; i < NUM_PARTICLES; i++) {
		host_velocities[i] = host_velocities[i] + timestep*(host_forces[i]/masses[host_types[i]]); //F = ma, thus a = F/m
		host_positions[i] = host_positions[i] + timestep*host_velocities[i];
	}
	timesteps++;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	double timems = elapsed.count() * 1000;
	printf("calculation time for one step took %f ms\n", timems);
}


/*Update every particles next position*/
__global__ static void simstepCuda(int NUM_PARTICLES, glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *forces, uint8_t *types) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= NUM_PARTICLES)return;

	particleStep(NUM_PARTICLES, i, positions, velocities, forces, types);
}

/*The CUDA variant of the particle n-body simulation loop iteration*/
static void simulateStepGPU() {
	int blocks = pow(2, ceil(log(NUM_PARTICLES) / log(2)));

	auto start = std::chrono::high_resolution_clock::now();

	simstepCuda <<< blocks, TPB >> >(NUM_PARTICLES, dev_positions, dev_velocities, dev_forces, dev_types);
	cudaDeviceSynchronize();
	updateposCuda <<< blocks, TPB >> >(NUM_PARTICLES, dev_positions, dev_velocities, dev_forces, dev_types, dev_acceleration);
	cudaDeviceSynchronize();

	cudaMemcpy(host_positions, dev_positions, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_velocities, dev_velocities, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_forces, dev_forces, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	timesteps++;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	double timems = elapsed.count() * 1000;
	printf("calculation time for one step took %f ms\n", timems);
}

void simulateStep() {
	printf("pos: %f %f %f\n", (double)(host_positions[0].x), (double)(host_positions[0].y), (double)(host_positions[0].z));
	printf("vel: %f %f %f\n", (double)(host_velocities[0].x), (double)(host_velocities[0].y), (double)(host_velocities[0].z));
	printf("for: %f %f %f\n", (double)(host_forces[0].x), (double)(host_forces[0].y), (double)(host_forces[0].z));
	for(int i = 0; i < SIM_PER_RENDER; ++i) {
		if(useGpu) {
			simulateStepGPU();
		} else {
			simulateStepCPU();
		}
	}

}

//**************************************************** SIMULATION SETUP ****************************************************
static void prep_planetoid(int i0, int i1, glm::vec3 centerpos, glm::vec3 dir, glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *forces, uint8_t *types, uint8_t coreMaterial, uint8_t shellMaterial, float shellThickness, float rotVel);

/*Initialize the two planetoids.*/
void init_particles_planets() {
	int mass1 = (NUM_PARTICLES * mass_ratio);
	glm::vec3 centerPos = glm::vec3((planet_offset * rad), 0, 0);	//The planetoid center position
	glm::vec3 dir = glm::vec3(1, 0, 0);							//The normalized vector of collision velocity direction (multiplied with the speed factor later)

	//allocate cpu buffers
	host_positions = (glm::vec3*)malloc(NUM_PARTICLES * sizeof(glm::vec3));
	host_velocities = (glm::vec3*)malloc(NUM_PARTICLES * sizeof(glm::vec3));
	host_forces = (glm::vec3*)malloc(NUM_PARTICLES * sizeof(glm::vec3));
	host_types = (uint8_t*)malloc(NUM_PARTICLES * sizeof(uint8_t));

	//Two planets equal in size, moving toward eachother on the x-axis, with a core reaching 50% towards the surface of each planetoid.
	prep_planetoid(0, mass1, centerPos+glm::vec3(0,0,1)*rad*planet_offset_z, dir, host_positions, host_velocities, host_forces, host_types, 1, 0, 0.5, rotational_speed);
	prep_planetoid(mass1, NUM_PARTICLES, -centerPos, -dir, host_positions, host_velocities, host_forces, host_types, 1, 0, 0.5, -rotational_speed);

	//printf("%f %f %f\n", (double)(host_positions[0].x), (double)(host_positions[0].y), (double)(host_positions[0].z));
	//printf("%f %f %f\n", (double)(host_positions[0].x), (double)(host_positions[0].y), (double)(host_positions[0].z));
	//copy cpu particles to the gpu
	cudaMalloc(&dev_positions, NUM_PARTICLES*sizeof(glm::vec3));
	cudaMalloc(&dev_velocities, NUM_PARTICLES*sizeof(glm::vec3));
	cudaMalloc(&dev_forces, NUM_PARTICLES*sizeof(glm::vec3));
	cudaMalloc(&dev_acceleration, NUM_PARTICLES*sizeof(glm::vec3));
	cudaMalloc(&dev_types, NUM_PARTICLES*sizeof(uint8_t));

	cudaMemcpy(dev_positions, host_positions, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_velocities, host_velocities, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_forces, host_forces, NUM_PARTICLES * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemset(dev_acceleration, 0, NUM_PARTICLES * sizeof(glm::vec3));
	cudaMemcpy(dev_types, host_types, NUM_PARTICLES * sizeof(uint8_t), cudaMemcpyHostToDevice);
}


/*If I've thought correctly, this should never be modified. Every planetoid will be created with this
*	Parameters are mass(interval of particle indices), position, and speed.*/
static void prep_planetoid(int i0, int i1, glm::vec3 centerpos, glm::vec3 dir, glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *forces, uint8_t *types, uint8_t coreMaterial, uint8_t shellMaterial, float shellThickness, float rotVel) {
	for (int i = i0; i < i1; i++) {
		//Here we randomly distribute particles uniformly within a sphere
		float rho1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float mu = (1 - 2 * rho2);
		positions[i].x = rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * cos(2 * pi * rho3);
		positions[i].y = rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * sin(2 * pi * rho3);
		positions[i].z = rad * pow(rho1, (1.0 / 3.0)) * mu;

		//To modify the composition of this particle, use the different types.
		//When we wish to create a core of a certain type of matter, we will turn every particle within a certain radius of the core into that type of matter
		if (glm::length(positions[i]) > rad*shellThickness) {//If particle is within shell
			types[i] = shellMaterial;
		} else {//If particle is within core
			types[i] = coreMaterial;
		}

		//Here we add the velocity too the particles to make them rotate along with the planet around its axis
		//float rc = pow((-1), ((int)(positions[i].x - centerpos.x) > 0));//this returns 1 or -1 dependign on if x is to the left or right of the center. Why did we use this?
		float r_xz = sqrt((positions[i].x)*(positions[i].x) + (positions[i].z)*(positions[i].z));
		float theta = atan2(positions[i].z, positions[i].x);//atan(positions[i].z) / (positions[i].x);
		velocities[i].x = rotVel*r_xz*sin(theta);//*rc;
		velocities[i].y = 0.0f;
		velocities[i].z = -rotVel*r_xz*cos(theta);//*rc;
		//Here we add the "collision" velocity to the planetoid
		velocities[i] += dir*collision_speed;


		//Here we position the planetoid to center it on a certain position.
		positions[i] += centerpos;
	}
}
//******************************************************************************************************************************
