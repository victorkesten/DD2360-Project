#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <stdlib.h>


/* PASTE THESE INTO THE TOP OF THE OPENGL-FILE*/
//	//---------------------------------------------------- SIMULATION VARIABLES ----------------------------------------------------
//	#include "nbodysim.cpp"
//	bool started = false;
//	const int NUM_PARTICLES = 100;			//currently takes 10ms for 100 particles, 1s for 1000 particles
//	Particle particles[NUM_PARTICLES];
//	//------------------------------------------------------------------------------------------------------------------------------


/* PASTE THIS LINE WHEREVER PARTICLE INITIALIZATION IS APPROPRIATE*/
// init_particles_planets(NUM_PARTICLES, particles);

/* PASTE THESE LINES INTO THE RENDER LOOP */
//	if (started){
//		for (int i = 0; i < 1; i++){
//			simulateStep(NUM_PARTICLES, particles);
//		}
//	}



struct Particle{
	int type;
	glm::vec3 pos;
	glm::vec3 nPos;
	glm::vec3 vel;
	bool lock;
};

//**************************************************** SIMULATION VARIABLES ****************************************************

static const float pi = 3.14159265358979323846;	//Life of Pi

static int timesteps = 0;
static const float epsilon = 47097.5;				//the "minimum" gravitational distance. If smaller, will threshhold at this value to avoid gravitational singularities
static const float D =	376780.0f;					//particle diameter

static const std::string names[2] = {	"silicate",		"iron" };			//element name
static const float masses[2] =		{	7.4161E+19f,	1.9549E+20f	};		//element mass(kg)
static const float rep[2] = {			2.9114E+11f,	5.8228E+11f };		//the repulsive factor
static const float rep_redc[2] = {		0.01,			0.02 };				//the modifier when particles repulse eachother while moving away from eachother
static const float shell_depth[2] = { 0.001, 0.002 };						//shell depth percentage

static const float G = 6.674E-11f;				//gravitational constant
static const float timestep = 1.0E-9f;			//time step			


// Planet spawning variables
static const float mass_ratio = 0.5f;			//the mass distribution between the two planetary bodies (0.5 means equal distribution, 1.0 means one gets all)
static const float rad = 3500000.0f;			//the radius of the planets (that is, the initial particle spawning radius)
static const float collision_speed = 1000;		//the speed with which the planetoids approach eachother
static const float rotational_speed = 1000000;	//the speed with which the planetoids rotate
static const float planet_offset = 2;			//the number times radius each planetoid is spawned from world origin


//**************************************************** SIMULATION FUNCTIONS ****************************************************








static void init_particles_planets(int NUM_PARTICLES, Particle *list){

	int mass1 = NUM_PARTICLES * mass_ratio;
	int mass2 = NUM_PARTICLES * (1.0 - mass_ratio);
	float center1_x = planet_offset * rad;
	float center2_x = -planet_offset * rad;

	//create the first planetary body
	for (int i = 0; i < mass1; i++){
		//POSITION
		float rho1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float mu = (1 - 2 * rho2);
		list[i].pos.x = center1_x + rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * cos(2 * pi * rho3);
		list[i].pos.y = rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * sin(2 * pi * rho3);
		list[i].pos.z = rad * pow(rho1, (1.0 / 3.0)) * mu;

		//TYPE
		list[i].type = 1;

		//ROTATIONAL SPEED		
		float rc = pow((-1), ((int)(list[i].pos.x - center1_x) > 0));
		float r_xz = sqrt(((list[i].pos.x - center1_x)*(list[i].pos.x - center1_x)) + ((list[i].pos.z)*(list[i].pos.z)));
		float theta = atan((list[i].pos.z) / (list[i].pos.x - center1_x));
		list[i].vel.x += rotational_speed*r_xz*sin(theta)*rc;
		list[i].vel.z += -rotational_speed*r_xz*cos(theta)*rc;

		//COLLISION SPEED
		list[i].vel.x += collision_speed;
	}


	//create the second planetary body
	for (int i = mass1; i < NUM_PARTICLES; i++){
		//POSITION
		float rho1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float rho3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float mu = (1 - 2 * rho2);
		list[i].pos.x = center2_x + rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * cos(2 * pi * rho3);
		list[i].pos.y = rad * pow(rho1, (1.0 / 3.0)) * pow((1 - mu*mu), (1.0 / 2.0)) * sin(2 * pi * rho3);
		list[i].pos.z = rad * pow(rho1, (1.0 / 3.0)) * mu;

		//TYPE
		list[i].type = 1;

		//ROTATIONAL SPEED		
		float rc = pow((-1), ((int)(list[i].pos.x - center2_x) > 0));
		float r_xz = sqrt(((list[i].pos.x - center2_x)*(list[i].pos.x - center2_x)) + ((list[i].pos.z)*(list[i].pos.z)));
		float theta = atan((list[i].pos.z) / (list[i].pos.x - center2_x));
		list[i].vel.x += rotational_speed*r_xz*sin(theta)*rc;
		list[i].vel.z += -rotational_speed*r_xz*cos(theta)*rc;

		//COLLISION SPEED
		list[i].vel.x += -collision_speed;
	}

}


/*simulate the next state of particle with index i*/
static void particleStep(int NUM_PARTICLES, Particle *list, int i){

	glm::vec3 force(0, 0, 0);

	int iType = list[i].type;
	float Mi = masses[iType];
	float Ki = rep[iType];
	float KRPi = rep_redc[iType];
	float SDPi = shell_depth[iType];

	for (int j = 0; j < NUM_PARTICLES; j++){
		if (j != i){
			
			int jType = list[j].type;		
			float Mj = masses[jType];			
			float Kj = rep[jType];			
			float KRPj = rep_redc[jType];			
			float SDPj = shell_depth[jType];

			bool isMerging = (glm::dot((list[j].pos - list[i].pos), (list[j].vel - list[i].vel)) <= 0);
			float r = glm::distance(list[i].pos, list[j].pos);
			glm::vec3 unit_vector = glm::normalize(list[j].pos - list[i].pos);
			float gravForce = (G * Mi * Mj / (r*r));
			float repForce = 0.0f;

			if (r < epsilon){ r = epsilon; }
			//--------------------------------------------------------

			//If the two particles doesn't touch at all
			if (D <= r){
				force += (gravForce)* unit_vector;
			}



			else if (D - D*SDPi <= r && D - D*SDPj <= r){
				repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
				force += (gravForce - repForce) * unit_vector;
			}

			//If the shell of one of the particles is penetrated, but not the other
			else if (D - D*SDPi <= r < D - D*SDPj){
				if (isMerging){
					repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = 0.5*(Ki + (Kj*KRPj))*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//If the shell of one of the particles is penetrated, but not the other(same as above, but if the ratios are the opposite)
			else if (D - D*SDPi <= r < D - D*SDPj){
				if (isMerging){
					repForce = 0.5*(Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = 0.5*((Ki*KRPi) + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//If both shells are penetrated
			else if (r < D - D*SDPj && r < D - D*SDPi){
				if (isMerging){
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

	//update nVel via force
	list[i].vel = list[i].vel + (timestep * force);
	list[i].nPos = list[i].pos + (timestep * list[i].vel);
}







/*Simulate one timestep for all particles*/
static void simulateStep(int NUM_PARTICLES, Particle *particles){
	auto start = std::chrono::high_resolution_clock::now();

	//calculate their next positions
	
	for (int i = 0; i < NUM_PARTICLES; i++){
		particleStep(NUM_PARTICLES, particles, i);
	}

	//update their positions
	for (int i = 0; i < NUM_PARTICLES; i++){
		if (particles[i].lock == false){
			particles[i].pos = particles[i].nPos;
		}
				
	}
	timesteps++;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	double timems = elapsed.count()*1000;
	printf("calculation time for one step took %f ms\n", timems);
}



//******************************************************************************************************************************

//			OBSOLETE METHODS

//static void init_particles(int NUM_PARTICLES, Particle *list){
//	for (int i = 0; i < NUM_PARTICLES; i++){
//		float LO = -5000000;
//		float HI = 5000000;
//		float x = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
//		float y = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
//		float z = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));


//		list[i].pos.x = x;
//		list[i].pos.y = y;
//		list[i].pos.z = z;

//		list[i].type = 1;

//	}
//}
