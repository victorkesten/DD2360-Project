#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <iostream>
#include <chrono>




/* PASTE THESE INTO THE TOP OF THE OPENGL-FILE*/
//	//---------------------------------------------------- SIMULATION VARIABLES ----------------------------------------------------
//	#include "nbodysim.cpp"
//	bool started = false;
//	const int NUM_PARTICLES = 100;			//currently takes 10ms for 100 particles, 1s for 1000 particles
//	Particle particles[NUM_PARTICLES];
//	//------------------------------------------------------------------------------------------------------------------------------



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

static int timesteps = 0;
static const float epsilon = 47097.5;
static const float D =	376780.0f;


static const std::string names[2] = { "silicate", "iron" };
static const float masses[2] =		{	7.4161E+19f,	1.9549E+20f		};		//UNIT: kilograms
static const float rep[2] = { 2.9114E+11f, 5.8228E+11f };
static const float rep_redc[2] = { 0.01, 0.02 };
static const float shell_depth[2] = { 0.001, 0.002 };

static const float G = 6.674E-11f;								//not specified in paper(?) guess value
static const float timestep = 1.0E-9f;												//UNIT: seconds











//**************************************************** SIMULATION FUNCTIONS ****************************************************


static void init_particles(int NUM_PARTICLES, Particle *list){
	for (int i = 0; i < NUM_PARTICLES; i++){
		float LO = -5000000;
		float HI = 5000000;
		float x = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		float y = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		float z = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));


		list[i].pos.x = x;
		list[i].pos.y = y;
		list[i].pos.z = z;

		list[i].type = 1;
		
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

			bool isMerging = (glm::dot((list[j].pos - list[i].pos), (list[j].vel - list[i].vel)) < 0);
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
				repForce = (Ki + Kj)*((D*D) - (r*r));
				force += (gravForce - repForce) * unit_vector;
			}

			//If the shell of one of the particles is penetrated, but not the other
			else if (D - D*SDPi <= r < D - D*SDPj){
				if (isMerging){
					repForce = (Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = (Ki + (Kj*KRPj))*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//If the shell of one of the particles is penetrated, but not the other(same as above, but if the ratios are the opposite)
			else if (D - D*SDPi <= r < D - D*SDPj){
				if (isMerging){
					repForce = (Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = ((Ki*KRPi) + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
			}

			//If both shells are penetrated
			else if (r < D - D*SDPj && r < D - D*SDPi){
				if (isMerging){
					repForce = (Ki + Kj)*((D*D) - (r*r));
					force += (gravForce - repForce) * unit_vector;
				}
				else {
					repForce = ((Ki*KRPi) + (Kj*KRPj))*((D*D) - (r*r));
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
