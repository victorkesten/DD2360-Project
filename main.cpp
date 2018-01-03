//#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "window.h"
#include "shader.h"
#include "mesh.h"
#include "nbodysim.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

//#define NUM_PARTICLES 100000
// TODO:
// Code comment/clean


void ProcessInputs(){

}

int main() {
	Window w;
	if(w.InitWindow(1000,1000, "Hey")) {
		return -1;
	}

	int NUM_PARTICLES = getParticleCount();

	Shader shade("basic");
	shade.DisplayOpenGLInfo();

	GLuint prog = shade.LoadShaders("basic.vs", "basic.fs");



	Mesh particle;
	float vertices[] = {
		// front
		-0.5, -0.5,  0.5,
		0.5, -0.5,  0.5,
		0.5,  0.5,  0.5,
		-0.5,  0.5,  0.5,
		// back
		-0.5, -0.5, -0.5,
		0.5, -0.5, -0.5,
		0.5,  0.5, -0.5,
		-0.5,  0.5, -0.5,
	};
	unsigned int indices[] = {
		// front
		0, 1, 2,
		2, 3, 0,
		// top
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// bottom
		4, 0, 3,
		3, 7, 4,
		// left
		4, 5, 1,
		1, 0, 4,
		// right
		3, 2, 6,
		6, 7, 3,
	};

	// This just changes the position of each particle a little bit.
	/*
	for (int i = 0; i < NUM_PARTICLES; i++) {
		particle_array[i] = Mesh();
		for (int j = 0; j < 24; j++) {
			vertices[j] += 1.0f;
		}
		particle_array[i].AddVertices(vertices, indices, sizeof(vertices), sizeof(indices));

	}*/
	particle = Mesh();
	particle.AddVertices(vertices, indices, sizeof(vertices), sizeof(indices));

	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	//float vertices[] = {
	//	0.5f, 0.5f, 0.0f,
	//	0.5f, -0.5f, 0.0f, // left
	//	-0.5f, -0.5f, 0.0f, // right
	//	-0.5f,  0.5f, 0.0f  // top
	//};

	//unsigned int indices[] = {
	//	//0,1,2,1,2,0
	//	0,1,3,
	//	1,2,3
	//};



	//Mesh m(vertices, indices, sizeof(vertices), sizeof(indices));

	//glPolygonMode()
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glm::vec3 move(0,0,67678000.0f);
	//glm::vec3 move (67328352.000000, -67196848.000000, 10654063.000000);
	glm::vec3 rotate(0, 0, 0);
	glm::mat4 view = glm::lookAt(glm::vec3(4.0f, 10.0f, -10.0f),
										glm::vec3(0.0f, 0.0f, 0.0f),
										glm::vec3(0.0f, 1.0f, 0.0f));

	//init particle simulation
	init_particles_planets();

	while (!w.ShouldClose()) {
		//update simulation
		simulateStep();
		//printf("%f %f %f\n", (double)(host_positions[0].x), (double)(host_positions[0].y), (double)(host_positions[0].z));
		//printf("%f %f %f\n", (double)(host_positions[NUM_PARTICLES/2].x), (double)(host_positions[NUM_PARTICLES/2].y), (double)(host_positions[NUM_PARTICLES/2].z));
		//printf("%f %f %f\n", (double)(host_positions[NUM_PARTICLES-1].x), (double)(host_positions[NUM_PARTICLES-1].y), (double)(host_positions[NUM_PARTICLES-1].z));
		w.Draw();

		glUseProgram(prog);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		//float timeValue = glfwGetTime();
		//float greenValue = sin(timeValue) / 2.0f + 0.5f;
		//int mvpLocation = glGetUniformLocation(prog, "MVP");
		//int transformLocation = glGetUniformLocation(prog, "transform");
		int colorLoc = glGetUniformLocation(prog, "color");
		int offsetLoc = glGetUniformLocation(prog, "offset");

		glm::vec4 vec(1.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4 trans;

		trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
		vec = trans * vec;
		int state = glfwGetKey(w.GetWindow(), GLFW_KEY_A);
		if (state == GLFW_PRESS) {
			move += glm::vec3(-1000000.0f, 0.0f, 0.0f);
		}

		int state2 = glfwGetKey(w.GetWindow(), GLFW_KEY_D);
		if (state2 == GLFW_PRESS) {
			move += glm::vec3(1000000.0f, 0.0f, 0.0f);
		}

		int state3 = glfwGetKey(w.GetWindow(), GLFW_KEY_W);
		if (state3 == GLFW_PRESS) {
			move += glm::vec3(0.0f, 1000000.0f, 0.0f);
		}

		int state4 = glfwGetKey(w.GetWindow(), GLFW_KEY_S);
		if (state4 == GLFW_PRESS) {
			move += glm::vec3(0.0f, -1000000.0f, 0.0f);
		}
		int state5 = glfwGetKey(w.GetWindow(), GLFW_KEY_RIGHT);
		if (state5 == GLFW_PRESS) {
			rotate.y += 0.05f;
		}
		int state6 = glfwGetKey(w.GetWindow(), GLFW_KEY_LEFT);
		if (state6 == GLFW_PRESS) {
			rotate.y -= 0.05f;
		}

		int state7 = glfwGetKey(w.GetWindow(), GLFW_KEY_UP);
		if (state7 == GLFW_PRESS) {
			rotate.x += 0.05f;
		}
		int state8 = glfwGetKey(w.GetWindow(), GLFW_KEY_DOWN);
		if (state8 == GLFW_PRESS) {
			rotate.x -= 0.05f;
		}

		int state9 = glfwGetKey(w.GetWindow(), GLFW_KEY_Q);
		if (state9 == GLFW_PRESS) {
			rotate.z+= 0.05f;
		}
		int state10 = glfwGetKey(w.GetWindow(), GLFW_KEY_E);
		if (state10 == GLFW_PRESS) {
			rotate.z -= 0.05f;
		}
		//generate rotation matrix
		//host_positions[NUM_PARTICLES/2] = glm::vec3(0,0,0);
		view = glm::lookAt(move,
									//host_positions[NUM_PARTICLES/2],//look at one particle
									//glm::vec3(0,0,),//look at origin
									move - glm::vec3(0,0,67678000.0f),//look minus z-wards
									//move+rot*vec3(0,0,1),//figure out a way to handle direction vectors, for camera rotation
									glm::vec3(0.0f, 1.0f, 0.0f));

		shade.UpdateUniforms(rotate, view);

		glm::vec4 colors[4];
		colors[0] = glm::vec4(1.0f, 0.1f, 0.5f, 0.1f);
		colors[1] = glm::vec4(1.0f, 0.1f, 0.0f, 0.6f);
		colors[2] = glm::vec4(0.1f, 1.0f, 0.5f, 0.1f);
		colors[3] = glm::vec4(0.1f, 1.0f, 0.0f, 0.6f);
		//m.Draw();
		//float a = 0.0f;
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		for (int i = 0; i < NUM_PARTICLES; i++) {
			//glm::mat4 translation = glm::mat4(1);
			//translation = glm::translate(translation, host_positions[i]);

			//glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
			//glUniformMatrix4fv(mvpLocation, count, transpose, value);
			//glUniformMatrix4fv(transformLocation, 1, GL_FALSE, glm::value_ptr(translation));
			int colIndex = i < NUM_PARTICLES*0.5f ? 0 : 2;
			colIndex += host_types[i];

			//col = glm::vec4(0.3f,  a+0.1f, (a/4)+0.5f, 0.5f);
			glUniform4fv(colorLoc, 1, (float*)glm::value_ptr(colors[colIndex]));
			glUniform3fv(offsetLoc, 1, (float*)glm::value_ptr(host_positions[i]));
			//a += 0.1f;
			particle.Draw();
		}


		//m.Draw();
		/*
		for (int i = 0; i < NUM_PARTICLES; i++) {
			shade.UpdateUniforms(move-host_positions[i], rotate, rot, view);

			glm::vec4 col = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
			glUniform4fv(colorLoc, 1, (float*)glm::value_ptr(col));
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			particle_array[0].Draw();
		}*/

		glfwSwapBuffers(w.GetWindow());

		glfwPollEvents();
	}



	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	return 0;
}
