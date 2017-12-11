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

#define NUM_PARTICLES 10
// TODO:
// Code comment/clean


void ProcessInputs(){

}

int main() {
	Window w;
	if(w.InitWindow(1000,1000, "Hey")) {
		return -1;
	}

	Shader shade("basic");
	shade.DisplayOpenGLInfo();

	GLuint prog = shade.LoadShaders("basic.vs", "basic.fs");

	Mesh * particle_array = new Mesh[10];
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
	for (int i = 0; i < NUM_PARTICLES; i++) {
		particle_array[i] = Mesh();
		for (int j = 0; j < 24; j++) {
			vertices[j] += 1.0f;
		}
		particle_array[i].AddVertices(vertices, indices, sizeof(vertices), sizeof(indices));

	}

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
	glm::vec3 move(0,0,0);
	float rot = 0.0f;
	glm::vec3 rotate(0, 0, 0);


	//init particle simulation
	init_particles_planets();

	while (!w.ShouldClose()) {
		w.Draw();

		glUseProgram(prog);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		//float timeValue = glfwGetTime();
		//float greenValue = sin(timeValue) / 2.0f + 0.5f;
		//int mvpLocation = glGetUniformLocation(prog, "MVP");
		int transformLocation = glGetUniformLocation(prog, "transform");
		int colorLoc = glGetUniformLocation(prog, "color");

		glm::vec4 vec(1.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4 trans;

		trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
		vec = trans * vec;
		int state = glfwGetKey(w.GetWindow(), GLFW_KEY_A);
		if (state == GLFW_PRESS) {
			move += glm::vec3(0.01f, 0.0f, 0.0f);
		}

		int state2 = glfwGetKey(w.GetWindow(), GLFW_KEY_D);
		if (state2 == GLFW_PRESS) {
			move += glm::vec3(-0.01f, 0.0f, 0.0f);
		}

		int state3 = glfwGetKey(w.GetWindow(), GLFW_KEY_W);
		if (state3 == GLFW_PRESS) {
			move += glm::vec3(0.0f, 0.0f, 0.01f);
		}

		int state4 = glfwGetKey(w.GetWindow(), GLFW_KEY_S);
		if (state4 == GLFW_PRESS) {
			move += glm::vec3(0.0f, 0.0f, -0.01f);
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


		shade.UpdateUniforms(move, rotate, rot);

		//glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
		//glUniformMatrix4fv(mvpLocation, count, transpose, value);
		glUniformMatrix4fv(transformLocation, 1, GL_FALSE, glm::value_ptr(trans));
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glm::vec4 col(1.0f, 0.1f, 0.5f, 0.5f);

		//m.Draw();
		float a = 0.0f;
		for (int i = 0; i < NUM_PARTICLES; i++) {
			col = glm::vec4(0.3f,  a+0.1f, (a/4)+0.5f, 0.5f);
			glUniform4fv(colorLoc, 1, (float*)glm::value_ptr(col));
			a += 0.1f;
			particle_array[i].Draw();
		}

		col = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		glUniform4fv(colorLoc, 1, (float*)glm::value_ptr(col));
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//m.Draw();
		for (int i = 0; i < NUM_PARTICLES; i++) {
			particle_array[i].Draw();
		}



		//glUniform3fv(colorLoc, 1, (float*)glm::value_ptr(col));

		//m.Draw();

		//glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		//						//glDrawArrays(GL_TRIANGLES, 0, 6);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		// Gotta do this last. Had this poorly done previously.
		glfwSwapBuffers(w.GetWindow());

		glfwPollEvents();
	}

	//run 10 simulation steps
	for(int i = 0; i < 10; ++i) {
		simulateStep();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}
