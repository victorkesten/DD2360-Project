//#pragma once
#include <glad/glad.h>
#include <glad/glad.c>
#include <GLFW/glfw3.h>
#include "window.h"
#include "shader.h"
#include "mesh.h"
#include "nbodysim.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include <iostream>

//#define NUM_PARTICLES 100000
// TODO:
// Code comment/clean

static bool hasFocus = false;

float camera_distance = 100000000.0f;
float move_speed = 1000000.0f;
glm::vec3 camPos(0, 0, camera_distance);
glm::vec3 rotate(0, 0, 0);

float horizontalAngle = 0;
float verticalAngle = 0;
glm::vec3 direction = glm::vec3(0, 0, -camera_distance);

glm::mat4 view = glm::lookAt(glm::vec3(4.0f, 10.0f, -10.0f),
	glm::vec3(0.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 1.0f, 0.0f));



void window_focus_callback(GLFWwindow* window, int focused);


int main() {
	Window w;
	if (w.InitWindow(1000, 1000, "Hey")) {
		return -1;
	}
	GLFWwindow* window = w.GetWindow();
	int NUM_PARTICLES = getParticleCount();

	glfwSetWindowFocusCallback(window, window_focus_callback);

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


	//init particle simulation
	init_particles_planets();

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);	//hide cursor :)
	int w0, h0;
	glfwGetWindowSize(window, &w0, &h0);
	glfwSetCursorPos(window, w0 / 2, h0 / 2);



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
		glm::vec3 right = glm::vec3(
			sin(horizontalAngle - 3.14f / 2.0f),
			0,
			cos(horizontalAngle - 3.14f / 2.0f)
		);


		trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
		vec = trans * vec;

		if (hasFocus) {
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { camPos += move_speed*direction / glm::length(direction); }
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { camPos -= move_speed*direction / glm::length(direction); }

			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { camPos += move_speed*right; }
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { camPos -= move_speed*right; }

			//******************* MOUSE INPUT *****************************

			double xpos, ypos;
			float pi = 3.1413;
			int width, height;
			glfwGetCursorPos(window, &xpos, &ypos);
			glfwGetWindowSize(window, &width, &height);

			glfwSetCursorPos(window, width / 2, height / 2);

			horizontalAngle += 0.0001 * float(width / 2 - xpos);
			verticalAngle -= 0.0001 * float(height / 2 - ypos);

			if (verticalAngle > 0.5 * pi) { verticalAngle = 0.5 * pi; }
			if (verticalAngle < -0.5 * pi) { verticalAngle = -0.5 * pi; }

			direction = glm::vec3(
				cos(verticalAngle) * sin(horizontalAngle),
				sin(verticalAngle),
				cos(verticalAngle) * cos(horizontalAngle)
			);
			direction *= -camera_distance;

			//********************************************************************************************
		}



		//generate rotation matrix


		view = glm::lookAt(
			camPos,
			camPos + direction,
			glm::vec3(0.0f, 1.0f, 0.0f)
		);

		shade.UpdateUniforms(rotate, view);

		glm::vec4 colors[4];
		colors[0] = glm::vec4(1.0f, 0.1f, 0.5f, 0.1f);
		colors[1] = glm::vec4(1.0f, 0.1f, 0.0f, 0.6f);
		colors[2] = glm::vec4(0.1f, 1.0f, 0.5f, 0.1f);
		colors[3] = glm::vec4(0.1f, 1.0f, 0.0f, 0.6f);
		//m.Draw();
		//float a = 0.0f;
		auto start = std::chrono::high_resolution_clock::now();
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
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		double timems = elapsed.count() * 1000;
		printf("drawing time for one step took %f ms\n", timems);

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


void window_focus_callback(GLFWwindow* window, int focused)
{
	if (focused)
	{
		hasFocus = true;
	}
	else
	{
		hasFocus = false;
	}
}
