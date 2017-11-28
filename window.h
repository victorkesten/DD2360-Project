#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

class Window
{
	GLFWwindow * window;
	unsigned int SCREEN_WIDTH;
	unsigned int SCREEN_HEIGHT;
	char * name;
public:
	Window();
	~Window();

	// Takes in settings.
	int InitWindow(int, int, char *);
	void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	bool ShouldClose();

	// bad programming but for now this is what I'll have to do.
	GLFWwindow * GetWindow() { return window; }

	void Draw();
	void ProcessInput();


};
