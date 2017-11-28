#pragma once
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class Shader
{
	std::string filename;
	int vertexShader;
	int fragmentShader;
	int prog;

public:
	Shader();
	Shader(std::string name);
	GLuint LoadShaders( const char* ,  const char* );
	void DisplayOpenGLInfo();
	void UpdateUniforms(glm::vec3, glm::vec3, float);



	~Shader();
};
