#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
 
class Mesh
{
private:
	unsigned int vbo;
	unsigned int vao;
	unsigned int ibo;
	int size;
	int in_size;
	int refCount;
public:
	Mesh(float * verts, unsigned int * indices, int, int);
	Mesh();
	~Mesh();

	void AddVertices(float *, unsigned int *, int, int);
	void Draw();
};

