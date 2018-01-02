#include "mesh.h"



Mesh::Mesh()
{
}

Mesh::Mesh(float * vertices, unsigned int * indices, int size1, int size2) {
	AddVertices(vertices, indices, size1,size2);
}

Mesh::~Mesh()
{
}

void Mesh::AddVertices(float * vertices, unsigned int * indices, int vert_size, int ind_size) {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ibo);
	//for (int i = 0; i < 9; i++) {
	//	std::cout << vertices[i] << std::endl;
	//}

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vert_size, vertices, GL_STATIC_DRAW);
	//std::cout << sizeof(vertices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, ind_size, indices, GL_STATIC_DRAW);
	size = vert_size/sizeof(float);
	in_size = ind_size/sizeof(float);

	std::cout << ind_size << std::endl;

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

// The vertex attribs will hold stuff like normals and color.
void Mesh::Draw() {
	//glEnableVertexAttribArray(0);
	//glEnableVertexAttribArray(1);
	//glEnableVertexAttribArray(2);
	//glEnableVertexAttribArray(3);

	//glBindBuffer(GL_ARRAY_BUFFER, vbo);	// 0 should be my VBO

	//glBindVertexArray(0); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
	//glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, in_size, GL_UNSIGNED_INT, 0);
	//glDrawElementsInstanced(GL_TRIANGLES, in_size,
	//												GL_UNSIGNED_INT, 0, count);
	//glDisableVertexAttribArray(0);
	//glDisableVertexAttribArray(1);
	//glDisableVertexAttribArray(2);
	//glDisableVertexAttribArray(3);
}
