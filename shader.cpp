#include "shader.h"

using namespace std;

Shader::Shader()
{
}

Shader::Shader(string name)
{
  filename = name;

  char tab2[1024];
  strncpy(tab2, filename.c_str(), sizeof(tab2));
  tab2[sizeof(tab2) - 1] = 0;

  //LoadShaders("basic.vs","basic.fs");

}

Shader::~Shader()
{
}



uint8_t* readFile(const char *filename)
{
    FILE *file  = NULL;
    uint8_t *data  = NULL;
    unsigned long length = 0;

    file = fopen(filename, "rb");

    if (!file)
    {
        return NULL;
    }

    // Calculate the length
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read the content of the file
    data = (uint8_t*)malloc(sizeof(uint8_t) * (length+1));
    unsigned long int c;
    if((c = fread(data, sizeof(uint8_t),length, file)) != length) {
      free(data);
      printf("only read %i out of %li bytes!\n", c, length);
      return 0;
    }
    data[length] = '\0';

    fclose(file);

    return data;
}

GLuint createShader(GLenum shader_type, const char *shader_filename)
{
  GLuint   shader      = 0;
	uint8_t *shader_data = NULL;
	GLint    hr          = GL_TRUE;
  int      length      = 0;

    // Create the shader and read the code from the file
	shader      = glCreateShader(shader_type);
	shader_data = readFile(shader_filename);//make a null check here

	// Compile the shader
	glShaderSource(shader, 1, (const GLchar**)&shader_data, NULL);
	glCompileShader(shader);

	// Check if the Vertex Shader compiled successfully
	glGetShaderiv(shader, GL_COMPILE_STATUS, &hr);
	if (hr == GL_FALSE)
  {
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

		vector<char> error_string(length+1);
		glGetShaderInfoLog(shader, length, NULL, &error_string[0]);

		fprintf(stderr, "Error compiling the shader:\n%s\n", &error_string[0]);
	}

	free(shader_data);

	return shader;
}

GLuint Shader::LoadShaders(const char *vertex_shader_filename, const char *fragment_shader_filename)
{
  GLuint program         = 0;
  GLuint vertex_shader   = 0;
  GLuint fragment_shader = 0;
	GLint  hr              = GL_TRUE;
  int    length          = 0;
	std::cout << vertex_shader_filename << std::endl;
    // Create the Vertex and Fragment Shaders
	vertex_shader   = createShader(GL_VERTEX_SHADER,   vertex_shader_filename);
	fragment_shader = createShader(GL_FRAGMENT_SHADER, fragment_shader_filename);

	// Create the program, attach the shaders and link it
	program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	// Bind the vertex position to index 0
    // glBindAttribLocation(program, 0, "vertexPosition");

    // Link the OpenGL program
	glLinkProgram(program);

	// Check if the program is linked
	glGetShaderiv(program, GL_LINK_STATUS, &hr);
	if (hr == GL_FALSE)
	{
	  glGetShaderiv(program, GL_INFO_LOG_LENGTH, &length);

		vector<char> error_string(length+1);
		glGetProgramInfoLog(program, length, NULL, &error_string[0]);

		fprintf(stderr, "Error linking the program:\n%s\n", &error_string[0]);
	}
	else {
		std::cout << "Linked and ready!" << std::endl;
	}

	// Release all the memory
	//glDetachShader(program, vertex_shader);
	//glDetachShader(program, fragment_shader);
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	prog = program;
	return program;
}

// This function usually takes in a Transform, material and a renderingEngine reference that contains the main camera.
void Shader::UpdateUniforms(glm::vec3 trans, glm::vec3 rotate, float rad) {
	// create transformation matrices
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	model = glm::rotate(model, glm::radians(-55.0f + rotate.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotate.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotate.z), glm::vec3(0.0f, 0.0f, 1.0f));

	view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
	view = glm::translate(view, trans);
	projection = glm::perspective(glm::radians(45.0f), (float)1000 / (float)1000, 0.1f, 100.0f);
	// retrieve the matrix uniform locations
	unsigned int modelLoc = glGetUniformLocation(prog, "model");
	unsigned int viewLoc = glGetUniformLocation(prog, "view");
	unsigned int projectionLoc = glGetUniformLocation(prog, "projection");
	// pass them to the shaders (3 different ways)
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
}

void Shader::DisplayOpenGLInfo()
{
    // Display information about the GPU and OpenGL version
    printf("OpenGL v%d.%d\n", GLVersion.major, GLVersion.minor);
    printf("Vendor: %s\n",    glGetString(GL_VENDOR));
    printf("Renderer: %s\n",  glGetString(GL_RENDERER));
    printf("Version: %s\n",   glGetString(GL_VERSION));
    printf("GLSL: %s\n",      glGetString(GL_SHADING_LANGUAGE_VERSION));
}
