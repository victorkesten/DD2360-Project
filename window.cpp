#include "window.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);


Window::Window()
{
}


Window::~Window()
{
}


// Creates window, sets size and window name.
int Window::InitWindow(int x, int y,  char * _name){
  SCREEN_WIDTH = x;
  SCREEN_HEIGHT = y;
  name = _name;
  glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(x,y,_name,NULL, NULL);

  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
 // glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

 	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
 	{
 		std::cout << "Failed to initialize GLAD" << std::endl;
 		return -1;
 	}
  return 0;
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

bool Window::ShouldClose(){
  return glfwWindowShouldClose(window);
}

void Window::Draw() {
			// input
			// -----
	ProcessInput();

			// Clear Color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
      // Don't do swap buffers just yet.
      // Save this and move into Main renderloop.
	//glfwSwapBuffers(window);

}


// Will be moved into Input Processor.
void Window::ProcessInput()
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}
