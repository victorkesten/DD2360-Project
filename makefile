CC = gcc
CXX = g++

COMPILE_FLAGS = -Wall -ggdb -O3 -Wno-write-strings
LINK_FLAGS = -lglfw -lGL -lGLU -ldl -lcudart#-lX11#-lgdi32

# glfw = d:/external/glfw-3.1
glfw = glfw-3.2.1

glfw_inc = $(glfw)/include
glfw_lib = $(glfw)/src

glad = OpenGLLibraries/glad
glad_inc = $(glad)/include

INCLUDES = -I$(glfw_inc) -I$(glad_inc)
LIBRARIES = -L$(glfw_lib) -L/usr/local/cuda/lib64

cpp_files = main.cpp mesh.cpp shader.cpp window.cpp
objects = $(cpp_files:.cpp=.o)
headers = mesh.h shader.h window.h nbodysim.h

all: motm

motm: $(objects) glad.o nbodysim.o
				$(CXX) $(LIBRARIES) -o motm nbodysim.o $(objects) glad.o $(LINK_FLAGS)

$(objects): %.o: %.cpp $(headers) makefile
				$(CXX) $(COMPILE_FLAGS) $(INCLUDES) -c -o $@ $<

glad.o: OpenGLLibraries/glad/src/glad.c
				$(CC) $(COMPILE_FLAGS) $(INCLUDES) -c -o glad.o OpenGLLibraries/glad/src/glad.c

nbodysim.o: nbodysim.cu
	nvcc -arch=sm_53 nbodysim.cu -c -o nbodysim.o -std=c++11

.PHONY: clean
clean:
	rm -f *.o
	rm -f motm
