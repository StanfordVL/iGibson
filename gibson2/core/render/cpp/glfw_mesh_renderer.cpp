#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class GLFWRendererContext {
public:
	GLFWRendererContext(int w, int h) :m_windowHeight(h), m_windowWidth(w) {};

	int m_windowWidth;
	int m_windowHeight;

	int verbosity;

	int init() {
		verbosity = 20;

		// Initialize GLFW context and window
		if (!glfwInit()) {
			fprintf(stderr, "ERROR: Failed to initialize GLFW.\n");
			exit(EXIT_FAILURE);
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// Hide GLFW window by default
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		GLFWwindow* window = glfwCreateWindow(m_windowHeight, m_windowHeight, "Gibson VR Renderer", NULL, NULL);
		if (window == NULL) {
			fprintf(stderr, "ERROR: Failed to create GLFW window.\n");
			exit(EXIT_FAILURE);
		}
		glfwMakeContextCurrent(window);

		// Load all OpenGL function pointers through GLAD
		if (!gladLoadGL(glfwGetProcAddress))
		{
			fprintf(stderr, "ERROR: Failed to load OpenGL function pointers through GLAD.\n");
			exit(EXIT_FAILURE);
		}

		if(verbosity >= 20) { printf("INFO: Succesfully initialized GLFW context and window!\n");}

		return 0;
	}

	void release() {
		glfwTerminate();
	}
};

PYBIND11_MODULE(GLFWRendererContext, m) {
	m.doc() = "C++ GLFW bindings";

	py::class_<GLFWRendererContext>(m, "GLFWRendererContext")
		.def(py::init<int, int>())
		.def("init", &GLFWRendererContext::init)
		.def("release", &GLFWRendererContext::release);

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}