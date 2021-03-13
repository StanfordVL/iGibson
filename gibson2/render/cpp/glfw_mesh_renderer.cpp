#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <fstream>
#include <sstream>

#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef USE_GLAD

#include  <glad/egl.h>

#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "glfw_mesh_renderer.h"

namespace py = pybind11;

int GLFWRendererContext::init() {
    verbosity = 20;

    // Initialize GLFW context and window
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: Failed to initialize GLFW.\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, m_glVersionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, m_glVersionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);
    glfwWindowHint(GLFW_SAMPLES, 0);

	// Hide GLFW window if user requests
	if (!m_render_window) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	}

	if (m_fullscreen) {
		this->window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Gibson Renderer Output", glfwGetPrimaryMonitor(), NULL);
	}
	else {
		this->window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Gibson Renderer Output", NULL, NULL);
	}

    if (this->window == NULL) {
        fprintf(stderr, "ERROR: Failed to create GLFW window.\n");

        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(this->window);
    glfwSwapInterval(0);


    // Load all OpenGL function pointers through GLAD
    if (!gladLoadGL(glfwGetProcAddress)) {
        fprintf(stderr, "ERROR: Failed to load OpenGL function pointers through GLAD.\n");
        exit(EXIT_FAILURE);
    }

    if (verbosity >= 20) { printf("INFO: Succesfully initialized GLFW context and window!\n"); }

    return 0;
}

void GLFWRendererContext::release() {
    glfwTerminate();
}

void GLFWRendererContext::render_companion_window_from_buffer(GLuint readBuffer) {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, readBuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glDrawBuffer(GL_BACK);
	glBlitFramebuffer(0, 0, m_windowWidth, m_windowHeight, 0, 0, m_windowWidth, m_windowHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glFlush();
	glfwSwapBuffers(this->window);
	glfwPollEvents();

	if (glfwGetKey(this->window, GLFW_KEY_ESCAPE)) {
		glfwTerminate();
	}
}

PYBIND11_MODULE(GLFWRendererContext, m) {

    py::class_<GLFWRendererContext> pymodule = py::class_<GLFWRendererContext>(m, "GLFWRendererContext");

    pymodule.def(py::init<int, int, int, int, bool, bool>());
    pymodule.def("init", &GLFWRendererContext::init);
    pymodule.def("release", &GLFWRendererContext::release);
	pymodule.def("render_companion_window_from_buffer", &GLFWRendererContext::render_companion_window_from_buffer);

    // class MeshRenderer
    pymodule.def("render_meshrenderer_pre", &GLFWRendererContext::render_meshrenderer_pre,
                 "pre-executed functions in MeshRenderer.render");
    pymodule.def("render_meshrenderer_post", &GLFWRendererContext::render_meshrenderer_post,
                 "post-executed functions in MeshRenderer.render");
    pymodule.def("getstring_meshrenderer", &GLFWRendererContext::getstring_meshrenderer, "return GL version string");
    pymodule.def("readbuffer_meshrenderer", &GLFWRendererContext::readbuffer_meshrenderer, "read pixel buffer");
    pymodule.def("readbuffer_meshrenderer_shadow_depth", &GLFWRendererContext::readbuffer_meshrenderer_shadow_depth,
                 "read pixel buffer");
    pymodule.def("clean_meshrenderer", &GLFWRendererContext::clean_meshrenderer, "clean meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer", &GLFWRendererContext::setup_framebuffer_meshrenderer,
                 "setup framebuffer in meshrenderer");
    pymodule.def("setup_pbr", &GLFWRendererContext::setup_pbr, "setup pbr");

    pymodule.def("setup_framebuffer_meshrenderer_ms", &GLFWRendererContext::setup_framebuffer_meshrenderer_ms,
                 "setup framebuffer in meshrenderer with MSAA");
    pymodule.def("blit_buffer", &GLFWRendererContext::blit_buffer, "blit buffer");

    pymodule.def("compile_shader_meshrenderer", &GLFWRendererContext::compile_shader_meshrenderer,
                 "compile vertex and fragment shader");
    pymodule.def("load_object_meshrenderer", &GLFWRendererContext::load_object_meshrenderer,
                 "load object into VAO and VBO");
    pymodule.def("loadTexture", &GLFWRendererContext::loadTexture, "load texture function");
    pymodule.def("allocateTexture", &GLFWRendererContext::allocateTexture, "load texture function");

    // class Instance
    pymodule.def("render_softbody_instance", &GLFWRendererContext::render_softbody_instance,
                 "render softbody in instance.render");
    pymodule.def("initvar", &GLFWRendererContext::initvar, "init uniforms in instance.render");
    pymodule.def("init_material_instance", &GLFWRendererContext::init_material_instance,
                 "init materials in instance.render");
    pymodule.def("draw_elements_instance", &GLFWRendererContext::draw_elements_instance,
                 "draw elements in instance.render and instancegroup.render");

    // class InstanceGroup
    pymodule.def("init_pos_instance", &GLFWRendererContext::init_pos_instance,
                 "init position in instancegroup.render");

    // misc
    pymodule.def("cglBindVertexArray", &GLFWRendererContext::cglBindVertexArray, "binding function");
    pymodule.def("cglUseProgram", &GLFWRendererContext::cglUseProgram, "binding function");

    // for optimized renderer
    pymodule.def("generateArrayTextures", &GLFWRendererContext::generateArrayTextures, "TBA");
    pymodule.def("renderSetup", &GLFWRendererContext::renderSetup, "TBA");
	pymodule.def("updateHiddenData", &GLFWRendererContext::updateHiddenData, "TBA");
	pymodule.def("updateUVData", &GLFWRendererContext::updateUVData, "TBA");
    pymodule.def("updateDynamicData", &GLFWRendererContext::updateDynamicData, "TBA");
    pymodule.def("renderOptimized", &GLFWRendererContext::renderOptimized, "TBA");
    pymodule.def("clean_meshrenderer_optimized", &GLFWRendererContext::clean_meshrenderer_optimized, "TBA");

    //for skybox
    pymodule.def("loadSkyBox", &GLFWRendererContext::loadSkyBox, "TBA");
    pymodule.def("renderSkyBox", &GLFWRendererContext::renderSkyBox, "TBA");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}