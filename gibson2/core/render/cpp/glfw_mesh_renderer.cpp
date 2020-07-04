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

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        glfwWindowHint(GLFW_DEPTH_BITS, 0);
        glfwWindowHint(GLFW_STENCIL_BITS, 0);
        glfwWindowHint(GLFW_SAMPLES, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        // Hide GLFW window by default

        this->window = glfwCreateWindow(m_windowHeight, m_windowHeight, "Gibson GLFW Renderer", NULL, NULL);
        if (this->window == NULL) {
            fprintf(stderr, "ERROR: Failed to create GLFW window.\n");

            exit(EXIT_FAILURE);
        }



        glfwMakeContextCurrent(this->window);
        glfwSwapInterval(0);


        // Load all OpenGL function pointers through GLAD
        if (!gladLoadGL(glfwGetProcAddress))
        {
            fprintf(stderr, "ERROR: Failed to load OpenGL function pointers through GLAD.\n");
            exit(EXIT_FAILURE);
        }

        if(verbosity >= 20) { printf("INFO: Succesfully initialized GLFW context and window!\n");}

        return 0;
    }

void GLFWRendererContext::release() {
    glfwTerminate();
}

PYBIND11_MODULE(GLFWRendererContext, m) {

    py::class_<GLFWRendererContext> pymodule = py::class_<GLFWRendererContext>(m, "GLFWRendererContext");

    pymodule.def(py::init<int, int>());
    pymodule.def("init", &GLFWRendererContext::init);
    pymodule.def("release", &GLFWRendererContext::release);

    // class MeshRenderer
    pymodule.def("render_meshrenderer_pre", &GLFWRendererContext::render_meshrenderer_pre, "pre-executed functions in MeshRenderer.render");
    pymodule.def("render_meshrenderer_post", &GLFWRendererContext::render_meshrenderer_post, "post-executed functions in MeshRenderer.render");
    pymodule.def("getstring_meshrenderer", &GLFWRendererContext::getstring_meshrenderer, "return GL version string");
    pymodule.def("readbuffer_meshrenderer", &GLFWRendererContext::readbuffer_meshrenderer, "read pixel buffer");
    pymodule.def("readbuffer_meshrenderer_shadow_depth", &GLFWRendererContext::readbuffer_meshrenderer_shadow_depth,"read pixel buffer");
    pymodule.def("clean_meshrenderer", &GLFWRendererContext::clean_meshrenderer, "clean meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer", &GLFWRendererContext::setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");
    pymodule.def("setup_pbr", &GLFWRendererContext::setup_pbr, "setup pbr");

    pymodule.def("setup_framebuffer_meshrenderer_ms", &GLFWRendererContext::setup_framebuffer_meshrenderer_ms, "setup framebuffer in meshrenderer with MSAA");
    pymodule.def("blit_buffer", &GLFWRendererContext::blit_buffer, "blit buffer");

    pymodule.def("compile_shader_meshrenderer", &GLFWRendererContext::compile_shader_meshrenderer, "compile vertex and fragment shader");
    pymodule.def("load_object_meshrenderer", &GLFWRendererContext::load_object_meshrenderer, "load object into VAO and VBO");
    pymodule.def("loadTexture", &GLFWRendererContext::loadTexture, "load texture function");
    pymodule.def("allocateTexture", &GLFWRendererContext::allocateTexture, "load texture function");

    // class Instance
    pymodule.def("render_softbody_instance", &GLFWRendererContext::render_softbody_instance, "render softbody in instance.render");
    pymodule.def("initvar_instance", &GLFWRendererContext::initvar_instance, "init uniforms in instance.render");
    pymodule.def("init_material_instance", &GLFWRendererContext::init_material_instance, "init materials in instance.render");
    pymodule.def("draw_elements_instance", &GLFWRendererContext::draw_elements_instance, "draw elements in instance.render and instancegroup.render");

    // class InstanceGroup
    pymodule.def("initvar_instance_group", &GLFWRendererContext::initvar_instance_group, "init uniforms in instancegroup.render");
    pymodule.def("init_material_pos_instance", &GLFWRendererContext::init_material_pos_instance, "init materials and position in instancegroup.render");

    // misc
    pymodule.def("cglBindVertexArray", &GLFWRendererContext::cglBindVertexArray, "binding function");
    pymodule.def("cglUseProgram", &GLFWRendererContext::cglUseProgram, "binding function");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}