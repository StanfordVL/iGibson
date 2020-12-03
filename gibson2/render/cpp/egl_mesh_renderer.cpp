#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <fstream>
#include <sstream>

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

#include "egl_mesh_renderer.h"

namespace py = pybind11;


int EGLRendererContext::init() {

    verbosity = 20;

#ifndef USE_GLAD
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
               (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT) {
         printf("ERROR: extension eglQueryDevicesEXT not available");
         return(-1);
    }

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
               (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if(!eglGetPlatformDisplayEXT) {
         printf("ERROR: extension eglGetPlatformDisplayEXT not available");
         return(-1);
    }
#endif

    m_data = new EGLInternalData2();

    EGLint egl_config_attribs[] = {EGL_RED_SIZE,
                                   8,
                                   EGL_GREEN_SIZE,
                                   8,
                                   EGL_BLUE_SIZE,
                                   8,
                                   EGL_DEPTH_SIZE,
                                   8,
                                   EGL_SURFACE_TYPE,
                                   EGL_PBUFFER_BIT,
                                   EGL_RENDERABLE_TYPE,
                                   EGL_OPENGL_BIT,
                                   EGL_NONE};

    EGLint egl_pbuffer_attribs[] = {
            EGL_WIDTH, m_windowWidth, EGL_HEIGHT, m_windowHeight,
            EGL_NONE,
    };

#ifdef USE_CUDA
    for (int i = 0; i < MAX_NUM_RESOURCES; i++)
        cuda_res[i] = NULL;
#endif

    // Load EGL functions
#ifdef USE_GLAD
    int egl_version = gladLoaderLoadEGL(NULL);
    if (!egl_version) {
        fprintf(stderr, "ERROR: Failed to EGL with glad.\n");
        exit(EXIT_FAILURE);

    };
#endif

    // Query EGL Devices
    const int max_devices = 32;
    EGLDeviceEXT egl_devices[max_devices];
    EGLint num_devices = 0;
    EGLint egl_error = eglGetError();
    if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
        egl_error != EGL_SUCCESS) {
        printf("WARN: eglQueryDevicesEXT Failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    m_data->m_renderDevice = m_renderDevice;
    // Query EGL Screens
    if (m_data->m_renderDevice == -1) {
        // Chose default screen, by trying all
        for (EGLint i = 0; i < num_devices; ++i) {
            // Set display
            EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                          egl_devices[i], NULL);
            if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {
                int major, minor;
                EGLBoolean initialized = eglInitialize(display, &major, &minor);
                if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
                    m_data->egl_display = display;
                }
            }
        }
    } else {
        // Chose specific screen, by using m_renderDevice
        if (m_data->m_renderDevice < 0 || m_data->m_renderDevice >= num_devices) {
            fprintf(stderr, "ERROR: Invalid render_device choice: %d < %d.\n", m_data->m_renderDevice, num_devices);
            exit(EXIT_FAILURE);
        }

        // Set display
        EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                      egl_devices[m_data->m_renderDevice], NULL);
        if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {
            int major, minor;
            EGLBoolean initialized = eglInitialize(display, &major, &minor);
            if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
                m_data->egl_display = display;
            }
        }
    }

    if (!eglInitialize(m_data->egl_display, NULL, NULL)) {
        fprintf(stderr, "ERROR: Unable to initialize EGL\n");
        exit(EXIT_FAILURE);
    }

#ifdef USE_GLAD
    egl_version = gladLoaderLoadEGL(m_data->egl_display);
    if (!egl_version) {
        fprintf(stderr, "ERROR: Unable to reload EGL.\n");
        exit(EXIT_FAILURE);
    }
#else
    if (verbosity >= 20) { printf("INFO: Not using glad\n");}
#endif

    m_data->success = eglBindAPI(EGL_OPENGL_API);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "ERROR: Failed to bind OpenGL API.\n");
        exit(EXIT_FAILURE);
    }

    m_data->success =
            eglChooseConfig(m_data->egl_display, egl_config_attribs,
                            &m_data->egl_config, 1, &m_data->num_configs);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "ERROR: Failed to choose config (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }
    if (m_data->num_configs != 1) {
        fprintf(stderr, "ERROR: Didn't get exactly one config, but %d\n", m_data->num_configs);
        exit(EXIT_FAILURE);
    }

    m_data->egl_surface = eglCreatePbufferSurface(
            m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
    if (m_data->egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "ERROR: Unable to create EGL surface (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }


    m_data->egl_context = eglCreateContext(
            m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
    if (!m_data->egl_context) {
        fprintf(stderr, "ERROR: Unable to create EGL context (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }

    m_data->success =
            eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface,
                           m_data->egl_context);
    if (!m_data->success) {
        fprintf(stderr, "ERROR: Failed to make context current (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }

    if (!gladLoadGL(eglGetProcAddress)) {
        fprintf(stderr, "ERROR: Failed to load GL with glad.\n");
        exit(EXIT_FAILURE);
    }


    return 0;
};

void EGLRendererContext::release() {
    eglTerminate(m_data->egl_display);
    delete m_data;
#ifdef USE_CUDA
    for (int i = 0; i < MAX_NUM_RESOURCES; i++)
          {
            if (cuda_res[i])
            {
              cudaError_t err = cudaGraphicsUnregisterResource(cuda_res[i]);
              if( err != cudaSuccess)
              {
                std::cout << "WARN: cudaGraphicsUnregisterResource failed: " << err << std::endl;
              }
            }
          }
#endif
}


PYBIND11_MODULE(EGLRendererContext, m) {
    py::class_<EGLRendererContext> pymodule = py::class_<EGLRendererContext>(m, "EGLRendererContext");

    pymodule.def(py::init<int, int, int>());
    pymodule.def("init", &EGLRendererContext::init);
    pymodule.def("release", &EGLRendererContext::release);

#ifdef USE_CUDA
    pymodule.def("map_tensor", &EGLRendererContext::map_tensor);
        pymodule.def("map_tensor_float", &EGLRendererContext::map_tensor_float);
#endif
    // class MeshRenderer
    pymodule.def("render_meshrenderer_pre", &EGLRendererContext::render_meshrenderer_pre,
                 "pre-executed functions in MeshRenderer.render");
    pymodule.def("render_meshrenderer_post", &EGLRendererContext::render_meshrenderer_post,
                 "post-executed functions in MeshRenderer.render");
    pymodule.def("getstring_meshrenderer", &EGLRendererContext::getstring_meshrenderer, "return GL version string");
    pymodule.def("readbuffer_meshrenderer", &EGLRendererContext::readbuffer_meshrenderer, "read pixel buffer");
    pymodule.def("clean_meshrenderer", &EGLRendererContext::clean_meshrenderer, "clean meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer", &EGLRendererContext::setup_framebuffer_meshrenderer,
                 "setup framebuffer in meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer_ms", &EGLRendererContext::setup_framebuffer_meshrenderer_ms,
                 "setup framebuffer in meshrenderer with MSAA");
    pymodule.def("blit_buffer", &EGLRendererContext::blit_buffer, "blit buffer");
    pymodule.def("compile_shader_meshrenderer", &EGLRendererContext::compile_shader_meshrenderer,
                 "compile vertex and fragment shader");
    pymodule.def("load_object_meshrenderer", &EGLRendererContext::load_object_meshrenderer,
                 "load object into VAO and VBO");
    pymodule.def("loadTexture", &EGLRendererContext::loadTexture, "load texture function");
    pymodule.def("setup_pbr", &EGLRendererContext::setup_pbr, "setup pbr");
    pymodule.def("readbuffer_meshrenderer_shadow_depth", &EGLRendererContext::readbuffer_meshrenderer_shadow_depth,
                 "read pixel buffer");
    pymodule.def("allocateTexture", &EGLRendererContext::allocateTexture, "load texture function");

    // class Instance
    pymodule.def("render_softbody_instance", &EGLRendererContext::render_softbody_instance,
                 "render softbody in instance.render");
    pymodule.def("init_material_instance", &EGLRendererContext::init_material_instance,
                 "init materials in instance.render");
    pymodule.def("draw_elements_instance", &EGLRendererContext::draw_elements_instance,
                 "draw elements in instance.render and instancegroup.render");

    // class InstanceGroup
    pymodule.def("initvar", &EGLRendererContext::initvar,
                 "init uniforms in instancegroup.render");
    pymodule.def("init_pos_instance", &EGLRendererContext::init_pos_instance,
                 "init position in instancegroup.render");

    // misc
    pymodule.def("cglBindVertexArray", &EGLRendererContext::cglBindVertexArray, "binding function");
    pymodule.def("cglUseProgram", &EGLRendererContext::cglUseProgram, "binding function");

    // for optimized renderer
    pymodule.def("generateArrayTextures", &EGLRendererContext::generateArrayTextures, "TBA");
    pymodule.def("renderSetup", &EGLRendererContext::renderSetup, "TBA");
    pymodule.def("updateHiddenData", &EGLRendererContext::updateHiddenData, "TBA");
	pymodule.def("updateUVData", &EGLRendererContext::updateUVData, "TBA");
    pymodule.def("updateDynamicData", &EGLRendererContext::updateDynamicData, "TBA");
    pymodule.def("renderOptimized", &EGLRendererContext::renderOptimized, "TBA");
    pymodule.def("clean_meshrenderer_optimized", &EGLRendererContext::clean_meshrenderer_optimized, "TBA");

    //for skybox
    pymodule.def("loadSkyBox", &EGLRendererContext::loadSkyBox, "TBA");
    pymodule.def("renderSkyBox", &EGLRendererContext::renderSkyBox, "TBA");
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
