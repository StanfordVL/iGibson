//g++  glad/egl.c glad/gl.c egl.cpp -I glad -lpthread -ldl
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

//#include  <glad/egl.h>
#include  <glad/gl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <EGL/egl.h>
#include <EGL/eglext.h>


namespace py = pybind11;

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0) {}
};


class CppMeshRenderer{
public:
    CppMeshRenderer(int w, int h, int d):m_windowHeight(h),m_windowWidth(w),m_renderDevice(d) {};

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;


    EGLInternalData2* m_data = NULL;


    int init() {

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

    // Load EGL functions
    /*int egl_version = gladLoaderLoadEGL(NULL);
    if(!egl_version) {
        fprintf(stderr, "failed to EGL with glad.\n");
        exit(EXIT_FAILURE);

    };*/

    // Query EGL Devices
    const int max_devices = 32;
    EGLDeviceEXT egl_devices[max_devices];
    EGLint num_devices = 0;
    EGLint egl_error = eglGetError();
    if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
        egl_error != EGL_SUCCESS) {
        printf("eglQueryDevicesEXT Failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    printf("number of devices found %d\n", num_devices);

    m_data->m_renderDevice = m_renderDevice;
    // Query EGL Screens
    if(m_data->m_renderDevice == -1) {
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
            fprintf(stderr, "Invalid render_device choice: %d < %d.\n", m_data->m_renderDevice, num_devices);
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
        fprintf(stderr, "Unable to initialize EGL\n");
        exit(EXIT_FAILURE);
    }

    /*egl_version = gladLoaderLoadEGL(m_data->egl_display);
    if (!egl_version) {
        fprintf(stderr, "Unable to reload EGL.\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded EGL %d.%d after reload.\n", GLAD_VERSION_MAJOR(egl_version),
           GLAD_VERSION_MINOR(egl_version));
    */

    m_data->success = eglBindAPI(EGL_OPENGL_API);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to bind OpenGL API.\n");
        exit(EXIT_FAILURE);
    }

    m_data->success =
    eglChooseConfig(m_data->egl_display, egl_config_attribs,
                    &m_data->egl_config, 1, &m_data->num_configs);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }
    if (m_data->num_configs != 1) {
        fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
        exit(EXIT_FAILURE);
    }

    m_data->egl_surface = eglCreatePbufferSurface(
                                                  m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
    if (m_data->egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Unable to create EGL surface (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }


    m_data->egl_context = eglCreateContext(
                                           m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
    if (!m_data->egl_context) {
        fprintf(stderr, "Unable to create EGL context (eglError: %d)\n",eglGetError());
        exit(EXIT_FAILURE);
    }

    m_data->success =
        eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface,
                   m_data->egl_context);
    if (!m_data->success) {
        fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }


    return 0;
    };

    void release(){
        eglTerminate(m_data->egl_display);
        delete m_data;
    }

};


PYBIND11_MODULE(CppMeshRenderer, m) {
    py::class_<CppMeshRenderer>(m, "CppMeshRenderer")
        .def(py::init<int, int, int>())
        .def("init", &CppMeshRenderer::init)
        .def("release", &CppMeshRenderer::release);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
