//g++  glad/egl.c glad/gl.c egl.cpp -I glad -lpthread -ldl
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef USE_GLAD
  #include  <glad/egl.h>
#else
  #include <EGL/egl.h>
  #include <EGL/eglext.h>
#endif

#include  <glad/gl.h>


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

int main(){

#ifndef USE_GLAD

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
               (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT) { 
         printf("ERROR: Extension eglQueryDevicesEXT not available"); 
         return(-1); 
    } 
    
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
               (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if(!eglGetPlatformDisplayEXT) { 
         printf("ERROR: Extension eglGetPlatformDisplayEXT not available"); 
         return(-1);  
    }
#endif

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;
    
    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;
    
    m_windowWidth = 256;
    m_windowHeight = 256;
    m_renderDevice = -1;

    int verbosity = 20;

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
    
    EGLInternalData2* m_data = new EGLInternalData2();

    // Load EGL functions
#ifdef USE_GLAD
    int egl_version = gladLoaderLoadEGL(NULL);
    if(!egl_version) {
        fprintf(stderr, "INFO: Probing, EGL cannot run on this device\n");
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
        printf("WARN: eglQueryDevicesEXT failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    printf("%d", num_devices);  //This prints to a file that will be read in python to know the index of the device to use

    return 0;
}


