#ifndef EGL_MESH_RENDERER_HEADER
#define EGL_MESH_RENDERER_HEADER

#include "mesh_renderer.h"

#ifdef USE_GLAD

#include  <glad/egl.h>

#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

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

class EGLRendererContext : public MeshRendererContext {
public:
    int m_renderDevice;
    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;
    EGLInternalData2 *m_data = NULL;

    EGLRendererContext(int w, int h, int d) : MeshRendererContext(w, h), m_renderDevice(d) {};

    int init();

    void release();
};

#endif