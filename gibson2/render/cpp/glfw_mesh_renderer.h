#ifndef GLFW_MESH_RENDERER_HEADER
#define GLFW_MESH_RENDERER_HEADER

#include "mesh_renderer.h"

class GLFWRendererContext: public MeshRendererContext {
public:
    int m_glVersionMajor, m_glVersionMinor;
    GLFWRendererContext(int w, int h, int glVersionMajor, int glVersionMinor): MeshRendererContext(w, h) {
        m_glVersionMajor = glVersionMajor;
        m_glVersionMinor = glVersionMinor;
    };
    GLFWwindow *window = NULL;
    int init();
    void release();
};

#endif