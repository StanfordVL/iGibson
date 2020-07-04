#ifndef GLFW_MESH_RENDERER_HEADER
#define GLFW_MESH_RENDERER_HEADER

#include "mesh_renderer.h"

class GLFWRendererContext: public MeshRendererContext {
public:
    GLFWRendererContext(int w, int h): MeshRendererContext(w, h, 0) {};
    GLFWwindow *window = NULL;
    int init();
    void release();
};

#endif