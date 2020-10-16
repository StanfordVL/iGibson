#ifndef GLFW_MESH_RENDERER_HEADER
#define GLFW_MESH_RENDERER_HEADER

#include "mesh_renderer.h"
#ifdef WIN32
#include <GLFW/glfw3.h>
#endif


class GLFWRendererContext: public MeshRendererContext {
public:
    int m_glVersionMajor, m_glVersionMinor;
    GLFWRendererContext(int w, int h, int glVersionMajor, int glVersionMinor): MeshRendererContext(w, h) {
        m_glVersionMajor = glVersionMajor;
        m_glVersionMinor = glVersionMinor;
    };

    GLFWwindow *window = NULL;
	// By default don't render window and don't use fullscreen
    int init(bool render_window = false, bool fullscreen = false);
    void release();
	void render_companion_window_from_buffer(GLuint readBuffer);
};

#endif
