#ifndef GLFW_MESH_RENDERER_HEADER
#define GLFW_MESH_RENDERER_HEADER

#include "mesh_renderer.h"
#ifdef WIN32
#include <GLFW/glfw3.h>
#endif


class GLFWRendererContext: public MeshRendererContext {
public:
    int m_glVersionMajor, m_glVersionMinor;
    bool m_render_window;
    bool m_fullscreen;

    GLFWRendererContext(int w, int h, int glVersionMajor, int glVersionMinor,
    bool render_window = false, bool fullscreen = false): MeshRendererContext(w, h) {
        m_glVersionMajor = glVersionMajor;
        m_glVersionMinor = glVersionMinor;
        m_render_window = render_window;
        m_fullscreen = fullscreen;
    };

    GLFWwindow *window = NULL;
	// By default don't render window and don't use fullscreen
    int init();
    void release();
	void render_companion_window_from_buffer(GLuint readBuffer);
};

#endif