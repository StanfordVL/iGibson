#ifndef GLFW_MESH_RENDERER_HEADER
#define GLFW_MESH_RENDERER_HEADER

#include "mesh_renderer.h"
#include <GLFW\glfw3.h>

class GLFWRendererContext: public MeshRendererContext {
public:
	int width;
	int height;

    GLFWRendererContext(int w, int h): MeshRendererContext(w, h), width(w), height(h) {};
    GLFWwindow *window = NULL;
	// By default don't render window and don't use fullscreen
    int init(bool render_window = false, bool fullscreen = false);
    void release();
	void render_companion_window_from_buffer(GLuint readBuffer);
};

#endif