#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <iostream>
#include <thread>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glad/gl.h>

#include <glad/gl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>

#include <openvr.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Enums.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define BUFFER_OFFSET(offset) (static_cast<char*>(0) + (offset))

#define MAX_NUM_RESOURCES 10

namespace py = pybind11;
#define MAX_ARRAY_SIZE 512

class MeshRendererContext{
public:
	MeshRendererContext(int w, int h) : renderHeight(h), renderWidth(w) {};

	GLFWwindow* window;
	int renderWidth;
	int renderHeight;
	bool usingCompanionWindow;
	bool fullscreen;

	// Index data
	std::vector<void*> multidrawStartIndices;
	std::vector<int> multidrawCounts;
	int multidrawCount;

	// UBO data
	GLuint uboTexColorData;
	GLuint uboTransformData;
	int texColorDataSize;
	int transformDataSize;

	struct VertexDataWindow {
		glm::vec2 position;
		glm::vec2 texCoord;

		//VertexDataWindow(const glm::vec2& pos, const glm::vec2& tex) : position(pos), texCoord(tex) {}
	};

	GLuint cwVAO, cwVBO, cwIndexBuffer, cwIndexSize;

	int init(bool usingCompanionWindow, bool fullscreen) {
		this->usingCompanionWindow = usingCompanionWindow;
		this->fullscreen = fullscreen;

		// Initialize GLFW context and window
		if (!glfwInit()) {
			fprintf(stderr, "Failed to initialize GLFW.\n");
			exit(EXIT_FAILURE);
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// Hide GLFW window if user requests
		if (!usingCompanionWindow) {
			glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		}

		if (fullscreen) {
			this->window = glfwCreateWindow(renderWidth, renderHeight, "Gibson Renderer Output", glfwGetPrimaryMonitor(), NULL);
		}
		else {
			this->window = glfwCreateWindow(renderWidth, renderHeight, "Gibson Renderer Output", NULL, NULL);
		}

		if (window == NULL) {
			fprintf(stderr, "Failed to create GLFW window.\n");
			exit(EXIT_FAILURE);
		}
		// Move window to top-left corner of the screeen
		glfwSetWindowPos(window, 0, 0);
		glfwMakeContextCurrent(window);

		// Turns Vsync off (1 to turn it on)
		glfwSwapInterval(0);

		printf("Succesfully initialized both GLFW context and window!\n");

		return 0;
	}

	void render_companion_window_from_buffer(GLuint readBuffer) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, readBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glDrawBuffer(GL_BACK);
		glBlitFramebuffer(0, 0, this->renderWidth, this->renderHeight, 0, 0, this->renderWidth, this->renderHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		// TODO: Do I need this?
		glFlush();
		glfwSwapBuffers(this->window);
		glfwPollEvents();

		// TODO: Add a more graceful exit?
		if (this->usingCompanionWindow) {
			if (glfwGetKey(this->window, GLFW_KEY_ESCAPE)) {
				glfwTerminate();
			}
		}
	}

	void release() {
		glfwTerminate();
	}

    void render_meshrenderer_pre(bool msaa, GLuint fb1, GLuint fb2) {

        glBindFramebuffer(GL_FRAMEBUFFER, fb2);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (msaa) {
            glBindFramebuffer(GL_FRAMEBUFFER, fb1);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        glEnable(GL_DEPTH_TEST);
    }

    void render_meshrenderer_post() {
        glDisable(GL_DEPTH_TEST);
    }

    void glad_init() {
		/* if (!gladLoadGL(eglGetProcAddress)) {
		fprintf(stderr, "failed to load GL with glad.\n");
		exit(EXIT_FAILURE);
		} */

		if (!gladLoadGL(glfwGetProcAddress)) {
			fprintf(stderr, "Failed to load GL with glad.\n");
			exit(EXIT_FAILURE);
		}
    }

    std::string getstring_meshrenderer() {
        return reinterpret_cast<char const *>(glGetString(GL_VERSION));
    }

    void blit_buffer(int width, int height, GLuint fb1, GLuint fb2) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb1);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb2);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        for (int i = 0; i < 4; i++) {
            glReadBuffer(GL_COLOR_ATTACHMENT0+i);
			glDrawBuffer(GL_COLOR_ATTACHMENT0+i);
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
    }

    py::array_t<float> readbuffer_meshrenderer(char* mode, int width, int height, GLuint fb2) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb2);
        if (!strcmp(mode, "rgb")) {
            glReadBuffer(GL_COLOR_ATTACHMENT0);
        }
        else if (!strcmp(mode, "normal")) {
            glReadBuffer(GL_COLOR_ATTACHMENT1);
        }
        else if (!strcmp(mode, "seg")) {
            glReadBuffer(GL_COLOR_ATTACHMENT2);
        }
        else if (!strcmp(mode, "3d")) {
            glReadBuffer(GL_COLOR_ATTACHMENT3);
        }
        else {
            fprintf(stderr, "ERROR: Unknown buffer mode.\n");
            exit(EXIT_FAILURE);
        }
        py::array_t<float> data = py::array_t<float>(4 * width * height);
        py::buffer_info buf = data.request();
        float* ptr = (float *) buf.ptr;
        glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, ptr);
        return data;
    }

    void clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2, std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos) {
        glDeleteTextures(texture1.size(), texture1.data());
        glDeleteTextures(texture2.size(), texture2.data());
        glDeleteFramebuffers(fbo.size(), fbo.data());
        glDeleteBuffers(vaos.size(), vaos.data());
        glDeleteBuffers(vbos.size(), vbos.data());
    }

	void clean_meshrenderer_optimized(std::vector<GLuint> color_attachments, std::vector<GLuint> textures, std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos, std::vector<GLuint> ebos) {
		glDeleteTextures(color_attachments.size(), color_attachments.data());
		glDeleteTextures(textures.size(), textures.data());
		glDeleteFramebuffers(fbo.size(), fbo.data());
		glDeleteBuffers(vaos.size(), vaos.data());
		glDeleteBuffers(vbos.size(), vbos.data());
		glDeleteBuffers(ebos.size(), ebos.data());
		glDeleteBuffers(1, &uboTexColorData);
		glDeleteBuffers(1, &uboTransformData);
	}

    py::list setup_framebuffer_meshrenderer(int width, int height) {
        GLuint *fbo_ptr = (GLuint*)malloc(sizeof(GLuint));
        GLuint *texture_ptr = (GLuint*)malloc(5 * sizeof(GLuint));
        glGenFramebuffers(1, fbo_ptr);
        glGenTextures(5, texture_ptr);
		int fbo = fbo_ptr[0];
        int color_tex_rgb = texture_ptr[0];
        int color_tex_normal = texture_ptr[1];
        int color_tex_semantics = texture_ptr[2];
        int color_tex_3d = texture_ptr[3];
        int depth_tex = texture_ptr[4];
        glBindTexture(GL_TEXTURE_2D, color_tex_rgb);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, color_tex_normal);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, color_tex_semantics);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, color_tex_3d);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_rgb, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, color_tex_normal, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, color_tex_semantics, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, color_tex_3d, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
        glViewport(0, 0, width, height);
        GLenum *bufs = (GLenum*)malloc(4 * sizeof(GLenum));
        bufs[0] = GL_COLOR_ATTACHMENT0;
        bufs[1] = GL_COLOR_ATTACHMENT1;
        bufs[2] = GL_COLOR_ATTACHMENT2;
        bufs[3] = GL_COLOR_ATTACHMENT3;
        glDrawBuffers(4, bufs);
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        py::list result;
        result.append(fbo);
        result.append(color_tex_rgb);
        result.append(color_tex_normal);
        result.append(color_tex_semantics);
        result.append(color_tex_3d);
        result.append(depth_tex);
        return result;
    }

    py::list setup_framebuffer_meshrenderer_ms(int width, int height) {
        GLuint *fbo_ptr = (GLuint*)malloc(sizeof(GLuint));
        GLuint *texture_ptr = (GLuint*)malloc(5 * sizeof(GLuint));
        glGenFramebuffers(1, fbo_ptr);
        glGenTextures(5, texture_ptr);
        int fbo = fbo_ptr[0];
        int color_tex_rgb = texture_ptr[0];
        int color_tex_normal = texture_ptr[1];
        int color_tex_semantics = texture_ptr[2];
        int color_tex_3d = texture_ptr[3];
        int depth_tex = texture_ptr[4];
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depth_tex);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_DEPTH24_STENCIL8, width, height, GL_TRUE);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depth_tex, 0);
        glViewport(0, 0, width, height);
        GLenum *bufs = (GLenum*)malloc(4 * sizeof(GLenum));
        bufs[0] = GL_COLOR_ATTACHMENT0;
        bufs[1] = GL_COLOR_ATTACHMENT1;
        bufs[2] = GL_COLOR_ATTACHMENT2;
        bufs[3] = GL_COLOR_ATTACHMENT3;
        glDrawBuffers(4, bufs);
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        py::list result;
        result.append(fbo);
        result.append(color_tex_rgb);
        result.append(color_tex_normal);
        result.append(color_tex_semantics);
        result.append(color_tex_3d);
        result.append(depth_tex);
        return result;
    }

    py::list compile_shader_meshrenderer(char* vertexShaderSource, char* fragmentShaderSource) {
        int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        int texUnitUniform = glGetUniformLocation(shaderProgram, "texUnit");
        py::list result;
        result.append(shaderProgram);
        result.append(texUnitUniform);
        return result;
    }

    py::list load_object_meshrenderer(int shaderProgram, py::array_t<float> vertexData) {
        GLuint VAO;
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        GLuint VBO;
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        py::buffer_info buf = vertexData.request();
        float* ptr = (float *) buf.ptr;
        glBufferData(GL_ARRAY_BUFFER, vertexData.size()*sizeof(float), ptr, GL_STATIC_DRAW);
        GLuint positionAttrib = glGetAttribLocation(shaderProgram, "position");
        GLuint normalAttrib = glGetAttribLocation(shaderProgram, "normal");
        GLuint coordsAttrib = glGetAttribLocation(shaderProgram, "texCoords");
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
        glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)12);
        glVertexAttribPointer(coordsAttrib, 2, GL_FLOAT, GL_TRUE, 32, (void*)24);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        py::list result;
        result.append(VAO);
        result.append(VBO);
        return result;
    }

    void render_softbody_instance(int vao, int vbo, py::array_t<float> vertexData) {
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        py::buffer_info buf = vertexData.request();
        float* ptr = (float *) buf.ptr;
        glBufferData(GL_ARRAY_BUFFER, vertexData.size()*sizeof(float), ptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void initvar_instance(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> pose_trans, py::array_t<float> pose_rot, py::array_t<float> lightpos, py::array_t<float> lightcolor) {
        glUseProgram(shaderProgram);
        float *Vptr = (float *) V.request().ptr;
        float *Pptr = (float *) P.request().ptr;
        float *transptr = (float *) pose_trans.request().ptr;
        float *rotptr = (float *) pose_rot.request().ptr;
        float *lightposptr = (float *) lightpos.request().ptr;
        float *lightcolorptr = (float *) lightcolor.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
    }

    void init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color, float use_texture) {
        float *diffuse_ptr = (float *) diffuse_color.request().ptr;
        glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
        glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
        glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
    }

	void draw_elements_instance(int shaderProgram, bool flag, int texture_id, int texUnitUniform, int vao, int face_size, py::array_t<unsigned int> faces, GLuint fb) {
		glActiveTexture(GL_TEXTURE0);
		if (flag) glBindTexture(GL_TEXTURE_2D, texture_id);
		glUniform1i(texUnitUniform, 0);
		glBindVertexArray(vao);
		glBindFramebuffer(GL_FRAMEBUFFER, fb);
		unsigned int* ptr = (unsigned int*)faces.request().ptr;

		GLuint elementBuffer;
		glGenBuffers(1, &elementBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_size * sizeof(unsigned int), &ptr[0], GL_STATIC_DRAW);
		glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, (void*)0);
		glDeleteBuffers(1, &elementBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

    void initvar_instance_group(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor) {
        glUseProgram(shaderProgram);
        float *Vptr = (float *) V.request().ptr;
        float *Pptr = (float *) P.request().ptr;
        float *lightposptr = (float *) lightpos.request().ptr;
        float *lightcolorptr = (float *) lightcolor.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
    }

    void init_material_pos_instance(int shaderProgram, py::array_t<float> pose_trans, py::array_t<float> pose_rot, float instance_color, py::array_t<float> diffuse_color, float use_texture) {
        float *transptr = (float *) pose_trans.request().ptr;
        float *rotptr = (float *) pose_rot.request().ptr;
        float *diffuse_ptr = (float *) diffuse_color.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
        glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
        glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
    }


    void render_tensor_pre(bool msaa, GLuint fb1, GLuint fb2) {

        glBindFramebuffer(GL_FRAMEBUFFER, fb2);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (msaa) {
            glBindFramebuffer(GL_FRAMEBUFFER, fb1);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        glEnable(GL_DEPTH_TEST);
    }


    void render_tensor_post() {
        glDisable(GL_DEPTH_TEST);
    }

    void cglBindVertexArray(int vao) {
        glBindVertexArray(vao);
    }

    void cglUseProgram(int shaderProgram) {
        glUseProgram(shaderProgram);
    }

    int loadTexture(std::string filename) {
        //width, height = img.size
        // glTexImage2D expects the first element of the image data to be the
        // bottom-left corner of the image.  Subsequent elements go left to right,
        // with subsequent lines going from bottom to top.

        // However, the image data was created with PIL Image tostring and numpy's
        // fromstring, which means we have to do a bit of reorganization. The first
        // element in the data output by tostring() will be the top-left corner of
        // the image, with following values going left-to-right and lines going
        // top-to-bottom.  So, we need to flip the vertical coordinate (y).

        int w;
        int h;
        int comp;
        stbi_set_flip_vertically_on_load(true);
        unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb);

        if(image == nullptr)
            throw(std::string("ERROR: Failed to load texture"));


        GLuint texture;
        glGenTextures(1, &texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                        GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(image);
        return texture;
    }

	// Generates large and small array textures and returns handles to the user (cutoff based on user variable), as well as index - tex num/layer mapping
	py::list generateArrayTextures(std::vector<std::string> filenames, int texCutoff, bool shouldShrinkSmallTextures, int smallTexBucketSize) {
		int num_textures = filenames.size();
		std::vector<unsigned char*> image_data;
		std::vector<int> texHeights;
		std::vector<int> texWidths;
		std::vector<int> texChannels;

		printf("number of textures %d\n", num_textures);
		for (int i = 0; i < num_textures; i++) {
			std::string filename = filenames[i];
			int w;
			int h;
			int comp;
			stbi_set_flip_vertically_on_load(true);
			unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb); // force to 3 channels
			if (image == nullptr)
				throw(std::string("Failed to load texture"));
			comp = 3;
			image_data.push_back(image);
			texHeights.push_back(h);
			texWidths.push_back(w);
			texChannels.push_back(comp);
		}


		GLuint texId1, texId2;
		glGenTextures(1, &texId1);
		glGenTextures(1, &texId2);

		py::list texInfo;
		py::list texLayerData;

		// Larger textures
		int firstTexLayerNum = 0;
		// Smaller textures
		int secondTexLayerNum = 0;

		std::vector<std::vector<int>> texIndices;
		std::vector<int> firstTexIndices, secondTexIndices;
		texIndices.push_back(firstTexIndices);
		texIndices.push_back(secondTexIndices);

		// w1, h1, w2, h2
		std::vector<int> texLayerDims;
		for (int i = 0; i < 4; i++) {
			texLayerDims.push_back(0);
		}

		for (int i = 0; i < image_data.size(); i++) {
			// Figure out if this texture goes in left group or right group based on w * h
			int w = texWidths[i];
			int h = texHeights[i];
			int score = w * h;

			py::list tex_info_i;

			// Texture goes in larger bucket if larger than cutoff
			if (score >= texCutoff) {
				texIndices[0].push_back(i);
				tex_info_i.append(0);
				tex_info_i.append(firstTexLayerNum);
				if (w > texLayerDims[0]) texLayerDims[0] = w;
				if (h > texLayerDims[1]) texLayerDims[1] = h;
				firstTexLayerNum++;
			}
			else {
				texIndices[1].push_back(i);
				tex_info_i.append(1);
				tex_info_i.append(secondTexLayerNum);
				if (w > texLayerDims[2]) texLayerDims[2] = w;
				if (h > texLayerDims[3]) texLayerDims[3] = h;
				secondTexLayerNum++;
			}

			texLayerData.append(tex_info_i);
		}

		printf("Texture 1 is w:%d by h:%d by depth:%d 3D array texture. ID %d\n", texLayerDims[0], texLayerDims[1], firstTexLayerNum, texId1);
		if (shouldShrinkSmallTextures) {
			printf("Texture 2 is w:%d by h:%d by depth:%d 3D array texture. ID %d\n", smallTexBucketSize, smallTexBucketSize, secondTexLayerNum, texId2);
		}
		else {
			printf("Texture 2 is w:%d by h:%d by depth:%d 3D array texture. ID %d\n", texLayerDims[2], texLayerDims[3], secondTexLayerNum, texId2);
		}

		for (int i = 0; i < 2; i++) {
			GLuint currTexId = texId1;
			if (i == 1) currTexId = texId2;

			glBindTexture(GL_TEXTURE_2D_ARRAY, currTexId);

			// Print texture array data
			if (i == 0) {
				GLint max_layers, max_size;
				glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &max_layers);
				glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_size);
				printf("Max layer number: %d\n", max_layers);
				printf("Max texture size: %d\n", max_size);
			}

			int layerNum = firstTexLayerNum;
			if (i == 1) layerNum = secondTexLayerNum;

			int out_w = texLayerDims[2 * i];
			int out_h = texLayerDims[2 * i + 1];

			// Gibson tends to have many more smaller textures, so we reduce their size to avoid memory overload
			if (i == 1 && shouldShrinkSmallTextures) {
				out_w = smallTexBucketSize;
				out_h = smallTexBucketSize;
			}

			// Deal with empty texture - create placeholder
			if (out_w == 0 || out_h == 0 || layerNum == 0) {
				glTexImage3D(GL_TEXTURE_2D_ARRAY,
					0,
					GL_RGB,
					1,
					1,
					1,
					0,
					GL_RGB,
					GL_UNSIGNED_BYTE,
					NULL
				);
			}
 
			glTexImage3D(GL_TEXTURE_2D_ARRAY,
				0,
				GL_RGB,
				out_w,
				out_h,
				layerNum,
				0,
				GL_RGB,
				GL_UNSIGNED_BYTE,
				NULL
			);

			// Add all textures in texture array i to that array texture
			for (int j = 0; j < layerNum; j++) {

				int idx = texIndices[i][j];

				int orig_w = texWidths[idx];
				int orig_h = texHeights[idx];
				int n_channels = texChannels[idx];
				unsigned char* input_data = image_data[idx];
				unsigned char* tex_bytes = input_data;
				bool shouldResize = (orig_w != out_w || orig_h != out_h);
				// Resize image to fit biggest texture in texture array
				if (shouldResize) {
					unsigned char* output_data = (unsigned char*)malloc(out_w * out_h * n_channels);
					stbir_resize_uint8(input_data, orig_w, orig_h, 0, output_data, out_w, out_h, 0, n_channels);
					tex_bytes = output_data;
				}

				glBindTexture(GL_TEXTURE_2D_ARRAY, currTexId);
				glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
					0,
					0,
					0,
					j,
					out_w,
					out_h,
					1,
					GL_RGB,
					GL_UNSIGNED_BYTE,
					tex_bytes
				);

				stbi_image_free(input_data);
				if (shouldResize) {
					free(tex_bytes);
				}
			}

			glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
			glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}

		texInfo.append(texId1);
		texInfo.append(texId2);
		texInfo.append(texLayerData);

		// texid1, texid2, list of idx - texid/layer data
		return texInfo;
	}

	// Performs optimized render setup
	py::list renderSetup(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor,
		py::array_t<float> mergedVertexData, py::array_t<int> index_ptr_offsets, py::array_t<int> index_counts,
		py::array_t<int> indices, py::array_t<float> mergedFragData, py::array_t<float> mergedDiffuseData,
		int tex_id_1, int tex_id_2, GLuint fb) {
		// First set up VAO and corresponding attributes
		GLuint VAO;
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		GLuint EBO;
		glGenBuffers(1, &EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		int* indicesPtr = (int*)indices.request().ptr;
		std::vector<unsigned int> indexData;
		for (int i = 0; i < indices.size(); i++) {
			indexData.push_back((unsigned int)indicesPtr[i]);
		}

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData.size() * sizeof(unsigned int), &indexData[0], GL_STATIC_DRAW);

		GLuint VBO;
		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		float* mergedVertexDataPtr = (float*)mergedVertexData.request().ptr;
		glBufferData(GL_ARRAY_BUFFER, mergedVertexData.size() * sizeof(float), mergedVertexDataPtr, GL_STATIC_DRAW);
		GLuint positionAttrib = glGetAttribLocation(shaderProgram, "position");
		GLuint normalAttrib = glGetAttribLocation(shaderProgram, "normal");
		GLuint texcoordAttrib = glGetAttribLocation(shaderProgram, "texCoords");
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
		glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)12);
		glVertexAttribPointer(texcoordAttrib, 2, GL_FLOAT, GL_TRUE, 32, (void*)24);

		glBindVertexArray(0);

		multidrawCount = index_ptr_offsets.size();
		int* indexOffsetPtr = (int*)index_ptr_offsets.request().ptr;

		for (int i = 0; i < multidrawCount; i++) {
			unsigned int offset = (unsigned int)indexOffsetPtr[i];
			this->multidrawStartIndices.push_back(BUFFER_OFFSET((offset * sizeof(unsigned int))));
			printf("multidraw start idx %d\n", offset);
		}

		// Store for rendering
		int* indices_count_ptr = (int*)index_counts.request().ptr;
		for (int i = 0; i < multidrawCount; i++) {
			this->multidrawCounts.push_back(indices_count_ptr[i]);
		}

		// Set up shaders
		float* fragData = (float*)mergedFragData.request().ptr;
		float* diffuseData = (float*)mergedDiffuseData.request().ptr;
		int fragDataSize = mergedFragData.size();
		int diffuseDataSize = mergedDiffuseData.size();

		glUseProgram(shaderProgram);

		float* Vptr = (float*)V.request().ptr;
		float* Pptr = (float*)P.request().ptr;
		float* lightposptr = (float*)lightpos.request().ptr;
		float* lightcolorptr = (float*)lightcolor.request().ptr;
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);

		glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
		glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
		printf("multidrawcount %d\n", multidrawCount);

		glGenBuffers(1, &uboTexColorData);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTexColorData);
		texColorDataSize = 2 * 16 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, texColorDataSize, NULL, GL_STATIC_DRAW);
		GLuint texColorDataIdx = glGetUniformBlockIndex(shaderProgram, "TexColorData");
		glUniformBlockBinding(shaderProgram, texColorDataIdx, 0);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboTexColorData);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, fragDataSize * sizeof(float), fragData);
		glBufferSubData(GL_UNIFORM_BUFFER, texColorDataSize / 2, diffuseDataSize * sizeof(float), diffuseData);

		glGenBuffers(1, &uboTransformData);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformData);
		transformDataSize = 2 * 64 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, transformDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint transformDataIdx = glGetUniformBlockIndex(shaderProgram, "TransformData");
		glUniformBlockBinding(shaderProgram, transformDataIdx, 1);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboTransformData);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		GLuint bigTexLoc = glGetUniformLocation(shaderProgram, "bigTex");
		GLuint smallTexLoc = glGetUniformLocation(shaderProgram, "smallTex");

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id_1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id_2);

		glUniform1i(bigTexLoc, 0);
		glUniform1i(smallTexLoc, 1);

		glUseProgram(0);

		py::list renderData;
		renderData.append(VAO);
		renderData.append(VBO);
		renderData.append(EBO);

		return renderData;
	}

	// Updates positions and rotations in vertex shader
	void updateDynamicData(int shaderProgram, py::array_t<float> pose_trans_array, py::array_t<float> pose_rot_array, py::array_t<float> V, py::array_t<float> P) {
		glUseProgram(shaderProgram);

		float* transPtr = (float*)pose_trans_array.request().ptr;
		float* rotPtr = (float*)pose_rot_array.request().ptr;
		int transDataSize = pose_trans_array.size();
		int rotDataSize = pose_rot_array.size();

		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformData);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, transDataSize * sizeof(float), transPtr);
		glBufferSubData(GL_UNIFORM_BUFFER, transformDataSize / 2, rotDataSize * sizeof(float), rotPtr);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		float* Vptr = (float*)V.request().ptr;
		float* Pptr = (float*)P.request().ptr;
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
	}

	// Optimized rendering function that is called once per frame for all merged data
	void renderOptimized(GLuint VAO) {
		glBindVertexArray(VAO);
		glMultiDrawElements(GL_TRIANGLES, &this->multidrawCounts[0], GL_UNSIGNED_INT, &this->multidrawStartIndices[0], this->multidrawCount);
	}
};

// Class containing values and functions needed to run VR around Gibson renderer
// This class uses the OpenGL coordinate system, and converts to Gibson's ROS coordinate system upon returning matrices
class VRSystem {
public:
	vr::IVRSystem* m_pHMD;
	uint32_t renderWidth;
	uint32_t renderHeight;
	float nearClip;
	float farClip;

	// Vector indicating the user-defined offset for the VR system (may be used if implementing a teleportation movement scheme, for example)
	glm::vec3 vrOffsetVec;

	// Device data stored in VR coordinates
	struct DeviceData {
		// standard 4x4 transform
		glm::mat4 deviceTransform;
		// x,y,z
		glm::vec3 devicePos;
		// w, x, y, z (quaternion)
		glm::vec4 deviceRot;
		// is device valid and being tracked
		bool isValidData = false;
		// index of current device in device array
		int index = -1;
		// trigger pressed fraction (0 min, 1 max) - controllers only!
		float trig_frac;
	    // analog touch vector - controllers only!
		glm::vec2 touchpad_analog_vec;
		// both indices are used to obtain analog data for trigger and touchpadd - controllers only!
		int trigger_axis_index;
		int touchpad_axis_index;
	};

	DeviceData hmdData;
	DeviceData leftControllerData;
	DeviceData rightControllerData;

	// Indicates where the headset actually is in the room
	glm::vec3 hmdActualPos;

	// View matrices for both left and right eyes (only proj and view are actually returned to the user)
	glm::mat4 leftEyeProj;
	glm::mat4 leftEyePos;
	glm::mat4 leftEyeView;
	glm::mat4 rightEyeProj;
	glm::mat4 rightEyePos;
	glm::mat4 rightEyeView;

	glm::mat4 gibToVR;
	glm::mat4 vrToGib;

	// SRAnipal variables
	bool useEyeTracking;
	std::thread* eyeTrackingThread;
	ViveSR::anipal::Eye::EyeData eyeData;
	int result;
	bool shouldShutDownEyeTracking;

	struct EyeTrackingData {
		glm::vec3 origin;
		glm::vec3 dir;
		glm::vec3 gazePoint;
		float leftPupilDiameter;
		float rightPupilDiameter;
	};

	EyeTrackingData eyeTrackingData;

	// Initialize VRSystem class
	// TIMELINE: Call before any other method in this class
	VRSystem() :m_pHMD(NULL), renderWidth(0), renderHeight(0), nearClip(0.1f), farClip(30.0f) {};

	// Initialize the VR system and compositor and return recommended dimensions
	// TIMELINE: Call during init of renderer, before height/width are set
	py::list initVR(bool useEyeTracking) {
		// Initialize VR systems
		if (!vr::VR_IsRuntimeInstalled()) {
			fprintf(stderr, "VR runtime not installed.\n");
			exit(EXIT_FAILURE);
		}

		vr::EVRInitError eError = vr::VRInitError_None;
		m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);

		if (eError != vr::VRInitError_None) {
			fprintf(stderr, "Unable to initialize VR runtime.\n");
			exit(EXIT_FAILURE);
		}

		if (!vr::VRCompositor()) {
			fprintf(stderr, "Unable to intialize VR compositor.\n");
		}

		leftEyeProj = getHMDEyeProjection(vr::Eye_Left);
		leftEyePos = getHMDEyePose(vr::Eye_Left);
		rightEyeProj = getHMDEyeProjection(vr::Eye_Right);
		rightEyePos = getHMDEyePose(vr::Eye_Right);

		m_pHMD->GetRecommendedRenderTargetSize(&renderWidth, &renderHeight);

		py::list renderDims;
		renderDims.append((int)renderWidth);
		renderDims.append((int)renderHeight);

		// Set gibToVR and vrToGib matrices
		setCoordinateTransformMatrices();
		// No VR system offset by default
		vrOffsetVec = glm::vec3(0, 0, 0);

		// Set eye tracking boolean
		this->useEyeTracking = useEyeTracking;
		if (useEyeTracking) {
			initAnipal();
			shouldShutDownEyeTracking = false;
		}

		return renderDims;
	}

	// Queries eye tracking data and returns to user
	// Returns in order gaze origin, gaze direction, gaze point, left pupil diameter (in mm), right pupil diameter (in mm)
	// TIMELINE: Can call any time
	py::list getEyeTrackingData() {
		py::list eyeData;

		// Transform data into Gibson coordinate system before returning to user
		glm::vec3 gibOrigin(vrToGib * glm::vec4(eyeTrackingData.origin, 1.0));
		glm::vec3 gibDir(vrToGib * glm::vec4(eyeTrackingData.dir, 1.0));
		glm::vec3 gibGazePoint(vrToGib * glm::vec4(eyeTrackingData.gazePoint, 1.0));

		py::list origin;
		origin.append(eyeTrackingData.origin.x);
		origin.append(eyeTrackingData.origin.y);
		origin.append(eyeTrackingData.origin.z);

		py::list dir;
		dir.append(gibDir.x);
		dir.append(gibDir.y);
		dir.append(gibDir.z);

		py::list gazePoint;
		gazePoint.append(gibGazePoint.x);
		gazePoint.append(gibGazePoint.y);
		gazePoint.append(gibGazePoint.z);

		eyeData.append(origin);
		eyeData.append(dir);
		eyeData.append(gazePoint);
		eyeData.append(eyeTrackingData.leftPupilDiameter);
		eyeData.append(eyeTrackingData.rightPupilDiameter);

		return eyeData;
	}

	// Sets the position of the VR headset
	// TIMELINE: Can call any time
	void setVRPosition(float x, float y, float z) {
		this->vrOffsetVec = glm::vec3(x, y, z);
	}

	// Returns the projection and view matrices for the left and right eyes, to be used in rendering
	// Returns in order Left P, left V, right P, right V
	// Note: GLM is column-major, whereas numpy is row major, so we need to tranpose view matrices before conversion
	// Note 2: Projection matrices are passed in to OpenGL assuming they are column-major, so we don't need to transpose them
	// TIMELINE: Call before rendering so the camera is set properly
	py::list preRenderVR() {
		py::array_t<float> leftEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(leftEyeProj));
		py::array_t<float> rightEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(rightEyeProj));

		glm::mat4 worldToHead = glm::inverse(hmdData.deviceTransform);

		leftEyeView = leftEyePos * worldToHead * gibToVR;
		rightEyeView = rightEyePos * worldToHead * gibToVR;

		py::array_t<float> leftEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(leftEyeView)));
		py::array_t<float> rightEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(rightEyeView)));

		py::list eyeMats;
		eyeMats.append(leftEyeProjNp);
		eyeMats.append(leftEyeViewNp);
		eyeMats.append(rightEyeProjNp);
		eyeMats.append(rightEyeViewNp);

		return eyeMats;
	}

	// Called after the renderer has finished rendering a single eye
	// TIMELINE: Call immediately after rendering for current eye is finished
	void postRenderVRForEye(char* eye, GLuint texID) {
		if (!strcmp(eye, "left")) {
			vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::EVRCompositorError err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
			// 0 is no error, 101 is no focus (happens at start of rendering)
			if (err != 0 && err != 101) {
				fprintf(stderr, "Compositor error: %d\n", err);
			}
		}
		else if (!strcmp(eye, "right")) {
			vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };;
			vr::EVRCompositorError err = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
			if (err != 0 && err != 101) {
				fprintf(stderr, "Compositor error: %d\n", err);
			}
		}
	}

	// Called after both eyes have been rendered 
	// Tell the compositor to begin work immediately instead of waiting for the next WaitGetPoses() call if the user wants
	// And then update VR data
	// TIMELINE: Call immediately after calling postRenderVRForEye on both left and right eyes
	void postRenderVRUpdate(bool shouldHandoff) {
		if (shouldHandoff) {
			vr::VRCompositor()->PostPresentHandoff();
		}

		updateVRData();
	}

	// Returns device data in order: isValidData, position, rotation, hmdActualPos (valid only if hmd)
	// Device type can be either hmd, left_controller or right_controller
	// TIMELINE: Call at any time after postRenderVR to poll the VR system for device data
	py::list getDataForVRDevice(char* deviceType) {
		bool isValid = false;

		py::array_t<float> positionData;
		py::array_t<float> rotationData;
		py::array_t<float> hmdActualPosData;

		// TODO: Extend this to work with multiple headsets in future
		if (!strcmp(deviceType, "hmd")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(hmdData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * hmdData.deviceRot));
			glm::vec3 transformedHmdPos(vrToGib * glm::vec4(hmdActualPos, 1.0));
			hmdActualPosData = py::array_t<float>({ 3, }, glm::value_ptr(transformedHmdPos));
			isValid = hmdData.isValidData;
		}
		else if (!strcmp(deviceType, "left_controller")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(leftControllerData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * leftControllerData.deviceRot));
			isValid = leftControllerData.isValidData;
		}
		else if (!strcmp(deviceType, "right_controller")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(rightControllerData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * rightControllerData.deviceRot));
			isValid = rightControllerData.isValidData;
		}

		py::list deviceData;
		deviceData.append(isValid);
		deviceData.append(positionData);
		deviceData.append(rotationData);
		deviceData.append(hmdActualPosData);

		return deviceData;
	}

	// Get button data for a specific controller - either left_controller or right_controller
	// Returns in order: trigger fraction, analog touch position x, analog touch position y
	// TIMELINE: Call directly after getDataForVRDevice (relies on isValid to determine data integrity)
	py::list getButtonDataForController(char* controllerType) {
		float trigger_fraction, touch_x, touch_y;
		bool isValid;

		if (!strcmp(controllerType, "left_controller")) {
			trigger_fraction = leftControllerData.trig_frac;
			touch_x = leftControllerData.touchpad_analog_vec.x;
			touch_y = leftControllerData.touchpad_analog_vec.y;
			isValid = leftControllerData.isValidData;
		}
		else if (!strcmp(controllerType, "right_controller")) {
			trigger_fraction = rightControllerData.trig_frac;
			touch_x = rightControllerData.touchpad_analog_vec.x;
			touch_y = rightControllerData.touchpad_analog_vec.y;
			isValid = rightControllerData.isValidData;
		}

		py::list buttonData;
		buttonData.append(trigger_fraction);
		buttonData.append(touch_x);
		buttonData.append(touch_y);

		return buttonData;
	}

	// Polls for VR events, such as button presses
	// TIMELINE: Ideally call before rendering (eg. before simulator step function)
	py::list pollVREvents() {
		vr::VREvent_t vrEvent;
		py::list eventData;

		while (m_pHMD->PollNextEvent(&vrEvent, sizeof(vrEvent))) {
			std::string deviceType, eventType;
			processVREvent(vrEvent, deviceType, eventType);

			if (deviceType == "invalid" || eventType == "invalid") {
				continue;
			}

			py::list singleEventData;
			singleEventData.append(deviceType);
			singleEventData.append(eventType);

			eventData.append(singleEventData);
		}

		return eventData;
	}

	// Releases and cleans up VR system
	// TIMELINE: Call when the renderer shuts down
	void releaseVR() {
		vr::VR_Shutdown();
		m_pHMD = NULL;

		if (this->useEyeTracking) {
			this->shouldShutDownEyeTracking = true;
			eyeTrackingThread->join();
		}
	}

private:
	// Initializes the SRAnipal runtime, if the user selects this option
	void initAnipal() {
		if (!ViveSR::anipal::Eye::IsViveProEye()) {
			fprintf(stderr, "This HMD does not support eye-tracking!\n");
		}

		int anipalError = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
		switch (anipalError) {
		case ViveSR::Error::WORK:
			break;
		case ViveSR::Error::RUNTIME_NOT_FOUND:
			fprintf(stderr, "SRAnipal runtime not found!\n");
		default:
			fprintf(stderr, "Failed to initialize SRAnipal!\n");
		}

		// Launch a thread to poll data from the SRAnipal SDK
		// We poll data asynchronously so as to not slow down the VR rendering loop
		eyeTrackingThread = new std::thread(&VRSystem::pollAnipal, this);
	}

	// Polls SRAnipal to get updated eye tracking information
	void pollAnipal() {
		while (!this->shouldShutDownEyeTracking) {
			this->result = ViveSR::anipal::Eye::GetEyeData(&this->eyeData);
			if (result == ViveSR::Error::WORK) {
				int isOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
					ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
				int isDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
					ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
				if (!isOriginValid || !isDirValid) continue;

				auto gazeOrigin = this->eyeData.verbose_data.combined.eye_data.gaze_origin_mm;
				if (gazeOrigin.x != -1.0f || gazeOrigin.y != 1.0f || gazeOrigin.z != 1.0f)
					eyeTrackingData.origin = glm::vec3(gazeOrigin.x, gazeOrigin.y, gazeOrigin.z);
				auto gazeDirection = this->eyeData.verbose_data.combined.eye_data.gaze_direction_normalized;
				if (gazeDirection.x != -1.0f || gazeDirection.y != 1.0f || gazeDirection.z != 1.0f)
					eyeTrackingData.dir = glm::vec3(gazeDirection.x, gazeDirection.y, gazeDirection.z);

				// Calculate intersection point of two eyes
				auto leftGazeOrigin = this->eyeData.verbose_data.left.gaze_origin_mm;
				glm::vec3 lgo = glm::vec3(leftGazeOrigin.x, leftGazeOrigin.y, leftGazeOrigin.z);
				auto leftGazeDir = this->eyeData.verbose_data.left.gaze_direction_normalized;
				glm::vec3 lgd = glm::vec3(leftGazeDir.x, leftGazeDir.y, leftGazeDir.z);
				auto rightGazeOrigin = this->eyeData.verbose_data.right.gaze_origin_mm;
				glm::vec3 rgo = glm::vec3(rightGazeOrigin.x, rightGazeOrigin.y, rightGazeOrigin.z);
				auto rightGazeDir = this->eyeData.verbose_data.right.gaze_direction_normalized;
				glm::vec3 rgd = glm::vec3(rightGazeDir.x, rightGazeDir.y, rightGazeDir.z);

				// Solve for closest point to each of two gaze lines, which is the point the user is looking at
				// This is the midpoint of the shortest line segment between them
				float s = (glm::dot(lgd, rgd) * (glm::dot(lgo, rgd) - glm::dot(lgd, rgo)) - glm::dot(lgo, rgd) * glm::dot(rgo, rgd))
					/ ((glm::dot(lgo, rgd) * glm::dot(lgo, rgd)) - 1);
				float t = (glm::dot(lgd, rgd) * (glm::dot(rgo, rgd) - glm::dot(lgo, rgd)) - glm::dot(lgd, rgo) * glm::dot(lgo, lgd))
					/ ((glm::dot(lgo, rgd) * glm::dot(lgo, rgd)) - 1);

				eyeTrackingData.gazePoint = 0.5f * (lgo + rgo + t * lgd + s * rgd);
				// x coordinates are opposite of OpenGL convention
				eyeTrackingData.gazePoint.x *= -1;
				
				// Record pupil measurements
				eyeTrackingData.leftPupilDiameter = this->eyeData.verbose_data.left.pupil_diameter_mm;
				eyeTrackingData.rightPupilDiameter = this->eyeData.verbose_data.right.pupil_diameter_mm;
			}
		}
	}

	// Calls WaitGetPoses and updates all hmd and controller transformations
	void updateVRData() {
		hmdData.isValidData = false;
		leftControllerData.isValidData = false;
		rightControllerData.isValidData = false;
		// Stores controller information - see github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState for more info
		vr::VRControllerState_t controllerState;

		vr::TrackedDevicePose_t trackedDevices[vr::k_unMaxTrackedDeviceCount];
		vr::VRCompositor()->WaitGetPoses(trackedDevices, vr::k_unMaxTrackedDeviceCount, NULL, 0);

		for (unsigned int idx = 0; idx < vr::k_unMaxTrackedDeviceCount; idx++) {
			if (!trackedDevices[idx].bPoseIsValid || !m_pHMD->IsTrackedDeviceConnected(idx)) continue;

			vr::HmdMatrix34_t transformMat = trackedDevices[idx].mDeviceToAbsoluteTracking;
			vr::ETrackedDeviceClass trackedDeviceClass = m_pHMD->GetTrackedDeviceClass(idx);

			if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_HMD) {
				hmdData.index = idx;
				hmdData.isValidData = true;
				hmdActualPos = getPositionFromSteamVRMatrix(transformMat);

				setSteamVRMatrixPos(hmdActualPos + vrOffsetVec, transformMat);

				hmdData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
				hmdData.devicePos = getPositionFromSteamVRMatrix(transformMat);
				hmdData.deviceRot = getRotationFromSteamVRMatrix(transformMat);
			}
			else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
				vr::ETrackedControllerRole role = m_pHMD->GetControllerRoleForTrackedDeviceIndex(idx);
				if (role == vr::TrackedControllerRole_Invalid) {
					continue;
				}

				int trigger_index, touchpad_index;

				// Figures out indices that correspond with trigger and trackpad axes. Index used to read into VRControllerState_t struct array of axes.
				for (int i = 0; i < vr::k_unControllerStateAxisCount; i++) {
					int axisType = m_pHMD->GetInt32TrackedDeviceProperty(idx, (vr::ETrackedDeviceProperty)(vr::Prop_Axis0Type_Int32 + i));
					if (axisType == vr::EVRControllerAxisType::k_eControllerAxis_Trigger) {
						trigger_index = i;
					}
					else if (axisType == vr::EVRControllerAxisType::k_eControllerAxis_TrackPad) {
						touchpad_index = i;
					}
				}

				// If false, sets the controller data validity to false, as data is not valid if we can't read analog touch coordinates and trigger close fraction
				bool getControllerDataResult = m_pHMD->GetControllerState(idx, &controllerState, sizeof(controllerState));

				if (role == vr::TrackedControllerRole_LeftHand) {
					leftControllerData.index = idx;
					leftControllerData.trigger_axis_index = trigger_index;
					leftControllerData.touchpad_axis_index = touchpad_index;
					leftControllerData.isValidData = getControllerDataResult;

					glm::vec3 leftControllerPos = getPositionFromSteamVRMatrix(transformMat);
					setSteamVRMatrixPos(leftControllerPos + vrOffsetVec, transformMat);

					leftControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
					leftControllerData.devicePos = getPositionFromSteamVRMatrix(transformMat);
					leftControllerData.deviceRot = getRotationFromSteamVRMatrix(transformMat);

					leftControllerData.trig_frac = controllerState.rAxis[leftControllerData.trigger_axis_index].x;
					leftControllerData.touchpad_analog_vec = glm::vec2(controllerState.rAxis[leftControllerData.touchpad_axis_index].x, controllerState.rAxis[leftControllerData.touchpad_axis_index].y);
				}
				else if (role == vr::TrackedControllerRole_RightHand) {
					rightControllerData.index = idx;
					rightControllerData.trigger_axis_index = trigger_index;
					rightControllerData.touchpad_axis_index = touchpad_index;
					rightControllerData.isValidData = getControllerDataResult;

					glm::vec3 rightControllerPos = getPositionFromSteamVRMatrix(transformMat);
					setSteamVRMatrixPos(rightControllerPos + vrOffsetVec, transformMat);

					rightControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
					rightControllerData.devicePos = getPositionFromSteamVRMatrix(transformMat);
					rightControllerData.deviceRot = getRotationFromSteamVRMatrix(transformMat);

					rightControllerData.trig_frac = controllerState.rAxis[rightControllerData.trigger_axis_index].x;
					rightControllerData.touchpad_analog_vec = glm::vec2(controllerState.rAxis[rightControllerData.touchpad_axis_index].x, controllerState.rAxis[rightControllerData.touchpad_axis_index].y);
				}
			}
		}
	}

	// Processes a single VR event
	void processVREvent(vr::VREvent_t& vrEvent, std::string& deviceType, std::string& eventType) {
		vr::ETrackedDeviceClass trackedDeviceClass = m_pHMD->GetTrackedDeviceClass(vrEvent.trackedDeviceIndex);

		// Exit if we found a non-controller event
		if (trackedDeviceClass != vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
			deviceType = "invalid";
			return;
		}

		vr::ETrackedControllerRole role = m_pHMD->GetControllerRoleForTrackedDeviceIndex(vrEvent.trackedDeviceIndex);
		if (role == vr::TrackedControllerRole_Invalid) {
			deviceType = "invalid";
		}
		else if (role == vr::TrackedControllerRole_LeftHand) {
			deviceType = "left_controller";
		}
		else if (role == vr::TrackedControllerRole_RightHand) {
			deviceType = "right_controller";
		}

		switch (vrEvent.data.controller.button) {
		case vr::k_EButton_Grip:
			switch (vrEvent.eventType) {
			case vr::VREvent_ButtonPress:
				eventType = "grip_press";
				break;

			case vr::VREvent_ButtonUnpress:
				eventType = "grip_unpress";
				break;
			default:
				eventType = "invalid";
				break;
			}
			break;

		case vr::k_EButton_SteamVR_Trigger:
			switch (vrEvent.eventType) {
			case vr::VREvent_ButtonPress:
				eventType = "trigger_press";
				break;

			case vr::VREvent_ButtonUnpress:
				eventType = "trigger_unpress";
				break;
			default:
				eventType = "invalid";
				break;
			}
			break;

		case vr::k_EButton_SteamVR_Touchpad:
			switch (vrEvent.eventType) {
			case vr::VREvent_ButtonPress:
				eventType = "touchpad_press";
				break;

			case vr::VREvent_ButtonUnpress:
				eventType = "touchpad_unpress";
				break;

			case vr::VREvent_ButtonTouch:
				eventType = "touchpad_touch";
				break;

			case vr::VREvent_ButtonUntouch:
				eventType = "touchpad_untouch";
				break;
			default:
				eventType = "invalid";
				break;
			}
			break;

		case vr::k_EButton_ApplicationMenu:
			switch (vrEvent.eventType) {
			case vr::VREvent_ButtonPress:
				eventType = "menu_press";
				break;

			case vr::VREvent_ButtonUnpress:
				eventType = "menu_unpress";
				break;
			default:
				eventType = "invalid";
				break;
			}
			break;

		default:
			eventType = "invalid";
			break;
		}
	}

	// Sets the position component of a SteamVR Matrix
	void setSteamVRMatrixPos(glm::vec3& pos, vr::HmdMatrix34_t& mat) {
		mat.m[0][3] = pos[0];
		mat.m[1][3] = pos[1];
		mat.m[2][3] = pos[2];
	}

	// Converts a SteamVR Matrix to a glm mat4
	glm::mat4 convertSteamVRMatrixToGlmMat4(const vr::HmdMatrix34_t& matPose) {
		glm::mat4 mat(
			matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
			matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
			matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
			matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
		);
		return mat;
	}

	// Gets position of HMD
	glm::vec3 getPositionFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
		return glm::vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
	}

	// Gets rotation of HMD in vec4 form
	glm::vec4 getRotationFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
		glm::vec4 q;

		q.w = (float)sqrt(fmax(0, 1 + matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2])) / 2;
		q.x = (float)sqrt(fmax(0, 1 + matrix.m[0][0] - matrix.m[1][1] - matrix.m[2][2])) / 2;
		q.y = (float)sqrt(fmax(0, 1 - matrix.m[0][0] + matrix.m[1][1] - matrix.m[2][2])) / 2;
		q.z = (float)sqrt(fmax(0, 1 - matrix.m[0][0] - matrix.m[1][1] + matrix.m[2][2])) / 2;
		q.x = copysign(q.x, matrix.m[2][1] - matrix.m[1][2]);
		q.y = copysign(q.y, matrix.m[0][2] - matrix.m[2][0]);
		q.z = copysign(q.z, matrix.m[1][0] - matrix.m[0][1]);

		return q;
	}

	// Generates a projection matrix for the specified eye (left or right)
	glm::mat4 getHMDEyeProjection(vr::Hmd_Eye eye) {
		vr::HmdMatrix44_t mat = m_pHMD->GetProjectionMatrix(eye, nearClip, farClip);

		glm::mat4 eyeProjMat(
			mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
			mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
			mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
			mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
		);

		return eyeProjMat;
	}

	// Generates a pose matrix for the specified eye (left or right)
	glm::mat4 getHMDEyePose(vr::Hmd_Eye eye) {
		vr::HmdMatrix34_t mat = m_pHMD->GetEyeToHeadTransform(eye);

		glm::mat4 eyeToHead(
			mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0,
			mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
			mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0,
			mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
		);

		// Return the head to eye transform
		return glm::inverse(eyeToHead);
	}

	// Print string version of mat4 for debugging purposes
	void printMat4(glm::mat4& m) {
		printf(glm::to_string(m).c_str());
		printf("\n");
	}

	// Sets coordinate transform matrices
	void setCoordinateTransformMatrices() {
		gibToVR[0] = glm::vec4(0.0, 0.0, -1.0, 0.0);
		gibToVR[1] = glm::vec4(-1.0, 0.0, 0.0, 0.0);
		gibToVR[2] = glm::vec4(0.0, 1.0, 0.0, 0.0);
		gibToVR[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);

		vrToGib[0] = glm::vec4(0.0, -1.0, 0.0, 0.0);
		vrToGib[1] = glm::vec4(0.0, 0.0, 1.0, 0.0);
		vrToGib[2] = glm::vec4(-1.0, 0.0, 0.0, 0.0);
		vrToGib[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);
	}
};

PYBIND11_MODULE(MeshRendererContext, m) {
        py::class_<MeshRendererContext> pymodule = py::class_<MeshRendererContext>(m, "MeshRendererContext");
        
        pymodule.def(py::init<int, int>());
        pymodule.def("init", &MeshRendererContext::init, "initialize glfw window and context");
		pymodule.def("render_companion_window_from_buffer", &MeshRendererContext::render_companion_window_from_buffer, "blit color texture to default framebuffer and show companion window");
		pymodule.def("release", &MeshRendererContext::release, "release glfw context");

        // class MeshRenderer
        pymodule.def("render_meshrenderer_pre", &MeshRendererContext::render_meshrenderer_pre, "pre-executed functions in MeshRenderer.render");
        pymodule.def("render_meshrenderer_post", &MeshRendererContext::render_meshrenderer_post, "post-executed functions in MeshRenderer.render");
        pymodule.def("getstring_meshrenderer", &MeshRendererContext::getstring_meshrenderer, "return GL version string");
        pymodule.def("readbuffer_meshrenderer", &MeshRendererContext::readbuffer_meshrenderer, "read pixel buffer");
		pymodule.def("glad_init", &MeshRendererContext::glad_init, "init glad");
        pymodule.def("clean_meshrenderer", &MeshRendererContext::clean_meshrenderer, "clean meshrenderer");
        pymodule.def("setup_framebuffer_meshrenderer", &MeshRendererContext::setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");
        pymodule.def("setup_framebuffer_meshrenderer_ms", &MeshRendererContext::setup_framebuffer_meshrenderer_ms, "setup framebuffer in meshrenderer with MSAA");
        pymodule.def("blit_buffer", &MeshRendererContext::blit_buffer, "blit buffer");

        pymodule.def("compile_shader_meshrenderer", &MeshRendererContext::compile_shader_meshrenderer, "compile vertex and fragment shader");
        pymodule.def("load_object_meshrenderer", &MeshRendererContext::load_object_meshrenderer, "load object into VAO and VBO");
        pymodule.def("loadTexture", &MeshRendererContext::loadTexture, "load texture function");

        // class MeshRendererG2G
        pymodule.def("render_tensor_pre", &MeshRendererContext::render_tensor_pre, "pre-executed functions in MeshRendererG2G.render");
        pymodule.def("render_tensor_post", &MeshRendererContext::render_tensor_post, "post-executed functions in MeshRendererG2G.render");

        // class Instance
        pymodule.def("render_softbody_instance", &MeshRendererContext::render_softbody_instance, "render softbody in instance.render");
        pymodule.def("initvar_instance", &MeshRendererContext::initvar_instance, "init uniforms in instance.render");
        pymodule.def("init_material_instance", &MeshRendererContext::init_material_instance, "init materials in instance.render");
        pymodule.def("draw_elements_instance", &MeshRendererContext::draw_elements_instance, "draw elements in instance.render and instancegroup.render");

        // class InstanceGroup
        pymodule.def("initvar_instance_group", &MeshRendererContext::initvar_instance_group, "init uniforms in instancegroup.render");
        pymodule.def("init_material_pos_instance", &MeshRendererContext::init_material_pos_instance, "init materials and position in instancegroup.render");

        // misc
        pymodule.def("cglBindVertexArray", &MeshRendererContext::cglBindVertexArray, "binding function");
        pymodule.def("cglUseProgram", &MeshRendererContext::cglUseProgram, "binding function");

		//renderer optimization
		pymodule.def("generateArrayTextures", &MeshRendererContext::generateArrayTextures, "generate array texture function");
		pymodule.def("renderSetup", &MeshRendererContext::renderSetup, "loads all merged graphics data");
		pymodule.def("updateDynamicData", &MeshRendererContext::updateDynamicData, "updates dynamic data such as object transforms");
		pymodule.def("renderOptimized", &MeshRendererContext::renderOptimized, "renders merged data in an optimized way");
		pymodule.def("clean_meshrenderer_optimized", &MeshRendererContext::clean_meshrenderer_optimized, "clean optimized mesh renderer");

		// VR pymodule needs to be part of MeshRendererContext for OpenGL context-sharing reasons
		py::class_<VRSystem> pymoduleVR = py::class_<VRSystem>(m, "VRSystem");

		pymoduleVR.def(py::init());
		pymoduleVR.def("initVR", &VRSystem::initVR);
		pymoduleVR.def("getEyeTrackingData", &VRSystem::getEyeTrackingData);
		pymoduleVR.def("setVRPosition", &VRSystem::setVRPosition);
		pymoduleVR.def("preRenderVR", &VRSystem::preRenderVR);
		pymoduleVR.def("postRenderVRForEye", &VRSystem::postRenderVRForEye);
		pymoduleVR.def("postRenderVRUpdate", &VRSystem::postRenderVRUpdate);
		pymoduleVR.def("getDataForVRDevice", &VRSystem::getDataForVRDevice);
		pymoduleVR.def("getButtonDataForController", &VRSystem::getButtonDataForController);
		pymoduleVR.def("pollVREvents", &VRSystem::pollVREvents);
		pymoduleVR.def("releaseVR", &VRSystem::releaseVR);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
