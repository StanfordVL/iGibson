#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
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

#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_gl_interop.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace py = pybind11;


class GLFWRendererContext {
public:
    GLFWRendererContext(int w, int h) :m_windowHeight(h), m_windowWidth(w) {};

    int m_windowWidth;
    int m_windowHeight;
    GLFWwindow* window = NULL;

    int verbosity;

    int init() {
        verbosity = 20;

        // Initialize GLFW context and window
        if (!glfwInit()) {
            fprintf(stderr, "ERROR: Failed to initialize GLFW.\n");
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        // Hide GLFW window by default
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        this->window = glfwCreateWindow(m_windowHeight, m_windowHeight, "Gibson VR Renderer", NULL, NULL);
        if (this->window == NULL) {
            fprintf(stderr, "ERROR: Failed to create GLFW window.\n");

            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(this->window);

        // Load all OpenGL function pointers through GLAD
        if (!gladLoadGL(glfwGetProcAddress))
        {
            fprintf(stderr, "ERROR: Failed to load OpenGL function pointers through GLAD.\n");
            exit(EXIT_FAILURE);
        }

        if(verbosity >= 20) { printf("INFO: Succesfully initialized GLFW context and window!\n");}

        return 0;
    }

    void release() {
        glfwTerminate();
    }


#ifdef USE_CUDA
    void map_tensor(GLuint tid, int width, int height, std::size_t data)
    {
       cudaError_t err;
       if (cuda_res[tid] == NULL)
       {
         err = cudaGraphicsGLRegisterImage(&(cuda_res[tid]), tid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
         if( err != cudaSuccess )
         {
           std::cout << "WARN: cudaGraphicsGLRegisterImage failed: " << err << std::endl;
         }
       }

       err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsMapResources failed: " << err << std::endl;
       }

       cudaArray* array;
       err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
       }

       // copy data
       err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(char), array, 0, 0, width*4*sizeof(char), height, cudaMemcpyDeviceToDevice);
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaMemcpy2DFromArray failed: " << err << std::endl;
       }

       err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsUnmapResources failed: " << err << std::endl;
       }
    }

    void map_tensor_float(GLuint tid, int width, int height, std::size_t data)
    {
       cudaError_t err;
       if (cuda_res[tid] == NULL)
       {
         err = cudaGraphicsGLRegisterImage(&(cuda_res[tid]), tid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
         if( err != cudaSuccess )
         {
           std::cout << "WARN: cudaGraphicsGLRegisterImage failed: " << err << std::endl;
         }
       }

       err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsMapResources failed: " << err << std::endl;
       }

       cudaArray* array;
       err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
       }

       // copy data
       err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(float), array, 0, 0, width*4*sizeof(float), height, cudaMemcpyDeviceToDevice);
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaMemcpy2DFromArray failed: " << err << std::endl;
       }

       err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "WARN: cudaGraphicsUnmapResources failed: " << err << std::endl;
       }
    }
#endif

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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
            fprintf(stderr, "ERROR: unknown buffer mode.\n");
            exit(EXIT_FAILURE);
        }
        py::array_t<float> data = py::array_t<float>(4 * width * height);
        py::buffer_info buf = data.request();
        float* ptr = (float *) buf.ptr;
        glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, ptr);
        return data;
    }

    py::array_t<float> readbuffer_meshrenderer_shadow_depth(int width, int height, GLuint fb2, GLuint texture_id) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb2);
        glReadBuffer(GL_COLOR_ATTACHMENT3);
        py::array_t<float> data = py::array_t<float>(3 * width * height);
        py::buffer_info buf = data.request();
        float* ptr = (float *) buf.ptr;
        glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, ptr);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, ptr);
        return data;
    }

    void clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2, std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos) {
        glDeleteTextures(texture1.size(), texture1.data());
        glDeleteTextures(texture2.size(), texture2.data());
        glDeleteFramebuffers(fbo.size(), fbo.data());
        glDeleteBuffers(vaos.size(), vaos.data());
        glDeleteBuffers(vbos.size(), vbos.data());
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
        py::list result;
        result.append(shaderProgram);
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

    void initvar_instance(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV, int enable_shadow, py::array_t<float> P,
                          py::array_t<float> pose_trans, py::array_t<float> pose_rot, py::array_t<float> lightpos, py::array_t<float> lightcolor) {
        glUseProgram(shaderProgram);
        float *Vptr = (float *) V.request().ptr;
        float *lightVptr = (float *) lightV.request().ptr;
        float *Pptr = (float *) P.request().ptr;
        float *transptr = (float *) pose_trans.request().ptr;
        float *rotptr = (float *) pose_rot.request().ptr;
        float *lightposptr = (float *) lightpos.request().ptr;
        float *lightcolorptr = (float *) lightcolor.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
        glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), enable_shadow);
    }

    void init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color, int
    use_texture) {
        float *diffuse_ptr = (float *) diffuse_color.request().ptr;
        glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
        glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
        glUniform1i(glGetUniformLocation(shaderProgram, "use_texture"), (GLint)use_texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "texUnit"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 1);
    }

    void draw_elements_instance(bool flag, int texture_id, int depth_texture_id, int vao, int
    face_size, py::array_t<unsigned int> faces, GLuint fb) {
        glActiveTexture(GL_TEXTURE0);
        if (flag) glBindTexture(GL_TEXTURE_2D, texture_id);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depth_texture_id);

        glBindVertexArray(vao);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        unsigned int *ptr = (unsigned int *) faces.request().ptr;

        GLuint elementBuffer;
        glGenBuffers(1, &elementBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_size * sizeof(unsigned int), &ptr[0], GL_STATIC_DRAW);
        glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, (void*)0);
        glDeleteBuffers(1, &elementBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);

    }

    void initvar_instance_group(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV, int enable_shadow, py::array_t<float>
    P, py::array_t<float>
                                lightpos, py::array_t<float> lightcolor) {
        glUseProgram(shaderProgram);
        float *Vptr = (float *) V.request().ptr;
        float *lightVptr = (float *) lightV.request().ptr;
        float *Pptr = (float *) P.request().ptr;
        float *lightposptr = (float *) lightpos.request().ptr;
        float *lightcolorptr = (float *) lightcolor.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
        glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
        glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), enable_shadow);
    }

    void init_material_pos_instance(int shaderProgram, py::array_t<float> pose_trans, py::array_t<float> pose_rot, float
    instance_color, py::array_t<float> diffuse_color, int use_texture) {
        float *transptr = (float *) pose_trans.request().ptr;
        float *rotptr = (float *) pose_rot.request().ptr;
        float *diffuse_ptr = (float *) diffuse_color.request().ptr;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
        glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
        glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
        glUniform1i(glGetUniformLocation(shaderProgram, "use_texture"), (GLint)use_texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "texUnit"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 1);
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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void cglBindVertexArray(int vao) {
        glBindVertexArray(vao);
    }

    void cglUseProgram(int shaderProgram) {
        glUseProgram(shaderProgram);
    }

    int loadTexture(std::string filename) {

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

    int allocateTexture(int w, int h) {
        GLuint texture;
        glGenTextures(1, &texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGBA,
                     GL_FLOAT, NULL);
        return texture;
    }


};

PYBIND11_MODULE(GLFWRendererContext, m) {

    py::class_<GLFWRendererContext> pymodule = py::class_<GLFWRendererContext>(m, "GLFWRendererContext");

    pymodule.def(py::init<int, int>());
    pymodule.def("init", &GLFWRendererContext::init);
    pymodule.def("release", &GLFWRendererContext::release);

    // class MeshRenderer
    pymodule.def("render_meshrenderer_pre", &GLFWRendererContext::render_meshrenderer_pre, "pre-executed functions in MeshRenderer.render");
    pymodule.def("render_meshrenderer_post", &GLFWRendererContext::render_meshrenderer_post, "post-executed functions in MeshRenderer.render");
    pymodule.def("getstring_meshrenderer", &GLFWRendererContext::getstring_meshrenderer, "return GL version string");
    pymodule.def("readbuffer_meshrenderer", &GLFWRendererContext::readbuffer_meshrenderer, "read pixel buffer");
    pymodule.def("readbuffer_meshrenderer_shadow_depth", &GLFWRendererContext::readbuffer_meshrenderer_shadow_depth,
    "read pixel buffer");
    pymodule.def("clean_meshrenderer", &GLFWRendererContext::clean_meshrenderer, "clean meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer", &GLFWRendererContext::setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer_ms", &GLFWRendererContext::setup_framebuffer_meshrenderer_ms, "setup framebuffer in meshrenderer with MSAA");
    pymodule.def("blit_buffer", &GLFWRendererContext::blit_buffer, "blit buffer");

    pymodule.def("compile_shader_meshrenderer", &GLFWRendererContext::compile_shader_meshrenderer, "compile vertex and fragment shader");
    pymodule.def("load_object_meshrenderer", &GLFWRendererContext::load_object_meshrenderer, "load object into VAO and VBO");
    pymodule.def("loadTexture", &GLFWRendererContext::loadTexture, "load texture function");
    pymodule.def("allocateTexture", &GLFWRendererContext::allocateTexture, "load texture function");

    // class Instance
    pymodule.def("render_softbody_instance", &GLFWRendererContext::render_softbody_instance, "render softbody in instance.render");
    pymodule.def("initvar_instance", &GLFWRendererContext::initvar_instance, "init uniforms in instance.render");
    pymodule.def("init_material_instance", &GLFWRendererContext::init_material_instance, "init materials in instance.render");
    pymodule.def("draw_elements_instance", &GLFWRendererContext::draw_elements_instance, "draw elements in instance.render and instancegroup.render");

    // class InstanceGroup
    pymodule.def("initvar_instance_group", &GLFWRendererContext::initvar_instance_group, "init uniforms in instancegroup.render");
    pymodule.def("init_material_pos_instance", &GLFWRendererContext::init_material_pos_instance, "init materials and position in instancegroup.render");

    // misc
    pymodule.def("cglBindVertexArray", &GLFWRendererContext::cglBindVertexArray, "binding function");
    pymodule.def("cglUseProgram", &GLFWRendererContext::cglUseProgram, "binding function");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}