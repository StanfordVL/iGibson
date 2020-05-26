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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define BUFFER_OFFSET(offset) (static_cast<char*>(0) + (offset))


namespace py = pybind11;
#define MAX_ARRAY_SIZE 512

class GLFWRendererContext {
public:
    GLFWRendererContext(int w, int h) :m_windowHeight(h), m_windowWidth(w) {};

    int m_windowWidth;
    int m_windowHeight;
    GLFWwindow* window = NULL;

    int verbosity;

    // Index data
    void* multidrawStartIndices[MAX_ARRAY_SIZE];
    int multidrawCounts[MAX_ARRAY_SIZE];
    int multidrawCount;
    //std::vector<void*> startIndices;

    // UBO data
    GLuint uboTexColorData;
    GLuint uboTransformData;
    int texColorDataSize;
    int transformDataSize;

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
        unsigned int *ptr = (unsigned int *) faces.request().ptr;

        GLuint elementBuffer;
        glGenBuffers(1, &elementBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_size * sizeof(unsigned int), &ptr[0], GL_STATIC_DRAW);
        glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, (void*)0);
        glDeleteBuffers(1, &elementBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);

    }


    void draw_elements_instance_optimized(int shaderProgram, bool flag, int texture_id, int texture_id2, int texture_bucket,
            int texture_layer, int texUnitUniform, int vao, int face_size, py::array_t<unsigned int> faces, GLuint fb) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, texture_id);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D_ARRAY, texture_id2);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture_bucket"), texture_bucket);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture_layer"), texture_layer);
        glUniform1i(glGetUniformLocation(shaderProgram, "tex1"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "tex2"), 1);

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

        int w;
        int h;
        int comp;
        stbi_set_flip_vertically_on_load(true);
        unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb);

        if(image == nullptr)
            throw(std::string("Failed to load texture"));


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
//
//    int loadTextureOptimized(std::string filename) {
//        int w;
//        int h;
//        int comp;
//        stbi_set_flip_vertically_on_load(true);
//        unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb);
//
//        if(image == nullptr)
//            throw(std::string("Failed to load texture"));
//
//    texData.push_back(image);
//    texHeights.push_back(h);
//    texWidths.push_back(w);
//    texChannels.push_back(comp);
//
//    // Texture index to be used by renderer
//    return texData.size() - 1;
//    }

  // Generates large and small array textures and returns handles to the user (cutoff based on user variable), as well as index - tex num/layer mapping
  py::list generateArrayTextures(std::vector<std::string> filenames, int texCutoff) {
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
        if(image == nullptr)
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
    printf("Texture 2 is w:%d by h:%d by depth:%d 3D array texture. ID %d\n", texLayerDims[2], texLayerDims[3], secondTexLayerNum, texId2);

    for (int i = 0; i < 2; i++) {
      GLuint currTexId = texId1;
      if (i == 1) currTexId = texId2;

      glBindTexture(GL_TEXTURE_2D_ARRAY, currTexId);

      int layerNum = firstTexLayerNum;
      if (i == 1) layerNum = secondTexLayerNum;

      int out_w = texLayerDims[2*i];
      int out_h = texLayerDims[2*i+1];

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
          printf("%d %d %d %d\n", j, orig_w, orig_h, n_channels);
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

   //TODO: Add the correct stuff to the pylist!
    py::list renderSetup(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor,
    py::array_t<float> mergedVertexData, py::array_t<int> index_ptr_offsets, py::array_t<int> index_counts,
    py::array_t<int> indices, py::array_t<float> mergedFragData, py::array_t<float> mergedDiffuseData,
    int tex_id_1, int tex_id_2, GLuint fb) {
        // First set up VAO and corresponding attributes
        GLuint VAO;
        glGenVertexArrays(1, &VAO);
        cglBindVertexArray(VAO);

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

        multidrawCount = index_ptr_offsets.size();
        int* indexOffsetPtr = (int*)index_ptr_offsets.request().ptr;


        for (int i = 0; i < multidrawCount; i++) {
          unsigned int offset = (unsigned int)indexOffsetPtr[i];
          this->multidrawStartIndices[i] = BUFFER_OFFSET((offset * sizeof(unsigned int)));
          printf("multidraw start idx %d\n", offset);
        }

        // Store for rendering
        //this->multidrawStartIndices = &this->startIndices[0];
        int * indices_count_ptr = (int*)index_counts.request().ptr;
        for (int i = 0; i < multidrawCount; i++) {
            this->multidrawCounts[i] = indices_count_ptr[i];
        }

        // Set up shaders
        float* fragData = (float*)mergedFragData.request().ptr;
        float* diffuseData = (float*)mergedDiffuseData.request().ptr;

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
        glBufferSubData(GL_UNIFORM_BUFFER, 0, texColorDataSize / 2, fragData);
        glBufferSubData(GL_UNIFORM_BUFFER, texColorDataSize / 2, texColorDataSize / 2, diffuseData);

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

        // Pre-render setup
        glBindVertexArray(VAO);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);

        py::list renderData;
        renderData.append(VAO);
        renderData.append(VBO);
        renderData.append(EBO);

        return renderData;
  }
//
  // Updates positions and rotations in vertex shader
  void updateDynamicData(int shaderProgram, py::array_t<float> pose_trans_array, py::array_t<float> pose_rot_array, py::array_t<float> V, py::array_t<float> P) {
    float* transPtr = (float*)pose_trans_array.request().ptr;
    float* rotPtr = (float*)pose_rot_array.request().ptr;

    glBindBuffer(GL_UNIFORM_BUFFER, uboTransformData);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, transformDataSize / 2, transPtr);
    glBufferSubData(GL_UNIFORM_BUFFER, transformDataSize / 2, transformDataSize / 2, rotPtr);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    float* Vptr = (float*)V.request().ptr;
    float* Pptr = (float*)P.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);

  }

  // Optimized rendering function that is called once per frame for all merged data
  void renderOptimized() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    //printf("multidraw count %d\n", this->multidrawCount);
    //printf("first count %d\n", *(int *)this->multidrawCounts);
    glMultiDrawElements(GL_TRIANGLES, this->multidrawCounts, GL_UNSIGNED_INT, this->multidrawStartIndices, this->multidrawCount);
    glDisable(GL_DEPTH_TEST);
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
    pymodule.def("clean_meshrenderer", &GLFWRendererContext::clean_meshrenderer, "clean meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer", &GLFWRendererContext::setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");
    pymodule.def("setup_framebuffer_meshrenderer_ms", &GLFWRendererContext::setup_framebuffer_meshrenderer_ms, "setup framebuffer in meshrenderer with MSAA");
    pymodule.def("blit_buffer", &GLFWRendererContext::blit_buffer, "blit buffer");

    pymodule.def("compile_shader_meshrenderer", &GLFWRendererContext::compile_shader_meshrenderer, "compile vertex and fragment shader");
    pymodule.def("load_object_meshrenderer", &GLFWRendererContext::load_object_meshrenderer, "load object into VAO and VBO");
    pymodule.def("loadTexture", &GLFWRendererContext::loadTexture, "load texture function");

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


    //renderer optimization
//    pymodule.def("loadTextureOptimized", &GLFWRendererContext::loadTextureOptimized, "optimized load texture function");
    pymodule.def("generateArrayTextures", &GLFWRendererContext::generateArrayTextures, "generate array texture function");
    pymodule.def("draw_elements_instance_optimized", &GLFWRendererContext::draw_elements_instance_optimized, "draw elements in instance.render and instancegroup.render");
    pymodule.def("renderSetup", &GLFWRendererContext::renderSetup, "loads all merged graphics data");
    pymodule.def("updateDynamicData", &GLFWRendererContext::updateDynamicData, "updates dynamic data such as object transforms");
    pymodule.def("renderOptimized", &GLFWRendererContext::renderOptimized, "renders merged data in an optimized way");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}