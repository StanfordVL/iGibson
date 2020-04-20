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
#endif

//#include  <glad/gl.h>


#include <GL/glcorearb.h>
#include <GL/glext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>

#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_gl_interop.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_NUM_RESOURCES 10

namespace py = pybind11;

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


class MeshRendererContext{
public:
    MeshRendererContext(int w, int h, int d):m_windowHeight(h),m_windowWidth(w),m_renderDevice(d) {};

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;


    EGLInternalData2* m_data = NULL;

#ifdef USE_CUDA
    cudaGraphicsResource* cuda_res[MAX_NUM_RESOURCES];
#endif

    int init() {


    m_data = new EGLInternalData2();

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
        EGL_OPENGL_ES2_BIT,
        EGL_NONE};

    EGLint egl_pbuffer_attribs[] = {
        EGL_WIDTH, m_windowWidth, EGL_HEIGHT, m_windowHeight,
        EGL_NONE,
    };

    		m_data->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
         EGLint major;
		EGLint minor;
		EGLBoolean initialized = eglInitialize(m_data->egl_display, &major, &minor);

        	const char *eglVendor = eglQueryString(m_data->egl_display, EGL_VENDOR);
			printf(eglVendor);

			const char *eglVersion = eglQueryString(m_data->egl_display, EGL_VERSION);
			printf(eglVersion);

			eglBindAPI(EGL_OPENGL_ES_API);
			EGLBoolean success = eglChooseConfig(m_data->egl_display, egl_config_attribs, &m_data->egl_config, 1,
			&m_data->num_configs);
			printf("egl success %d\n", success);

			m_data->egl_surface = eglCreatePbufferSurface(m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);

            EGLint contextAttributes[] = {
                EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL_NONE
            };

		m_data->egl_context = eglCreateContext(m_data->egl_display, m_data->egl_config, NULL, contextAttributes);
		success = eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface, m_data->egl_context);

        const GLubyte* ven = glGetString(GL_VENDOR);
	    printf("GL_VENDOR=%s\n", ven);

	    const GLubyte* ren = glGetString(GL_RENDERER);
	    printf("GL_RENDERER=%s\n", ren);
	    const GLubyte* ver = glGetString(GL_VERSION);
	    printf("GL_VERSION=%s\n", ver);
	    const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
	    printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);

    return 0;
    };

    void release(){
        eglTerminate(m_data->egl_display);
        delete m_data;
#ifdef USE_CUDA        
        for (int i = 0; i < MAX_NUM_RESOURCES; i++)
          {
            if (cuda_res[i])
            {
              cudaError_t err = cudaGraphicsUnregisterResource(cuda_res[i]);
              if( err != cudaSuccess )
              {
                std::cout << "cudaGraphicsUnregisterResource failed: " << err << std::endl;
              }
            }
          }
#endif
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
           std::cout << "cudaGraphicsGLRegisterImage failed: " << err << std::endl;
         }
       }

       err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsMapResources failed: " << err << std::endl;
       }

       cudaArray* array;
       err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
       }

       // copy data
       err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(char), array, 0, 0, width*4*sizeof(char), height, cudaMemcpyDeviceToDevice);
       if( err != cudaSuccess )
       {
         std::cout << "cudaMemcpy2DFromArray failed: " << err << std::endl;
       }

       err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsUnmapResources failed: " << err << std::endl;
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
           std::cout << "cudaGraphicsGLRegisterImage failed: " << err << std::endl;
         }
       }

       err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsMapResources failed: " << err << std::endl;
       }

       cudaArray* array;
       err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
       }

       // copy data
       err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(float), array, 0, 0, width*4*sizeof(float), height, cudaMemcpyDeviceToDevice);
       if( err != cudaSuccess )
       {
         std::cout << "cudaMemcpy2DFromArray failed: " << err << std::endl;
       }

       err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
       if( err != cudaSuccess )
       {
         std::cout << "cudaGraphicsUnmapResources failed: " << err << std::endl;
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

    void glad_init() {
        //if (!gladLoadGL(eglGetProcAddress)) {
        //    fprintf(stderr, "failed to load GL with glad.\n");
        //    exit(EXIT_FAILURE);
        //}
    }

    std::string getstring_meshrenderer() {
        return reinterpret_cast<char const *>(glGetString(GL_VERSION));
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
            fprintf(stderr, "unknown buffer mode.\n");
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

    void draw_elements_instance(bool flag, int texture_id, int texUnitUniform, int vao, int face_size, py::array_t<unsigned int> faces, GLuint fb) {
        glActiveTexture(GL_TEXTURE0);
        if (flag) glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(texUnitUniform, 0);
        glBindVertexArray(vao);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        //unsigned int *ptr = (unsigned int *) faces.request().ptr;
        //glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, ptr);

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
    //    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    //    w, h = img.size
    //
    //    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    //    img_data = np.frombuffer(img.tobytes(), np.uint8)
    //    #print(img_data.shape)
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

};


PYBIND11_MODULE(MeshRendererContext, m) {
        py::class_<MeshRendererContext> pymodule = py::class_<MeshRendererContext>(m, "MeshRendererContext");
        
        pymodule.def(py::init<int, int, int>());
        pymodule.def("init", &MeshRendererContext::init);
        pymodule.def("release", &MeshRendererContext::release);

#ifdef USE_CUDA
        pymodule.def("map_tensor", &MeshRendererContext::map_tensor);
        pymodule.def("map_tensor_float", &MeshRendererContext::map_tensor_float);
#endif
        // class MeshRenderer
        pymodule.def("render_meshrenderer_pre", &MeshRendererContext::render_meshrenderer_pre, "pre-executed functions in MeshRenderer.render");
        pymodule.def("render_meshrenderer_post", &MeshRendererContext::render_meshrenderer_post, "post-executed functions in MeshRenderer.render");
        pymodule.def("getstring_meshrenderer", &MeshRendererContext::getstring_meshrenderer, "return GL version string");
        pymodule.def("readbuffer_meshrenderer", &MeshRendererContext::readbuffer_meshrenderer, "read pixel buffer");
        pymodule.def("glad_init", &MeshRendererContext::glad_init, "init glad");
        pymodule.def("clean_meshrenderer", &MeshRendererContext::clean_meshrenderer, "clean meshrenderer");
        pymodule.def("setup_framebuffer_meshrenderer", &MeshRendererContext::setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");

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
        

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
