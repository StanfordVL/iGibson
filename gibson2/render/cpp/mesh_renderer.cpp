#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include <fstream>

#ifdef USE_GLAD

#include  <glad/egl.h>

#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include  <glad/gl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "mesh_renderer.h"
#define MAX_ARRAY_SIZE 1024
#define BUFFER_OFFSET(offset) (static_cast<char*>(0) + (offset))

namespace py = pybind11;

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

class Image {
public:
    static std::shared_ptr<Image> fromFile(const std::string &filename, int channels) {
        std::printf("Loading image: %s\n", filename.c_str());
        stbi_set_flip_vertically_on_load(false);
        std::shared_ptr<Image> image{new Image};

        if (stbi_is_hdr(filename.c_str())) {
            float *pixels = stbi_loadf(filename.c_str(), &image->m_width, &image->m_height, &image->m_channels,
                                       channels);
            if (pixels) {
                image->m_pixels.reset(reinterpret_cast<unsigned char *>(pixels));
                image->m_hdr = true;
            }
        } else {
            unsigned char *pixels = stbi_load(filename.c_str(), &image->m_width, &image->m_height, &image->m_channels,
                                              channels);
            if (pixels) {
                image->m_pixels.reset(pixels);
                image->m_hdr = false;
            }
        }
        if (channels > 0) {
            image->m_channels = channels;
        }

        if (!image->m_pixels) {
            throw std::runtime_error("Failed to load image file: " + filename);
        }
        return image;
    }


    int width() const { return m_width; }

    int height() const { return m_height; }

    int channels() const { return m_channels; }

    int bytesPerPixel() const { return m_channels * (m_hdr ? sizeof(float) : sizeof(unsigned char)); }

    int pitch() const { return m_width * bytesPerPixel(); }

    bool isHDR() const { return m_hdr; }

    template<typename T>
    const T *pixels() const {
        assert(m_channels * sizeof(T) == bytesPerPixel());
        return reinterpret_cast<const T *>(m_pixels.get());
    }

private:
    Image()
            : m_width(0), m_height(0), m_channels(0), m_hdr(false) {}

    int m_width;
    int m_height;
    int m_channels;
    bool m_hdr;
    std::unique_ptr<unsigned char> m_pixels;
};


#ifdef USE_CUDA
void MeshRendererContext::map_tensor(GLuint tid, int width, int height, std::size_t data)
{
   cudaError_t err;
   if (cuda_res[tid] == NULL)
   {
     err = cudaGraphicsGLRegisterImage(&(cuda_res[tid]), tid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
     if( err != cudaSuccess)
     {
       std::cout << "WARN: cudaGraphicsGLRegisterImage failed: " << err << std::endl;
     }
   }

   err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsMapResources failed: " << err << std::endl;
   }

   cudaArray* array;
   err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
   }

   // copy data
   err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(char), array, 0, 0, width*4*sizeof(char), height, cudaMemcpyDeviceToDevice);
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaMemcpy2DFromArray failed: " << err << std::endl;
   }

   err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsUnmapResources failed: " << err << std::endl;
   }
}

void MeshRendererContext::map_tensor_float(GLuint tid, int width, int height, std::size_t data)
{
   cudaError_t err;
   if (cuda_res[tid] == NULL)
   {
     err = cudaGraphicsGLRegisterImage(&(cuda_res[tid]), tid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
     if( err != cudaSuccess)
     {
       std::cout << "WARN: cudaGraphicsGLRegisterImage failed: " << err << std::endl;
     }
   }

   err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsMapResources failed: " << err << std::endl;
   }

   cudaArray* array;
   err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
   }

   // copy data
   err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(float), array, 0, 0, width*4*sizeof(float), height, cudaMemcpyDeviceToDevice);
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaMemcpy2DFromArray failed: " << err << std::endl;
   }

   err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
   if( err != cudaSuccess)
   {
     std::cout << "WARN: cudaGraphicsUnmapResources failed: " << err << std::endl;
   }
}
#endif

void MeshRendererContext::render_meshrenderer_pre(bool msaa, GLuint fb1, GLuint fb2) {

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

void MeshRendererContext::render_meshrenderer_post() {
    glDisable(GL_DEPTH_TEST);
}

std::string MeshRendererContext::getstring_meshrenderer() {
    return reinterpret_cast<char const *>(glGetString(GL_VERSION));
}

void MeshRendererContext::blit_buffer(int width, int height, GLuint fb1, GLuint fb2) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb1);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb2);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    for (int i = 0; i < 6; i++) {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
        glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
}

py::array_t<float> MeshRendererContext::readbuffer_meshrenderer(char *mode, int width, int height, GLuint fb2) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    if (!strcmp(mode, "rgb")) {
        glReadBuffer(GL_COLOR_ATTACHMENT0);
    } else if (!strcmp(mode, "normal")) {
        glReadBuffer(GL_COLOR_ATTACHMENT1);
    } else if (!strcmp(mode, "seg")) {
        glReadBuffer(GL_COLOR_ATTACHMENT2);
    } else if (!strcmp(mode, "3d")) {
        glReadBuffer(GL_COLOR_ATTACHMENT3);
    } else if (!strcmp(mode, "scene_flow")) {
        glReadBuffer(GL_COLOR_ATTACHMENT4);
    }
    else if (!strcmp(mode, "optical_flow")) {
        glReadBuffer(GL_COLOR_ATTACHMENT5);
    }
    else {
        fprintf(stderr, "ERROR: Unknown buffer mode.\n");
        exit(EXIT_FAILURE);
    }
    py::array_t<float> data = py::array_t<float>(4 * width * height);
    py::buffer_info buf = data.request();
    float *ptr = (float *) buf.ptr;
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, ptr);
    return data;
}


void MeshRendererContext::clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2,
                                             std::vector<GLuint> fbo, std::vector<GLuint> vaos,
                                             std::vector<GLuint> vbos) {
    glDeleteTextures(texture1.size(), texture1.data());
    glDeleteTextures(texture2.size(), texture2.data());
    glDeleteFramebuffers(fbo.size(), fbo.data());
    glDeleteBuffers(vaos.size(), vaos.data());
    glDeleteBuffers(vbos.size(), vbos.data());

    if (m_envTexture.id != 0) glDeleteTextures(1, &m_envTexture.id);
    if (m_irmapTexture.id != 0) glDeleteTextures(1, &m_irmapTexture.id);
    if (m_spBRDF_LUT.id != 0) glDeleteTextures(1, &m_spBRDF_LUT.id);
    if (m_envTexture2.id != 0) glDeleteTextures(1, &m_envTexture2.id);
    if (m_irmapTexture2.id != 0) glDeleteTextures(1, &m_irmapTexture2.id);
    if (m_spBRDF_LUT2.id != 0) glDeleteTextures(1, &m_spBRDF_LUT2.id);
    if (m_envTexture3.id != 0) glDeleteTextures(1, &m_envTexture3.id);
    if (m_light_modulation_map.id != 0) glDeleteTextures(1, &m_light_modulation_map.id);
    if (m_default_metallic_texture.id != 0) glDeleteTextures(1, &m_default_metallic_texture.id);
    if (m_default_roughness_texture.id != 0) glDeleteTextures(1, &m_default_roughness_texture.id);
    if (m_default_normal_texture.id != 0) glDeleteTextures(1, &m_default_normal_texture.id);

}

py::list MeshRendererContext::setup_framebuffer_meshrenderer(int width, int height) {
    GLuint *fbo_ptr = (GLuint *) malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint *) malloc(7 * sizeof(GLuint));
    glGenFramebuffers(1, fbo_ptr);
    glGenTextures(7, texture_ptr);
    int fbo = fbo_ptr[0];
    int color_tex_rgb = texture_ptr[0];
    int color_tex_normal = texture_ptr[1];
    int color_tex_semantics = texture_ptr[2];
    int color_tex_3d = texture_ptr[3];
    int color_tex_scene_flow = texture_ptr[4];
    int color_tex_optical_flow = texture_ptr[5];
    int depth_tex = texture_ptr[6];

    glBindTexture(GL_TEXTURE_2D, color_tex_rgb);
    // Note: VR textures need these settings, otherwise they won't display on the HMD
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_normal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_semantics);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_3d);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_scene_flow);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_optical_flow);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_rgb, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, color_tex_normal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, color_tex_semantics, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, color_tex_3d, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, color_tex_scene_flow, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, color_tex_optical_flow, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
    glViewport(0, 0, width, height);
    GLenum *bufs = (GLenum *) malloc(6 * sizeof(GLenum));
    bufs[0] = GL_COLOR_ATTACHMENT0;
    bufs[1] = GL_COLOR_ATTACHMENT1;
    bufs[2] = GL_COLOR_ATTACHMENT2;
    bufs[3] = GL_COLOR_ATTACHMENT3;
    bufs[4] = GL_COLOR_ATTACHMENT4;
    bufs[5] = GL_COLOR_ATTACHMENT5;
    glDrawBuffers(6, bufs);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    py::list result;
    result.append(fbo);
    result.append(color_tex_rgb);
    result.append(color_tex_normal);
    result.append(color_tex_semantics);
    result.append(color_tex_3d);
    result.append(color_tex_scene_flow);
    result.append(color_tex_optical_flow);
    result.append(depth_tex);
    return result;
}

py::list MeshRendererContext::setup_framebuffer_meshrenderer_ms(int width, int height) {
    GLuint *fbo_ptr = (GLuint *) malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint *) malloc(7 * sizeof(GLuint));
    glGenFramebuffers(1, fbo_ptr);
    glGenTextures(7, texture_ptr);
    int fbo = fbo_ptr[0];
    int color_tex_rgb = texture_ptr[0];
    int color_tex_normal = texture_ptr[1];
    int color_tex_semantics = texture_ptr[2];
    int color_tex_3d = texture_ptr[3];
    int color_tex_scene_flow = texture_ptr[4];
    int color_tex_optical_flow = texture_ptr[5];
    int depth_tex = texture_ptr[6];
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_scene_flow);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_optical_flow);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depth_tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_DEPTH24_STENCIL8, width, height, GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D_MULTISAMPLE, color_tex_scene_flow, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D_MULTISAMPLE, color_tex_optical_flow, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depth_tex, 0);
    glViewport(0, 0, width, height);
    GLenum *bufs = (GLenum *) malloc(6 * sizeof(GLenum));
    bufs[0] = GL_COLOR_ATTACHMENT0;
    bufs[1] = GL_COLOR_ATTACHMENT1;
    bufs[2] = GL_COLOR_ATTACHMENT2;
    bufs[3] = GL_COLOR_ATTACHMENT3;
    bufs[4] = GL_COLOR_ATTACHMENT4;
    bufs[5] = GL_COLOR_ATTACHMENT5;
    glDrawBuffers(6, bufs);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    py::list result;
    result.append(fbo);
    result.append(color_tex_rgb);
    result.append(color_tex_normal);
    result.append(color_tex_semantics);
    result.append(color_tex_3d);
    result.append(color_tex_scene_flow);
    result.append(color_tex_optical_flow);
    result.append(depth_tex);
    return result;
}

int MeshRendererContext::compile_shader_meshrenderer(char *vertexShaderSource, char *fragmentShaderSource) {
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
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
    return shaderProgram;
}

py::list MeshRendererContext::load_object_meshrenderer(int shaderProgram, py::array_t<float> vertexData) {
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    py::buffer_info buf = vertexData.request();
    float *ptr = (float *) buf.ptr;
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), ptr, GL_STATIC_DRAW);
    GLuint positionAttrib = glGetAttribLocation(shaderProgram, "position");
    GLuint normalAttrib = glGetAttribLocation(shaderProgram, "normal");
    GLuint coordsAttrib = glGetAttribLocation(shaderProgram, "texCoords");
    GLuint tangentlAttrib = glGetAttribLocation(shaderProgram, "tangent");
    GLuint bitangentAttrib = glGetAttribLocation(shaderProgram, "bitangent");

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);

    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void *) 0);
    glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void *) 12);
    glVertexAttribPointer(coordsAttrib, 2, GL_FLOAT, GL_TRUE, 56, (void *) 24);
    glVertexAttribPointer(tangentlAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void *) 32);
    glVertexAttribPointer(bitangentAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void *) 44);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    py::list result;
    result.append(VAO);
    result.append(VBO);
    return result;
}

void MeshRendererContext::render_softbody_instance(int vao, int vbo, py::array_t<float> vertexData) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    py::buffer_info buf = vertexData.request();
    float *ptr = (float *) buf.ptr;
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), ptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void
MeshRendererContext::init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color,
                                            float use_texture, float use_pbr, float use_pbr_mapping, float metallic,
                                            float roughness, py::array_t<float> transform_param) {
    float *diffuse_ptr = (float *) diffuse_color.request().ptr;
    float *transform_param_ptr = (float *) transform_param.request().ptr;

    glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "uv_transform_param"), transform_param_ptr[0],
                                                                        transform_param_ptr[1],
                                                                        transform_param_ptr[2]);

    glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr"), use_pbr);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr_mapping"), use_pbr_mapping);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_two_light_probe"), (float)m_use_two_light_probe);

    glUniform1f(glGetUniformLocation(shaderProgram, "metallic"), metallic);
    glUniform1f(glGetUniformLocation(shaderProgram, "roughness"), roughness);
    glUniform1i(glGetUniformLocation(shaderProgram, "texUnit"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture"), 2);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT"), 3);

    glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture2"), 4);
    glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture2"), 5);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT2"), 6);

    glUniform1i(glGetUniformLocation(shaderProgram, "metallicTexture"), 7);
    glUniform1i(glGetUniformLocation(shaderProgram, "roughnessTexture"), 8);
    glUniform1i(glGetUniformLocation(shaderProgram, "normalTexture"), 9);
    glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 10);

    glUniform1i(glGetUniformLocation(shaderProgram, "lightModulationMap"), 11);


}

void MeshRendererContext::draw_elements_instance(bool flag, int texture_id, int metallic_texture_id,
                                                 int roughness_texture_id,
                                                 int normal_texture_id, int depth_texture_id, int vao, int face_size,
                                                 py::array_t<unsigned int> faces, GLuint fb) {
    glActiveTexture(GL_TEXTURE0);
    if (flag) glBindTexture(GL_TEXTURE_2D, texture_id);

    glActiveTexture(GL_TEXTURE1);
    if (flag) glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTexture.id);

    glActiveTexture(GL_TEXTURE2);
    if (flag) glBindTexture(GL_TEXTURE_CUBE_MAP, m_irmapTexture.id);

    glActiveTexture(GL_TEXTURE3);
    if (flag) glBindTexture(GL_TEXTURE_2D, m_spBRDF_LUT.id);

    glActiveTexture(GL_TEXTURE4);
    if (flag) glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTexture2.id);

    glActiveTexture(GL_TEXTURE5);
    if (flag) glBindTexture(GL_TEXTURE_CUBE_MAP, m_irmapTexture2.id);

    glActiveTexture(GL_TEXTURE6);
    if (flag) glBindTexture(GL_TEXTURE_2D, m_spBRDF_LUT2.id);


    if (metallic_texture_id != -1) {
        glActiveTexture(GL_TEXTURE7);
        if (flag) glBindTexture(GL_TEXTURE_2D, metallic_texture_id);
    } else {
        glActiveTexture(GL_TEXTURE7);
        if (flag) glBindTexture(GL_TEXTURE_2D, m_default_metallic_texture.id);
    }

    if (roughness_texture_id != -1) {
        glActiveTexture(GL_TEXTURE8);
        if (flag) glBindTexture(GL_TEXTURE_2D, roughness_texture_id);
    } else {
        glActiveTexture(GL_TEXTURE8);
        if (flag) glBindTexture(GL_TEXTURE_2D, m_default_roughness_texture.id);
    }

    if (normal_texture_id != -1) {
        glActiveTexture(GL_TEXTURE9);
        if (flag) glBindTexture(GL_TEXTURE_2D, normal_texture_id);
    } else {
        glActiveTexture(GL_TEXTURE9);
        if (flag) glBindTexture(GL_TEXTURE_2D, m_default_normal_texture.id);
    }

    glActiveTexture(GL_TEXTURE10);
    glBindTexture(GL_TEXTURE_2D, depth_texture_id);

    if (m_use_two_light_probe) {
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D, m_light_modulation_map.id);
    }
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    unsigned int *ptr = (unsigned int *) faces.request().ptr;

    GLuint elementBuffer;
    glGenBuffers(1, &elementBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_size * sizeof(unsigned int), &ptr[0], GL_STATIC_DRAW);
    glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, (void *) 0);
    glDeleteBuffers(1, &elementBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void MeshRendererContext::initvar(int shaderProgram, py::array_t<float> V, py::array_t<float> last_V, py::array_t<float>
                                                 lightV, int shadow_pass, py::array_t<float> P, py::array_t<float> lightP,
                                                 py::array_t<float> eye_pos, py::array_t<float> lightpos,
                                                 py::array_t<float> lightcolor) {
    glUseProgram(shaderProgram);
    float *Vptr = (float *) V.request().ptr;
    float *last_Vptr = (float *) last_V.request().ptr;
    float *lightVptr = (float *) lightV.request().ptr;
    float *Pptr = (float *) P.request().ptr;
    float *lightPptr = (float *) lightP.request().ptr;
    float *lightposptr = (float *) lightpos.request().ptr;
    float *lightcolorptr = (float *) lightcolor.request().ptr;
    float *eye_pos_ptr = (float *) eye_pos.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "last_V"), 1, GL_TRUE, last_Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightP"), 1, GL_FALSE, lightPptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "eyePosition"), eye_pos_ptr[0], eye_pos_ptr[1], eye_pos_ptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1],
                lightcolorptr[2]);
    glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), shadow_pass);
}

void MeshRendererContext::init_pos_instance(int shaderProgram, py::array_t<float> pose_trans,
                                                     py::array_t<float> pose_rot, py::array_t<float> last_trans,
                                                     py::array_t<float> last_rot) {
    float *transptr = (float *) pose_trans.request().ptr;
    float *rotptr = (float *) pose_rot.request().ptr;
    float *lasttransptr = (float *) last_trans.request().ptr;
    float *lastrotptr = (float *) last_rot.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "last_trans"), 1, GL_FALSE, lasttransptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "last_rot"), 1, GL_TRUE, lastrotptr);
}


void MeshRendererContext::cglBindVertexArray(int vao) {
    glBindVertexArray(vao);
}

void MeshRendererContext::cglUseProgram(int shaderProgram) {
    glUseProgram(shaderProgram);
}

int MeshRendererContext::loadTexture(std::string filename, float texture_scale) {
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
    unsigned char *image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb);
    if (image == nullptr)
        throw (std::string("ERROR: Failed to load texture"));

    int new_w = (int)(w * texture_scale);
    int new_h = (int)(h * texture_scale);

    unsigned char *resized_image = (unsigned char *)malloc(w*h*comp);
    stbir_resize_uint8(image, w, h, 0, resized_image, new_w, new_h, 0, comp);
//    STBIRDEF int stbir_resize_uint8(     const unsigned char *input_pixels , int input_w , int input_h , int input_stride_in_bytes,
//                                           unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
//                                     int num_channels);

    GLuint texture;
    glGenTextures(1, &texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, new_w, new_h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, resized_image);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(image);
    stbi_image_free(resized_image);
    return texture;
}

void MeshRendererContext::generate_light_maps(
    GLuint equirectToCubeProgram,
    GLuint spmapProgram,
    GLuint irmapProgram,
    GLuint spBRDFProgram,
    std::string env_texture_filename,
    Texture& envTexture,
    Texture& irmapTexture,
    Texture& spBRDF_LUT,
    float light_dimming_factor
    ){

    envTextureUnfiltered = createTexture(GL_TEXTURE_CUBE_MAP, kEnvMapSize, kEnvMapSize, GL_RGBA16F, 0);

    // Load & convert equirectangular environment map to a cubemap texture.
    {
        envTextureEquirect = createTexture(Image::fromFile(env_texture_filename, 3), GL_RGB, GL_RGB16F, 1);
        glUseProgram(equirectToCubeProgram);
        glBindTextureUnit(0, envTextureEquirect.id);
        glUniform1f(glGetUniformLocation(equirectToCubeProgram, "light_dimming_factor"), (float)light_dimming_factor);
        glBindImageTexture(0, envTextureUnfiltered.id, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glDispatchCompute(envTextureUnfiltered.width / 32, envTextureUnfiltered.height / 32, 6);
    }
    glDeleteTextures(1, &envTextureEquirect.id);
    glGenerateTextureMipmap(envTextureUnfiltered.id);
    {
        envTexture = createTexture(GL_TEXTURE_CUBE_MAP, kEnvMapSize, kEnvMapSize, GL_RGBA16F, 0);

        // Copy 0th mipmap level into destination environment map.
        glCopyImageSubData(envTextureUnfiltered.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           envTexture.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           envTexture.width, envTexture.height, 6);

        glUseProgram(spmapProgram);
        glBindTextureUnit(0, envTextureUnfiltered.id);

        // Pre-filter rest of the mip chain.
        const float deltaRoughness = 1.0f / glm::max(float(envTexture.levels - 1), 1.0f);
        for (int level = 1, size = kEnvMapSize / 2; level <= envTexture.levels; ++level, size /= 2) {
            const GLuint numGroups = glm::max(1, size / 32);
            glBindImageTexture(0, envTexture.id, level, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            glProgramUniform1f(spmapProgram, 0, level * deltaRoughness);
            glDispatchCompute(numGroups, numGroups, 6);
        }
    }

    glDeleteTextures(1, &envTextureUnfiltered.id);
    // Compute diffuse irradiance cubemap.
    {
        irmapTexture = createTexture(GL_TEXTURE_CUBE_MAP, kIrradianceMapSize, kIrradianceMapSize, GL_RGBA16F, 1);
        glUseProgram(irmapProgram);
        glBindTextureUnit(0, envTexture.id);
        glBindImageTexture(0, irmapTexture.id, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glDispatchCompute(irmapTexture.width / 32, irmapTexture.height / 32, 6);

    }

    // Compute Cook-Torrance BRDF 2D LUT for split-sum approximation.
    {
        spBRDF_LUT = createTexture(GL_TEXTURE_2D, kBRDF_LUT_Size, kBRDF_LUT_Size, GL_RG16F, 1);
        glTextureParameteri(m_spBRDF_LUT.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(m_spBRDF_LUT.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glUseProgram(spBRDFProgram);
        glBindImageTexture(0, spBRDF_LUT.id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
        glDispatchCompute(spBRDF_LUT.width / 32, spBRDF_LUT.height / 32, 1);
    }

}


void MeshRendererContext::generate_env_map(
    GLuint equirectToCubeProgram,
    std::string env_texture_filename,
    Texture& envTexture
    ){

    envTextureUnfiltered = createTexture(GL_TEXTURE_CUBE_MAP, kSkyBoxMapSize, kSkyBoxMapSize, GL_RGBA16F, 0);

    // Load & convert equirectangular environment map to a cubemap texture.
    {
        envTextureEquirect = createTexture(Image::fromFile(env_texture_filename, 3), GL_RGB, GL_RGB16F, 1);
        glUseProgram(equirectToCubeProgram);
        glBindTextureUnit(0, envTextureEquirect.id);
        glBindImageTexture(0, envTextureUnfiltered.id, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glDispatchCompute(envTextureUnfiltered.width / 32, envTextureUnfiltered.height / 32, 6);
    }
    glDeleteTextures(1, &envTextureEquirect.id);
    glGenerateTextureMipmap(envTextureUnfiltered.id);
    {
        envTexture = createTexture(GL_TEXTURE_CUBE_MAP, kSkyBoxMapSize, kSkyBoxMapSize, GL_RGBA16F, 0);
        // Copy 0th mipmap level into destination environment map.
        glCopyImageSubData(envTextureUnfiltered.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           envTexture.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           envTexture.width, envTexture.height, 6);

    }
    glGenerateTextureMipmap(envTexture.id);
    glDeleteTextures(1, &envTextureUnfiltered.id);
}

void MeshRendererContext::setup_pbr(std::string shader_path,
                                    std::string env_texture_filename,
                                    std::string env_texture_filename2,
                                    std::string env_texture_filename3,
                                    std::string light_modulation_map_filename,
                                    float light_dimming_factor
                                    )
                                    {

    //glEnable(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    if (light_modulation_map_filename.length() > 0) {
        m_light_modulation_map = createTexture(Image::fromFile(light_modulation_map_filename, 3), GL_RGB, GL_RGB16F, 1);
        m_use_two_light_probe = true;
    }
    //glFrontFace(GL_CCW);
    // create all the programs
    GLuint equirectToCubeProgram = linkProgram({compileShader(shader_path + OS_SEP + "equirect2cube_cs.glsl",
                                                                         GL_COMPUTE_SHADER)});
    GLuint spmapProgram = linkProgram({compileShader(shader_path + OS_SEP + "spmap_cs.glsl", GL_COMPUTE_SHADER)});
    GLuint irmapProgram = linkProgram({compileShader(shader_path + OS_SEP + "irmap_cs.glsl", GL_COMPUTE_SHADER)});
    GLuint spBRDFProgram = linkProgram({compileShader(shader_path + OS_SEP + "spbrdf_cs.glsl", GL_COMPUTE_SHADER)});

    // run subroutine to generate light map
    generate_light_maps(equirectToCubeProgram,
                        spmapProgram,
                        irmapProgram,
                        spBRDFProgram,
                        env_texture_filename,
                        m_envTexture,
                        m_irmapTexture,
                        m_spBRDF_LUT,
                        light_dimming_factor
                        );

    if (env_texture_filename2.length() > 0) {
        if (env_texture_filename2 == env_texture_filename) {
            m_envTexture2 = m_envTexture;
            m_irmapTexture2 = m_irmapTexture;
            m_spBRDF_LUT2 = m_spBRDF_LUT;
        } else {
            generate_light_maps(equirectToCubeProgram,
                        spmapProgram,
                        irmapProgram,
                        spBRDFProgram,
                        env_texture_filename2,
                        m_envTexture2,
                        m_irmapTexture2,
                        m_spBRDF_LUT2,
                        light_dimming_factor
                        );
        }
    }

    if (env_texture_filename3.length() > 0) {
        if (env_texture_filename3 == env_texture_filename2) {
            m_envTexture3 = m_envTexture2;
        } else if (env_texture_filename3 == env_texture_filename) {
            m_envTexture3 = m_envTexture;
        } else {
            generate_env_map(equirectToCubeProgram,
                         env_texture_filename3,
                         m_envTexture3);
        }
    }

    // delete all the programs
    glDeleteProgram(equirectToCubeProgram);
    glDeleteProgram(spmapProgram);
    glDeleteProgram(irmapProgram);
    glDeleteProgram(spBRDFProgram);



    m_default_metallic_texture = createTexture(GL_TEXTURE_2D, 1, 1, GL_R32F, 1);
    std::vector<GLfloat> zeros(1 * 1 * 1, 0.0);
    glBindTexture(GL_TEXTURE_2D, m_default_metallic_texture.id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &zeros[0]);
    glGenerateTextureMipmap(m_default_metallic_texture.id);

    m_default_roughness_texture = createTexture(GL_TEXTURE_2D, 1, 1, GL_R32F, 1);
    std::vector<GLfloat> ones(1 * 1 * 1, 1.0);
    glBindTexture(GL_TEXTURE_2D, m_default_roughness_texture.id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &ones[0]);
    glGenerateTextureMipmap(m_default_roughness_texture.id);

    m_default_normal_texture = createTexture(GL_TEXTURE_2D, 1, 1, GL_RGBA32F, 1);
    std::vector<GLfloat> default_normal(1 * 1 * 4, 1.0);
    default_normal[0] = 0.5;
    default_normal[1] = 0.5;
    glBindTexture(GL_TEXTURE_2D, m_default_normal_texture.id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, &default_normal[0]);
    glGenerateTextureMipmap(m_default_normal_texture.id);

    glFinish();

    std::cout << "INFO: compiled pbr shaders" << std::endl;
}

GLuint MeshRendererContext::linkProgram(std::initializer_list<GLuint> shaders) {
    GLuint program = glCreateProgram();

    for (GLuint shader : shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);
    for (GLuint shader : shaders) {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_TRUE) {
        glValidateProgram(program);
        glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
    }
    if (status != GL_TRUE) {
        GLsizei infoLogSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogSize);
        std::unique_ptr<GLchar[]> infoLog(new GLchar[infoLogSize]);
        glGetProgramInfoLog(program, infoLogSize, nullptr, infoLog.get());
        throw std::runtime_error(std::string("Program link failed\n") + infoLog.get());
    }
    return program;
}

std::string MeshRendererContext::readText(const std::string &filename) {
    std::ifstream file{filename};
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint MeshRendererContext::compileShader(const std::string &filename, GLenum type) {
    const std::string src = readText(filename);
    if (src.empty()) {
        throw std::runtime_error("Cannot read shader source file: " + filename);
    }
    const GLchar *srcBufferPtr = src.c_str();

    std::printf("Compiling GLSL shader: %s\n", filename.c_str());

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &srcBufferPtr, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLsizei infoLogSize;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogSize);
        std::unique_ptr<GLchar[]> infoLog(new GLchar[infoLogSize]);
        glGetShaderInfoLog(shader, infoLogSize, nullptr, infoLog.get());
        throw std::runtime_error(std::string("Shader compilation failed: ") + filename + "\n" + infoLog.get());
    }
    return shader;
}

int MeshRendererContext::numMipmapLevels(int width, int height)
	{
		int levels = 1;
		while((width|height) >> levels) {
			++levels;
		}
		return levels;
	}

Texture
MeshRendererContext::createTexture(GLenum target, int width, int height, GLenum internalformat, int levels) const {
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.levels = (levels > 0) ? levels : numMipmapLevels(width, height);

    glCreateTextures(target, 1, &texture.id);
    glTextureStorage2D(texture.id, texture.levels, internalformat, width, height);
    glTextureParameteri(texture.id, GL_TEXTURE_MIN_FILTER, texture.levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTextureParameteri(texture.id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTextureParameterf(texture.id, GL_TEXTURE_MAX_ANISOTROPY_EXT, m_capabilities.maxAnisotropy);
    return texture;
}

Texture
MeshRendererContext::createTexture(const std::shared_ptr<class Image> &image, GLenum format, GLenum internalformat,
                                   int levels) const {
    Texture texture = createTexture(GL_TEXTURE_2D, image->width(), image->height(), internalformat, levels);
    if (image->isHDR()) {
        glTextureSubImage2D(texture.id, 0, 0, 0, texture.width, texture.height, format, GL_FLOAT,
                            image->pixels<float>());
    } else {
        glTextureSubImage2D(texture.id, 0, 0, 0, texture.width, texture.height, format, GL_UNSIGNED_BYTE,
                            image->pixels<unsigned char>());
    }


    //std::vector<unsigned char> emptyData(texture.width * texture.height * 3, 0);
    //glTextureSubImage2D(texture.id, 0, 0, 0, texture.width, texture.height, format, GL_UNSIGNED_BYTE, &emptyData);


    if (texture.levels > 1) {
        glGenerateTextureMipmap(texture.id);
    }
    return texture;
}

int MeshRendererContext::allocateTexture(int w, int h) {
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

void MeshRendererContext::readbuffer_meshrenderer_shadow_depth(int width, int height, GLuint fb2, GLuint texture_id) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
	glCopyTextureSubImage2D(texture_id, 0, 0, 0, 0, 0, width, height);
}

py::list MeshRendererContext::generateArrayTextures(std::vector<std::string> filenames, int texCutoff, bool shouldShrinkSmallTextures, int smallTexBucketSize) {
		int num_textures = filenames.size();
		std::vector<unsigned char*> image_data;
		std::vector<int> texHeights;
		std::vector<int> texWidths;
		std::vector<int> texChannels;

		printf("number of textures %d\n", num_textures);
		for (int i = 0; i < num_textures; i++) {
			std::string filename = filenames[i];
			std::cout << "Filename is: " << filename << std::endl;
			int w;
			int h;
			int comp;
			stbi_set_flip_vertically_on_load(true);
			unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb); // force to 3 channels
			if (image == nullptr)
				throw(std::string("Failed to load texture"));
			std::cout << "Size is w: " << w << " by h: " << h << std::endl;
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
			// We also put wall, floor and ceiling textures in the larger bucket since they cover a large surface area,
			// so should look good! Note: this method will also catch some objects that have they keywords in them,
			// but it should never be enough to overload VRAM :)
			std::string tex_filename = filenames[i];
			bool contains_keyword = tex_filename.find("floor") != std::string::npos || tex_filename.find("wall") != std::string::npos || tex_filename.find("ceiling") != std::string::npos;
			if (score >= texCutoff || contains_keyword) {
				std::cout << "Appending texture with name: " << tex_filename << " to large bucket" << std::endl;
				texIndices[0].push_back(i);
				tex_info_i.append(0);
				tex_info_i.append(firstTexLayerNum);
				if (w > texLayerDims[0]) texLayerDims[0] = w;
				if (h > texLayerDims[1]) texLayerDims[1] = h;
				firstTexLayerNum++;
			}
			else {
				std::cout << "Appending texture with name: " << tex_filename << " to small bucket" << std::endl;
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

			int out_w = texLayerDims[2 * static_cast<long long int>(i)];
			int out_h = texLayerDims[2 * static_cast<long long int>(i) + 1];


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
	py::list MeshRendererContext::renderSetup(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor,
		py::array_t<float> mergedVertexData, py::array_t<int> index_ptr_offsets, py::array_t<int> index_counts,
		py::array_t<int> indices, py::array_t<float> mergedFragData, py::array_t<float> mergedFragRMData,
		py::array_t<float> mergedFragNData,
		py::array_t<float> mergedDiffuseData,
		py::array_t<float> mergedPBRData,
		py::array_t<float> mergedHiddenData,
		py::array_t<float> mergedUVData,
		int tex_id_1, int tex_id_2, GLuint fb,
		float use_pbr, int depth_tex_id) {
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
		GLuint coordsAttrib = glGetAttribLocation(shaderProgram, "texCoords");
		GLuint tangentlAttrib = glGetAttribLocation(shaderProgram, "tangent");
		GLuint bitangentAttrib = glGetAttribLocation(shaderProgram, "bitangent");

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
		glEnableVertexAttribArray(4);

		glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void*)0);
		glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void*)12);
		glVertexAttribPointer(coordsAttrib, 2, GL_FLOAT, GL_TRUE, 56, (void*)24);
		glVertexAttribPointer(tangentlAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void*)32);
		glVertexAttribPointer(bitangentAttrib, 3, GL_FLOAT, GL_FALSE, 56, (void*)44);

		glBindVertexArray(0);

		multidrawCount = index_ptr_offsets.size();
		int* indexOffsetPtr = (int*)index_ptr_offsets.request().ptr;

		for (int i = 0; i < multidrawCount; i++) {
			unsigned int offset = (unsigned int)indexOffsetPtr[i];
			this->multidrawStartIndices.push_back(BUFFER_OFFSET((offset * sizeof(unsigned int))));
			//printf("multidraw start idx %d\n", offset);
		}

		// Store for rendering
		int* indices_count_ptr = (int*)index_counts.request().ptr;
		for (int i = 0; i < multidrawCount; i++) {
			this->multidrawCounts.push_back(indices_count_ptr[i]);
		}

		// Set up shaders
		float* fragData = (float*)mergedFragData.request().ptr;
		float* fragRMData = (float*)mergedFragRMData.request().ptr;
		float* fragNData = (float*)mergedFragNData.request().ptr;
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
		glUniform1f(glGetUniformLocation(shaderProgram, "use_two_light_probe"), (float)m_use_two_light_probe);

		printf("multidrawcount %d\n", multidrawCount);

		glGenBuffers(1, &uboTexColorData);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTexColorData);
		texColorDataSize = 4 * 16 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, texColorDataSize, NULL, GL_STATIC_DRAW);
		GLuint texColorDataIdx = glGetUniformBlockIndex(shaderProgram, "TexColorData");
		glUniformBlockBinding(shaderProgram, texColorDataIdx, 0);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboTexColorData);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, fragDataSize * sizeof(float), fragData);
		glBufferSubData(GL_UNIFORM_BUFFER, 16 * MAX_ARRAY_SIZE, fragDataSize * sizeof(float), fragRMData);
		glBufferSubData(GL_UNIFORM_BUFFER, 2 * 16 * MAX_ARRAY_SIZE, fragDataSize * sizeof(float), fragNData);
		glBufferSubData(GL_UNIFORM_BUFFER, 3 * 16 * MAX_ARRAY_SIZE, diffuseDataSize * sizeof(float), diffuseData);

		float* pbrData = (float*)mergedPBRData.request().ptr;
		int pbrDataSize = mergedPBRData.size();

		glGenBuffers(1, &uboPbrData);
		glBindBuffer(GL_UNIFORM_BUFFER, uboPbrData);
		int pbrDataMaxSize = 16 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, pbrDataMaxSize, NULL, GL_STATIC_DRAW);
		GLuint pbrDataIdx = glGetUniformBlockIndex(shaderProgram, "PBRData");
		glUniformBlockBinding(shaderProgram, pbrDataIdx, 1);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboPbrData);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, pbrDataSize * sizeof(float), pbrData);

		glGenBuffers(1, &uboTransformDataTrans);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataTrans);
		transformDataSize = 64 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, transformDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint transformDataTransIdx = glGetUniformBlockIndex(shaderProgram, "TransformDataTrans");
		glUniformBlockBinding(shaderProgram, transformDataTransIdx, 2);
		glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboTransformDataTrans);

		glGenBuffers(1, &uboTransformDataRot);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataRot);
		glBufferData(GL_UNIFORM_BUFFER, transformDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint transformDataRotIdx = glGetUniformBlockIndex(shaderProgram, "TransformDataRot");
		glUniformBlockBinding(shaderProgram, transformDataRotIdx, 3);
		glBindBufferBase(GL_UNIFORM_BUFFER, 3, uboTransformDataRot);

        glGenBuffers(1, &uboTransformDataLastTrans);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataLastTrans);
		transformDataSize = 64 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, transformDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint transformDataLastTransIdx = glGetUniformBlockIndex(shaderProgram, "TransformDataLastTrans");
		glUniformBlockBinding(shaderProgram, transformDataLastTransIdx, 6);
		glBindBufferBase(GL_UNIFORM_BUFFER, 6, uboTransformDataLastTrans);

		glGenBuffers(1, &uboTransformDataLastRot);
		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataLastRot);
		glBufferData(GL_UNIFORM_BUFFER, transformDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint transformDataLastRotIdx = glGetUniformBlockIndex(shaderProgram, "TransformDataLastRot");
		glUniformBlockBinding(shaderProgram, transformDataLastRotIdx, 7);
		glBindBufferBase(GL_UNIFORM_BUFFER, 7, uboTransformDataLastRot);

		float *hiddenData = (float*)mergedHiddenData.request().ptr;
		int hiddenDataSize = mergedHiddenData.size();

		glGenBuffers(1, &uboHidden);
		glBindBuffer(GL_UNIFORM_BUFFER, uboHidden);
		int hiddenDataMaxSize = 16 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, hiddenDataMaxSize, NULL, GL_DYNAMIC_DRAW);
		GLuint hiddenIdx = glGetUniformBlockIndex(shaderProgram, "Hidden");
		glUniformBlockBinding(shaderProgram, hiddenIdx, 4);
		glBindBufferBase(GL_UNIFORM_BUFFER, 4, uboHidden);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, hiddenDataSize * sizeof(float), hiddenData);

		float* uvData = (float*)mergedUVData.request().ptr;
		int uvDataSize = mergedUVData.size();

		glGenBuffers(1, &uboUV);
		glBindBuffer(GL_UNIFORM_BUFFER, uboUV);
		int uvMaxDataSize = 16 * MAX_ARRAY_SIZE;
		glBufferData(GL_UNIFORM_BUFFER, uvMaxDataSize, NULL, GL_DYNAMIC_DRAW);
		GLuint uvIdx = glGetUniformBlockIndex(shaderProgram, "UVData");
		glUniformBlockBinding(shaderProgram, uvIdx, 5);
		glBindBufferBase(GL_UNIFORM_BUFFER, 5, uboUV);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, uvDataSize * sizeof(float), uvData);

		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		GLuint bigTexLoc = glGetUniformLocation(shaderProgram, "bigTex");
		GLuint smallTexLoc = glGetUniformLocation(shaderProgram, "smallTex");

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id_1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id_2);

		glActiveTexture(GL_TEXTURE2);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTexture.id);

		glActiveTexture(GL_TEXTURE3);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_CUBE_MAP, m_irmapTexture.id);

		glActiveTexture(GL_TEXTURE4);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_2D, m_spBRDF_LUT.id);

		glActiveTexture(GL_TEXTURE5);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTexture2.id);

		glActiveTexture(GL_TEXTURE6);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_CUBE_MAP, m_irmapTexture2.id);

		glActiveTexture(GL_TEXTURE7);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_2D, m_spBRDF_LUT2.id);

		glActiveTexture(GL_TEXTURE8);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_2D, m_default_metallic_texture.id);

		glActiveTexture(GL_TEXTURE9);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_2D, m_default_roughness_texture.id);

		glActiveTexture(GL_TEXTURE10);
		if (use_pbr == 1) glBindTexture(GL_TEXTURE_2D, m_default_normal_texture.id);

		glActiveTexture(GL_TEXTURE11);
		if (m_use_two_light_probe) {
			glBindTexture(GL_TEXTURE_2D, m_light_modulation_map.id);
		}

		glActiveTexture(GL_TEXTURE12);
		glBindTexture(GL_TEXTURE_2D, depth_tex_id);

		glUniform1i(bigTexLoc, 0);
		glUniform1i(smallTexLoc, 1);
		glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture"), 2);
		glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture"), 3);
		glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT"), 4);

		glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture2"), 5);
		glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture2"), 6);
		glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT2"), 7);

		glUniform1i(glGetUniformLocation(shaderProgram, "defaultMetallicTexture"), 8);
		glUniform1i(glGetUniformLocation(shaderProgram, "defaultRoughnessTexture"), 9);
		glUniform1i(glGetUniformLocation(shaderProgram, "defaultNormalTexture"), 10);

		glUniform1i(glGetUniformLocation(shaderProgram, "lightModulationMap"), 11);

		glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 12);

		glUseProgram(0);

		py::list renderData;
		renderData.append(VAO);
		renderData.append(VBO);
		renderData.append(EBO);

		return renderData;
	}

	// Updates hidden states in vertex shader
	void MeshRendererContext::updateHiddenData(int shaderProgram, py::array_t<float> hidden_array) {
		glUseProgram(shaderProgram);

		float* hiddenData = (float*)hidden_array.request().ptr;
		int hiddenDataSize = hidden_array.size();

		glBindBuffer(GL_UNIFORM_BUFFER, uboHidden);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, hiddenDataSize * sizeof(float), hiddenData);

		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	// Updates UV data in vertex shader
	void MeshRendererContext::updateUVData(int shaderProgram, py::array_t<float> uv_data) {
		glUseProgram(shaderProgram);

		float* uvData = (float*)uv_data.request().ptr;
		int uvDataSize = uv_data.size();

		glBindBuffer(GL_UNIFORM_BUFFER, uboUV);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, uvDataSize * sizeof(float), uvData);

		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	// Updates positions and rotations in vertex shader
	void MeshRendererContext::updateDynamicData(int shaderProgram, py::array_t<float> pose_trans_array,
	py::array_t<float> pose_rot_array, py::array_t<float> last_trans_array,
	py::array_t<float> last_rot_array, py::array_t<float> V, py::array_t<float> last_V, py::array_t<float> P,
	py::array_t<float> lightV,
	py::array_t<float> lightP, int shadow_pass,
		py::array_t<float> eye_pos) {
		glUseProgram(shaderProgram);

		float* transPtr = (float*)pose_trans_array.request().ptr;
		float* rotPtr = (float*)pose_rot_array.request().ptr;
		int transDataSize = pose_trans_array.size();
		int rotDataSize = pose_rot_array.size();

        if (transDataSize > MAX_ARRAY_SIZE * 16) transDataSize = MAX_ARRAY_SIZE * 16;
        if (rotDataSize > MAX_ARRAY_SIZE * 16) rotDataSize = MAX_ARRAY_SIZE * 16;

		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataTrans);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, transDataSize * sizeof(float), transPtr);

		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataRot);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, rotDataSize * sizeof(float), rotPtr);


        float* lastTransPtr = (float*)last_trans_array.request().ptr;
		float* lastRotPtr = (float*)last_rot_array.request().ptr;
		transDataSize = last_trans_array.size();
		rotDataSize = last_rot_array.size();

        if (transDataSize > MAX_ARRAY_SIZE * 16) transDataSize = MAX_ARRAY_SIZE * 16;
        if (rotDataSize > MAX_ARRAY_SIZE * 16) rotDataSize = MAX_ARRAY_SIZE * 16;

		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataLastTrans);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, transDataSize * sizeof(float), lastTransPtr);

		glBindBuffer(GL_UNIFORM_BUFFER, uboTransformDataLastRot);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, rotDataSize * sizeof(float), lastRotPtr);

		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		float* Vptr = (float*)V.request().ptr;
        float *last_Vptr = (float *) last_V.request().ptr;
		float* Pptr = (float*)P.request().ptr;
		float* lightVptr = (float*)lightV.request().ptr;
		float* lightPptr = (float*)lightP.request().ptr;
        float *eye_pos_ptr = (float *) eye_pos.request().ptr;

		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "last_V"), 1, GL_TRUE, last_Vptr);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightP"), 1, GL_FALSE, lightPptr);
		glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), shadow_pass);
        glUniform3f(glGetUniformLocation(shaderProgram, "eyePosition"), eye_pos_ptr[0], eye_pos_ptr[1], eye_pos_ptr[2]);
	}

	// Optimized rendering function that is called once per frame for all merged data
	void MeshRendererContext::renderOptimized(GLuint VAO) {
		glBindVertexArray(VAO);
		int draw_count = this->multidrawCount;
		if (draw_count > MAX_ARRAY_SIZE) {
		    draw_count = MAX_ARRAY_SIZE;
		    printf("Warning: not all objects are drawn\n");
		}
		glMultiDrawElements(GL_TRIANGLES, &this->multidrawCounts[0], GL_UNSIGNED_INT, &this->multidrawStartIndices[0], draw_count);
	}

void MeshRendererContext::clean_meshrenderer_optimized(std::vector<GLuint> color_attachments, std::vector<GLuint> textures, std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos, std::vector<GLuint> ebos) {
		glDeleteTextures(color_attachments.size(), color_attachments.data());
		glDeleteTextures(textures.size(), textures.data());
		glDeleteFramebuffers(fbo.size(), fbo.data());
		glDeleteBuffers(vaos.size(), vaos.data());
		glDeleteBuffers(vbos.size(), vbos.data());
		glDeleteBuffers(ebos.size(), ebos.data());
		glDeleteBuffers(1, &uboTexColorData);
		glDeleteBuffers(1, &uboPbrData);
		glDeleteBuffers(1, &uboTransformDataTrans);
		glDeleteBuffers(1, &uboTransformDataRot);
		glDeleteBuffers(1, &uboHidden);
		// Delete skybox VAO and buffers
		glDeleteVertexArrays(1, &m_skybox_vao);
		glDeleteBuffers(1, &m_skybox_vbo);
	}


void MeshRendererContext::loadSkyBox(int shaderProgram, float skybox_size) {
	// First set up VAO and corresponding attributes
	glGenVertexArrays(1, &m_skybox_vao);
	glBindVertexArray(m_skybox_vao);

	float cube_vertices[] = {
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
	};
	// Scale cube vertices up to skybox size
	for (int i = 0; i < 108; i++) cube_vertices[i] *= skybox_size;
	glGenBuffers(1, &m_skybox_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_skybox_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
	GLuint vertexAttrib = glGetAttribLocation(shaderProgram, "position");
	glEnableVertexAttribArray(vertexAttrib);
	glVertexAttribPointer(vertexAttrib, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

void MeshRendererContext::renderSkyBox(int shaderProgram, py::array_t<float> V, py::array_t<float> P){
    glUseProgram(shaderProgram);
    float* Vptr = (float*)V.request().ptr;
    float* Pptr = (float*)P.request().ptr;

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_envTexture3.id);
    glUniform1i(glGetUniformLocation(shaderProgram, "envTexture"), 0);

	glBindVertexArray(m_skybox_vao);
	glDrawArrays(GL_TRIANGLES, 0, 36);
}