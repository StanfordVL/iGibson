#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
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

#include "mesh_renderer.h"

namespace py = pybind11;


class Image {
public:
    static std::shared_ptr<Image> fromFile(const std::string &filename, int channels) {
        std::printf("Loading image: %s\n", filename.c_str());

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

    for (int i = 0; i < 4; i++) {
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
    } else {
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
}

py::list MeshRendererContext::setup_framebuffer_meshrenderer(int width, int height) {
    GLuint *fbo_ptr = (GLuint *) malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint *) malloc(5 * sizeof(GLuint));
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
    GLenum *bufs = (GLenum *) malloc(4 * sizeof(GLenum));
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

py::list MeshRendererContext::setup_framebuffer_meshrenderer_ms(int width, int height) {
    GLuint *fbo_ptr = (GLuint *) malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint *) malloc(5 * sizeof(GLuint));
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
    GLenum *bufs = (GLenum *) malloc(4 * sizeof(GLenum));
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

void MeshRendererContext::initvar_instance(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV,
                                           int shadow_pass, py::array_t<float> P, py::array_t<float> eye_pos,
                                           py::array_t<float> pose_trans,
                                           py::array_t<float> pose_rot, py::array_t<float> lightpos,
                                           py::array_t<float> lightcolor) {
    glUseProgram(shaderProgram);
    float *Vptr = (float *) V.request().ptr;
    float *lightVptr = (float *) lightV.request().ptr;
    float *Pptr = (float *) P.request().ptr;
    float *transptr = (float *) pose_trans.request().ptr;
    float *rotptr = (float *) pose_rot.request().ptr;
    float *lightposptr = (float *) lightpos.request().ptr;
    float *lightcolorptr = (float *) lightcolor.request().ptr;
    float *eye_pos_ptr = (float *) eye_pos.request().ptr;

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "eyePosition"), eye_pos_ptr[0], eye_pos_ptr[1], eye_pos_ptr[2]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1],
                lightcolorptr[2]);
    glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), shadow_pass);

}


void
MeshRendererContext::init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color,
                                            float use_texture, float use_pbr, float use_pbr_mapping, float metallic,
                                            float roughness) {
    float *diffuse_ptr = (float *) diffuse_color.request().ptr;
    glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr"), use_pbr);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr_mapping"), use_pbr_mapping);
    glUniform1f(glGetUniformLocation(shaderProgram, "metallic"), metallic);
    glUniform1f(glGetUniformLocation(shaderProgram, "roughness"), roughness);
    glUniform1i(glGetUniformLocation(shaderProgram, "texUnit"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture"), 2);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT"), 3);
    glUniform1i(glGetUniformLocation(shaderProgram, "metallicTexture"), 4);
    glUniform1i(glGetUniformLocation(shaderProgram, "roughnessTexture"), 5);
    glUniform1i(glGetUniformLocation(shaderProgram, "normalTexture"), 6);
    glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 7);
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

    if (metallic_texture_id != -1) {
        glActiveTexture(GL_TEXTURE4);
        if (flag) glBindTexture(GL_TEXTURE_2D, metallic_texture_id);
    }

    if (roughness_texture_id != -1) {
        glActiveTexture(GL_TEXTURE5);
        if (flag) glBindTexture(GL_TEXTURE_2D, roughness_texture_id);
    }

    if (normal_texture_id != -1) {
        glActiveTexture(GL_TEXTURE6);
        if (flag) glBindTexture(GL_TEXTURE_2D, normal_texture_id);
    }

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, depth_texture_id);

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

void MeshRendererContext::initvar_instance_group(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV,
                                                 int shadow_pass, py::array_t<float> P, py::array_t<float> eye_pos,
                                                 py::array_t<float> lightpos, py::array_t<float> lightcolor) {
    glUseProgram(shaderProgram);
    float *Vptr = (float *) V.request().ptr;
    float *lightVptr = (float *) lightV.request().ptr;
    float *Pptr = (float *) P.request().ptr;
    float *lightposptr = (float *) lightpos.request().ptr;
    float *lightcolorptr = (float *) lightcolor.request().ptr;
    float *eye_pos_ptr = (float *) eye_pos.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightV"), 1, GL_TRUE, lightVptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "eyePosition"), eye_pos_ptr[0], eye_pos_ptr[1], eye_pos_ptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1],
                lightcolorptr[2]);
    glUniform1i(glGetUniformLocation(shaderProgram, "shadow_pass"), shadow_pass);
}

void MeshRendererContext::init_material_pos_instance(int shaderProgram, py::array_t<float> pose_trans,
                                                     py::array_t<float> pose_rot,
                                                     float instance_color, py::array_t<float> diffuse_color,
                                                     float use_texture, float use_pbr, float use_pbr_mapping,
                                                     float metalness,
                                                     float roughness) {
    float *transptr = (float *) pose_trans.request().ptr;
    float *rotptr = (float *) pose_rot.request().ptr;
    float *diffuse_ptr = (float *) diffuse_color.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr"), use_pbr);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_pbr_mapping"), use_pbr_mapping);
    glUniform1f(glGetUniformLocation(shaderProgram, "metalness"), metalness);
    glUniform1f(glGetUniformLocation(shaderProgram, "roughness"), roughness);
    glUniform1i(glGetUniformLocation(shaderProgram, "texUnit"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram, "irradianceTexture"), 2);
    glUniform1i(glGetUniformLocation(shaderProgram, "specularBRDF_LUT"), 3);
    glUniform1i(glGetUniformLocation(shaderProgram, "metallicTexture"), 4);
    glUniform1i(glGetUniformLocation(shaderProgram, "roughnessTexture"), 5);
    glUniform1i(glGetUniformLocation(shaderProgram, "normalTexture"), 6);
    glUniform1i(glGetUniformLocation(shaderProgram, "depthMap"), 7);

}


void MeshRendererContext::render_tensor_pre(bool msaa, GLuint fb1, GLuint fb2) {

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


void MeshRendererContext::render_tensor_post() {
    glDisable(GL_DEPTH_TEST);
}

void MeshRendererContext::cglBindVertexArray(int vao) {
    glBindVertexArray(vao);
}

void MeshRendererContext::cglUseProgram(int shaderProgram) {
    glUseProgram(shaderProgram);
}

int MeshRendererContext::loadTexture(std::string filename) {
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

void MeshRendererContext::setup_pbr(std::string shader_path, std::string env_texture_filename) {

    glEnable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glFrontFace(GL_CCW);


    envTextureUnfiltered = createTexture(GL_TEXTURE_CUBE_MAP, kEnvMapSize, kEnvMapSize, GL_RGBA16F, 0);

    // Load & convert equirectangular environment map to a cubemap texture.
    {
        GLuint equirectToCubeProgram = linkProgram({
                                                           compileShader(shader_path + "equirect2cube_cs.glsl",
                                                                         GL_COMPUTE_SHADER)
                                                   });

        envTextureEquirect = createTexture(Image::fromFile(env_texture_filename, 3), GL_RGB, GL_RGB16F, 1);

        glUseProgram(equirectToCubeProgram);
        glBindTextureUnit(0, envTextureEquirect.id);
        glBindImageTexture(0, envTextureUnfiltered.id, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glDispatchCompute(envTextureUnfiltered.width / 32, envTextureUnfiltered.height / 32, 6);

        //glDeleteTextures(1, &envTextureEquirect.id);
        glDeleteProgram(equirectToCubeProgram);
    }
    glGenerateTextureMipmap(envTextureUnfiltered.id);

    {
        GLuint spmapProgram = linkProgram({
                                                  compileShader(shader_path + "/450/spmap_cs.glsl", GL_COMPUTE_SHADER)
                                          });

        m_envTexture = createTexture(GL_TEXTURE_CUBE_MAP, kEnvMapSize, kEnvMapSize, GL_RGBA16F, 0);

        // Copy 0th mipmap level into destination environment map.
        glCopyImageSubData(envTextureUnfiltered.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           m_envTexture.id, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
                           m_envTexture.width, m_envTexture.height, 6);

        glUseProgram(spmapProgram);
        glBindTextureUnit(0, envTextureUnfiltered.id);

        // Pre-filter rest of the mip chain.
        const float deltaRoughness = 1.0f / glm::max(float(m_envTexture.levels - 1), 1.0f);
        for (int level = 1, size = kEnvMapSize / 2; level <= m_envTexture.levels; ++level, size /= 2) {
            const GLuint numGroups = glm::max(1, size / 32);
            glBindImageTexture(0, m_envTexture.id, level, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            glProgramUniform1f(spmapProgram, 0, level * deltaRoughness);
            glDispatchCompute(numGroups, numGroups, 6);
        }
        glDeleteProgram(spmapProgram);
    }

    //glDeleteTextures(1, &envTextureUnfiltered.id);

    // Compute diffuse irradiance cubemap.
    {
        GLuint irmapProgram = linkProgram({
                                                  compileShader(shader_path + "/450/irmap_cs.glsl", GL_COMPUTE_SHADER)
                                          });

        m_irmapTexture = createTexture(GL_TEXTURE_CUBE_MAP, kIrradianceMapSize, kIrradianceMapSize, GL_RGBA16F, 1);

        glUseProgram(irmapProgram);
        glBindTextureUnit(0, m_envTexture.id);
        glBindImageTexture(0, m_irmapTexture.id, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glDispatchCompute(m_irmapTexture.width / 32, m_irmapTexture.height / 32, 6);
        glDeleteProgram(irmapProgram);
    }

    // Compute Cook-Torrance BRDF 2D LUT for split-sum approximation.
    {
        GLuint spBRDFProgram = linkProgram({
                                                   compileShader(shader_path + "/450/spbrdf_cs.glsl", GL_COMPUTE_SHADER)
                                           });

        m_spBRDF_LUT = createTexture(GL_TEXTURE_2D, kBRDF_LUT_Size, kBRDF_LUT_Size, GL_RG16F, 1);
        glTextureParameteri(m_spBRDF_LUT.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(m_spBRDF_LUT.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glUseProgram(spBRDFProgram);
        glBindImageTexture(0, m_spBRDF_LUT.id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
        glDispatchCompute(m_spBRDF_LUT.width / 32, m_spBRDF_LUT.height / 32, 1);
        glDeleteProgram(spBRDFProgram);
    }

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


Texture
MeshRendererContext::createTexture(GLenum target, int width, int height, GLenum internalformat, int levels) const {
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.levels = (levels > 0) ? levels : 6;

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


py::array_t<float>
MeshRendererContext::readbuffer_meshrenderer_shadow_depth(int width, int height, GLuint fb2, GLuint texture_id) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    py::array_t<float> data = py::array_t<float>(3 * width * height);
    py::buffer_info buf = data.request();
    float *ptr = (float *) buf.ptr;
    glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, ptr);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, ptr);
    return data;
}
