#ifndef MESH_RENDERER_HEADER
#define MESH_RENDERER_HEADER


#include  <glad/gl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
  #include <cuda_gl_interop.h>
#endif

#define MAX_NUM_RESOURCES 10

namespace py = pybind11;

struct Texture
{
    Texture() : id(0) {}
    GLuint id;
    int width, height;
    int levels;
};


class MeshRendererContext{
public:
    MeshRendererContext(int w, int h):m_windowHeight(h),m_windowWidth(w){};

    int m_windowWidth;
    int m_windowHeight;
    int verbosity;

    const int kEnvMapSize = 256;
    const int kIrradianceMapSize = 32;
    const int kBRDF_LUT_Size = 256;

    Texture m_envTexture;
    Texture m_irmapTexture;
    Texture m_spBRDF_LUT;
    Texture envTextureEquirect;
    Texture envTextureUnfiltered;


#ifdef USE_CUDA
    cudaGraphicsResource* cuda_res[MAX_NUM_RESOURCES];
#endif

    int init(){};
    void release(){};

#ifdef USE_CUDA
    void map_tensor(GLuint tid, int width, int height, std::size_t data);

    void map_tensor_float(GLuint tid, int width, int height, std::size_t data);
#endif

    void render_meshrenderer_pre(bool msaa, GLuint fb1, GLuint fb2);

    void render_meshrenderer_post();

    std::string getstring_meshrenderer();

    void blit_buffer(int width, int height, GLuint fb1, GLuint fb2);

    py::array_t<float> readbuffer_meshrenderer(char* mode, int width, int height, GLuint fb2);

    void clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2, std::vector<GLuint> fbo,
            std::vector<GLuint> vaos, std::vector<GLuint> vbos);

    py::list setup_framebuffer_meshrenderer(int width, int height);

    py::list setup_framebuffer_meshrenderer_ms(int width, int height);

    int compile_shader_meshrenderer(char* vertexShaderSource, char* fragmentShaderSource);

    py::list load_object_meshrenderer(int shaderProgram, py::array_t<float> vertexData);

    void render_softbody_instance(int vao, int vbo, py::array_t<float> vertexData);

    void initvar_instance(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV, int shadow_pass, py::array_t<float> P,
                          py::array_t<float> eye_pos, py::array_t<float> pose_trans, py::array_t<float> pose_rot,
                          py::array_t<float> lightpos, py::array_t<float> lightcolor);

    void init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color,
                                float use_texture, float use_pbr, float metallic, float roughness);

    void draw_elements_instance(bool flag, int texture_id, int metallic_texture_id, int roughness_texture_id,
                                int normal_texture_id, int depth_texture_id, int vao, int face_size, py::array_t<unsigned int> faces, GLuint fb);

    void initvar_instance_group(int shaderProgram, py::array_t<float> V, py::array_t<float> lightV, int shadow_pass,
            py::array_t<float> P, py::array_t<float> eye_pos, py::array_t<float> lightpos, py::array_t<float> lightcolor);

    void init_material_pos_instance(int shaderProgram, py::array_t<float> pose_trans, py::array_t<float> pose_rot,
                               float instance_color, py::array_t<float> diffuse_color, float use_texture,
                               float use_pbr, float metalness, float roughness);

    void render_tensor_pre(bool msaa, GLuint fb1, GLuint fb2);
    void render_tensor_post();
    void cglBindVertexArray(int vao);
    void cglUseProgram(int shaderProgram);
    int loadTexture(std::string filename);

    void setup_pbr(std::string shader_path, std::string env_texture_filename);

    GLuint linkProgram(std::initializer_list<GLuint> shaders);
    std::string readText(const std::string& filename);
    GLuint compileShader(const std::string& filename, GLenum type);

    Texture createTexture(GLenum target, int width, int height, GLenum internalformat, int levels) const;
    Texture createTexture(const std::shared_ptr<class Image>& image, GLenum format, GLenum internalformat, int levels) const;

    int allocateTexture(int w, int h);
    py::array_t<float> readbuffer_meshrenderer_shadow_depth(int width, int height, GLuint fb2, GLuint texture_id);

};


#endif