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

struct Texture {
    Texture() : id(0) {}

    GLuint id;
    int width, height;
    int levels;
};


class MeshRendererContext {
public:
    MeshRendererContext(int w, int h) : m_windowHeight(h), m_windowWidth(w) {};

    int m_windowWidth;
    int m_windowHeight;
    int verbosity;

    const int kEnvMapSize = 256;
    const int kSkyBoxMapSize = 1024;
    const int kIrradianceMapSize = 32;
    const int kBRDF_LUT_Size = 256;

    Texture m_envTexture;  // indoor 1
    Texture m_irmapTexture; // indoor 1
    Texture m_spBRDF_LUT; // indoor 1

    Texture m_envTexture2; // indoor 2
    Texture m_irmapTexture2; // indoor 2
    Texture m_spBRDF_LUT2; // indoor 2

    Texture m_envTexture3; // outdoor

    Texture m_light_modulation_map; // modulate indoor 1 and indoor 2

    Texture m_default_metallic_texture;
    Texture m_default_roughness_texture;
    Texture m_default_normal_texture;

    bool m_use_two_light_probe = false;

    Texture envTextureEquirect;
    Texture envTextureUnfiltered;

    // Index data
	std::vector<void*> multidrawStartIndices;
	std::vector<int> multidrawCounts;
	int multidrawCount;

	// UBO data
	GLuint uboTexColorData;
	GLuint uboPbrData;
	GLuint uboTransformDataRot;
	GLuint uboTransformDataTrans;
    GLuint uboTransformDataLastRot;
	GLuint uboTransformDataLastTrans;
	GLuint uboHidden;
	GLuint uboUV;

	int texColorDataSize;
	int transformDataSize;

	// Skybox data
	GLuint m_skybox_vao;
	GLuint m_skybox_vbo;

#ifdef USE_CUDA
    cudaGraphicsResource* cuda_res[MAX_NUM_RESOURCES];
#endif

    int init() {};

    void release() {};

    static int numMipmapLevels(int width, int height);


#ifdef USE_CUDA
    void map_tensor(GLuint tid, int width, int height, std::size_t data);

    void map_tensor_float(GLuint tid, int width, int height, std::size_t data);
#endif

    void render_meshrenderer_pre(bool msaa, GLuint fb1, GLuint fb2);

    void render_meshrenderer_post();

    std::string getstring_meshrenderer();

    void blit_buffer(int width, int height, GLuint fb1, GLuint fb2);

    py::array_t<float> readbuffer_meshrenderer(char *mode, int width, int height, GLuint fb2);

    void clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2, std::vector<GLuint> fbo,
                            std::vector<GLuint> vaos, std::vector<GLuint> vbos);

    void clean_meshrenderer_optimized(std::vector<GLuint> color_attachments, std::vector<GLuint> textures,
    std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos, std::vector<GLuint> ebos);

    py::list setup_framebuffer_meshrenderer(int width, int height);

    py::list setup_framebuffer_meshrenderer_ms(int width, int height);

    int compile_shader_meshrenderer(char *vertexShaderSource, char *fragmentShaderSource);

    py::list load_object_meshrenderer(int shaderProgram, py::array_t<float> vertexData);

    void render_softbody_instance(int vao, int vbo, py::array_t<float> vertexData);

    void init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color,
                                float use_texture, float use_pbr, float use_pbr_mapping, float metallic,
                                float roughness, py::array_t<float> transform_param);

    void draw_elements_instance(bool flag, int texture_id, int metallic_texture_id, int roughness_texture_id,
                                int normal_texture_id, int depth_texture_id, int vao, int face_size,
                                py::array_t<unsigned int> faces, GLuint fb);

    void initvar(int shaderProgram, py::array_t<float> V, py::array_t<float> last_V, py::array_t<float> lightV, int
                                shadow_pass, py::array_t<float> P, py::array_t<float> lightP, py::array_t<float>
                                eye_pos, py::array_t<float> lightpos, py::array_t<float> lightcolor);

    void init_pos_instance(int shaderProgram, py::array_t<float> pose_trans, py::array_t<float> pose_rot,
                           py::array_t<float> last_trans, py::array_t<float> last_rot);

    void cglBindVertexArray(int vao);

    void cglUseProgram(int shaderProgram);

    int loadTexture(std::string filename, float texture_scale);

    void setup_pbr(std::string shader_path,
    std::string env_texture_filename,
    std::string env_texture_filename2,
    std::string env_texture_filename3,
    std::string light_modulation_map_filename,
    float light_dimming_factor);

    void generate_light_maps(
    GLuint equirectToCubeProgram,
    GLuint spmapProgram,
    GLuint irmapProgram,
    GLuint spBRDFProgram,
    std::string env_texture_filename,
    Texture& envTexture,
    Texture& irmapTexture,
    Texture& spBRDF_LUT,
    float light_dimming_factor
    );

    void generate_env_map(
    GLuint equirectToCubeProgram,
    std::string env_texture_filename,
    Texture& envTexture
    );

    GLuint linkProgram(std::initializer_list<GLuint> shaders);

    std::string readText(const std::string &filename);

    GLuint compileShader(const std::string &filename, GLenum type);

    Texture createTexture(GLenum target, int width, int height, GLenum internalformat, int levels) const;

    Texture
    createTexture(const std::shared_ptr<class Image> &image, GLenum format, GLenum internalformat, int levels) const;

    int allocateTexture(int w, int h);

    void readbuffer_meshrenderer_shadow_depth(int width, int height, GLuint fb2, GLuint texture_id);

    // Generates large and small array textures and returns handles to the user (cutoff based on user variable), as well as index - tex num/layer mapping
	py::list generateArrayTextures(std::vector<std::string> filenames, int texCutoff, bool shouldShrinkSmallTextures,
	int smallTexBucketSize);

	py::list renderSetup(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor,
		py::array_t<float> mergedVertexData, py::array_t<int> index_ptr_offsets, py::array_t<int> index_counts,
		py::array_t<int> indices, py::array_t<float> mergedFragData, py::array_t<float> mergedFragRMData,
		py::array_t<float> mergedFragNData,
		py::array_t<float> mergedDiffuseData,
		py::array_t<float> mergedPBRData,
		py::array_t<float> mergedHiddenData,
		py::array_t<float> mergedUVData,
		int tex_id_1, int tex_id_2, GLuint fb,
		float use_pbr,
		int depth_tex_id);

	void updateHiddenData(int shaderProgram, py::array_t<float> hidden_array);

	void updateUVData(int shaderProgram, py::array_t<float> uv_data);

    void updateDynamicData(int shaderProgram, py::array_t<float> pose_trans_array,
        py::array_t<float> pose_rot_array, py::array_t<float> last_trans_array,
        py::array_t<float> last_rot_array, py::array_t<float> V, py::array_t<float> last_V, py::array_t<float> P,
        py::array_t<float> lightV,
        py::array_t<float> lightP, int shadow_pass,
		py::array_t<float> eye_pos);

	void renderOptimized(GLuint VAO);

	void loadSkyBox(int shaderProgram, float skybox_size);
	void renderSkyBox(int shaderProgram, py::array_t<float> V, py::array_t<float> P);
};


#endif