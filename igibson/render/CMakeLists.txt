cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)


set(USE_GLAD TRUE)

if (MAC_PLATFORM)
    set(USE_CUDA FALSE)
    set(USE_GLFW TRUE)
else ()
    set(USE_CUDA TRUE)
    set(USE_GLFW FALSE)
endif ()

include_directories(glad)

if (NOT USE_GLAD)
    find_package(OpenGL)
else ()
    add_definitions(-DUSE_GLAD)
endif ()

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(OpenGL)
    #OpenGL is still needed for cuda-gl interoperation
endif ()

add_subdirectory(pybind11)
# Add GLM include directory
include_directories("glm")

if (USE_GLFW)
    # Find GLFW and OpenVR
    set(GLFW_DIR glfw)
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
    set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
    add_subdirectory("${GLFW_DIR}")
    include_directories("${GLFW_DIR}/include")
endif ()

if (USE_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
    cuda_add_library(EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/mesh_renderer.cpp cpp/egl_mesh_renderer.cpp)
else ()
    add_library(EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/mesh_renderer.cpp cpp/egl_mesh_renderer.cpp)
endif ()

add_library(tinyobjloader MODULE cpp/tinyobjloader/tiny_obj_loader.cc cpp/tinyobjloader/bindings.cc)
if (USE_GLFW)
    add_library(GLFWRendererContext MODULE glad/gl.cpp cpp/glfw_mesh_renderer.cpp cpp/mesh_renderer.cpp)
endif ()


if (USE_GLAD)
    target_link_libraries(EGLRendererContext PRIVATE pybind11::module dl pthread)
    if (USE_GLFW)
        target_link_libraries(GLFWRendererContext PRIVATE pybind11::module dl glfw ${GLFW_LIBRARIES})
    endif ()
else ()
    target_link_libraries(EGLRendererContext PRIVATE pybind11::module dl pthread EGL ${OPENGL_LIBRARIES})
    if (USE_GLFW)
        target_link_libraries(GLFWRendererContext PRIVATE pybind11::module dl glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})
    endif ()
endif ()

target_link_libraries(tinyobjloader PRIVATE pybind11::module)

set_target_properties(EGLRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

if (USE_GLFW)
    set_target_properties(GLFWRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif ()

set_target_properties(tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)

if (USE_GLAD)
    target_link_libraries(query_devices dl pthread)
    target_link_libraries(test_device dl pthread)
else ()
    target_link_libraries(query_devices dl pthread EGL ${OPENGL_LIBRARIES})
    target_link_libraries(test_device dl pthread EGL ${OPENGL_LIBRARIES})
endif ()
