cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)

set (USE_GLAD TRUE)

#if (MAC_PLATFORM)
  set (USE_CUDA FALSE)
  set (USE_GLFW TRUE)
#else()
#  set (USE_CUDA TRUE)
#  set (USE_GLFW FALSE)
#endif()

# This CMake file is currently Windows/VR-specific

include_directories(glad)

find_package(OpenGL REQUIRED)

add_subdirectory(pybind11)

# Add GLM include directory
include_directories("glm")

# Find GLFW and OpenVR
set(GLFW_DIR glfw)
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
add_subdirectory("${GLFW_DIR}")
include_directories("${GLFW_DIR}/include")

set(WINDOWS_PATH_SUFFIXES win64 Win64 x64)
set(OPENVR_DIR openvr)
find_library(OPENVR_LIBRARIES
	NAMES
		openvr_api
	PATHS
		"${OPENVR_DIR}/bin"
		"${OPENVR_DIR}/lib"
	PATH_SUFFIXES
    	${WINDOWS_PATH_SUFFIXES}
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

set(OPENVR_INCLUDE_DIR "${OPENVR_DIR}/headers")
include_directories("${OPENVR_INCLUDE_DIR}")

add_library(MeshRendererContext MODULE glad/gl.cpp cpp/Mesh_renderer.cpp)
add_library(tinyobjloader MODULE cpp/tinyobjloader/tiny_obj_loader.cc cpp/tinyobjloader/bindings.cc)

target_link_libraries(MeshRendererContext PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OPENVR_LIBRARIES})
target_link_libraries(tinyobjloader PRIVATE pybind11::module)

set_target_properties(MeshRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
