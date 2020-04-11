cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)
set (USE_GLAD TRUE)
set (USE_CUDA TRUE)
set (USE_GLFW FALSE)

include_directories(glad)

find_package(OpenGL REQUIRED)

if (USE_CUDA)
  add_definitions(-DUSE_CUDA) 
  find_package(OpenGL) 
  #OpenGL is still needed for cuda-gl interoperation
endif()

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

#find_package(CUDA REQUIRED)
#set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

#cuda_add_library(MeshRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/Mesh_renderer.cpp)
add_library(CGLUtils MODULE glad/gl.cpp cpp/cgl_utils.cpp)
add_library(tinyobjloader MODULE cpp/tinyobjloader/tiny_obj_loader.cc cpp/tinyobjloader/bindings.cc)
#add_library(GLFWRendererContext MODULE glad/gl.cpp cpp/glfw_mesh_renderer.cpp)
#add_library(VRUtils MODULE glad/gl.cpp cpp/vr_utils.cpp)

# TODO: Got rid of use_glad if statement - could add this back in at some point
#target_link_libraries(MeshRendererContext PRIVATE pybind11::module ${CMAKE_DL_LIBS} pthread  EGL ${OPENGL_LIBRARIES})
target_link_libraries(CGLUtils PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OPENVR_LIBRARIES})
#target_link_libraries(GLFWRendererContext PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})
#target_link_libraries(VRUtils PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENVR_LIBRARIES} ${OPENGL_LIBRARIES})

target_link_libraries(tinyobjloader PRIVATE pybind11::module)

#set_target_properties(MeshRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         #SUFFIX "${PYTHON_MODULE_EXTENSION}")
#set_target_properties(GLFWRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         #SUFFIX "${PYTHON_MODULE_EXTENSION}")
#set_target_properties(VRUtils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         #SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(CGLUtils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

#add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
#add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)

#if (USE_GLAD)
#  target_link_libraries(query_devices dl pthread)
#  target_link_libraries(test_device dl pthread)
#else ()
#  target_link_libraries(query_devices dl pthread  EGL ${OPENGL_LIBRARIES})
#  target_link_libraries(test_device dl pthread  EGL ${OPENGL_LIBRARIES})
#endif()
