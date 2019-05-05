cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)

include_directories(glad)
find_package(OpenGL)
add_subdirectory(pybind11)

add_library(CppMeshRenderer MODULE glad/egl.cpp glad/gl.cpp cpp/Mesh_renderer.cpp)

target_link_libraries(CppMeshRenderer PRIVATE pybind11::module dl pthread  EGL ${OPENGL_LIBRARIES})
set_target_properties(CppMeshRenderer PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)

target_link_libraries(query_devices dl pthread  EGL ${OPENGL_LIBRARIES})
target_link_libraries(test_device dl pthread  EGL ${OPENGL_LIBRARIES})