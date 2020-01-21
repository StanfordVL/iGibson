cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)
set (USE_GLAD TRUE)

include_directories(glad)

if (NOT USE_GLAD)
  find_package(OpenGL)
else()
  add_definitions(-DUSE_GLAD) 
endif()

add_subdirectory(pybind11)

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

cuda_add_library(MeshRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/Mesh_renderer.cpp)
add_library(tinyobjloader MODULE cpp/tinyobjloader/tiny_obj_loader.cc cpp/tinyobjloader/bindings.cc)

if (USE_GLAD)
  target_link_libraries(MeshRendererContext PRIVATE pybind11::module dl pthread)
else ()
  target_link_libraries(MeshRendererContext PRIVATE pybind11::module dl pthread  EGL ${OPENGL_LIBRARIES})
endif()

target_link_libraries(tinyobjloader PRIVATE pybind11::module)

set_target_properties(MeshRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

set_target_properties(tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)

if (USE_GLAD)
  target_link_libraries(query_devices dl pthread)
  target_link_libraries(test_device dl pthread)
else ()
  target_link_libraries(query_devices dl pthread  EGL ${OPENGL_LIBRARIES})
  target_link_libraries(test_device dl pthread  EGL ${OPENGL_LIBRARIES})
endif()
