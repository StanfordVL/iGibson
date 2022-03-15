cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)

set(USE_GLAD TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(OpenGL_GL_PREFERENCE GLVND)

if(WIN32)
  set(USE_GLFW FALSE)
  # iGibson on Windows is always in VR mode
  set(USE_VR TRUE)
  set(USE_EGL FALSE)
elseif(UNIX)
  if(MAC_PLATFORM)
    set(USE_GLFW TRUE)
    set(USE_VR FALSE)
    set(USE_EGL FALSE)
  else()
    set(USE_GLFW FALSE)
    set(USE_EGL TRUE)
  endif()
else()
  message( FATAL_ERROR "Unsupported platform: iGibson only supports macOS, linux, and windows" )
endif()

# If a user is using a new enough cmake version (Ubuntu 18.04 or greater)
# Automatically detect and enable/disable CUDA, otherwise assume true
if (UNIX AND (CMAKE_VERSION VERSION_GREATER 3.17 OR CMAKE_VERSION VERSION_EQUAL 3.17))
  find_package(CUDAToolkit)
  if (NOT CUDAToolkit_FOUND)
    message(WARNING [=[
      nvcc not detected in path. iGibson will be built without CUDA support used
      for rendering to tensors. If desired, please install cudatoolkit with your
      package manager or cudatoolkit-dev with conda.
    ]=])
    set(USE_CUDA FALSE)
  else ()
    set(USE_CUDA TRUE)
  endif ()
elseif(UNIX AND NOT MAC_PLATFORM)
  # Assume CUDA true otherwise
  set(USE_CUDA TRUE)
else()
  # Assume CUDA false otherwise (macOS, Windows)
  set(USE_CUDA FALSE)
endif()

include_directories(glad)

if(NOT USE_GLAD)
  find_package(OpenGL)
else()
  add_definitions(-DUSE_GLAD)
endif()

if(USE_VR)
  find_package(OpenGL)
  # OpenGL is needed for vr-gl interoperation
endif()

if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
  find_package(OpenGL)
  # OpenGL is still needed for cuda-gl interoperation
endif()

add_subdirectory(pybind11)
add_subdirectory(cryptopp)
include_directories(cryptopp)

# Add GLM include directory
include_directories("glm")

if(USE_GLFW OR USE_VR)
  set(GLFW_DIR glfw)
  set(GLFW_BUILD_EXAMPLES
      OFF
      CACHE INTERNAL "Build the GLFW example programs")
  set(GLFW_BUILD_TESTS
      OFF
      CACHE INTERNAL "Build the GLFW test programs")
  set(GLFW_BUILD_DOCS
      OFF
      CACHE INTERNAL "Build the GLFW documentation")
  set(GLFW_INSTALL
      OFF
      CACHE INTERNAL "Generate installation target")
  add_subdirectory("${GLFW_DIR}")
  include_directories("${GLFW_DIR}/include")
endif()

if(USE_VR)
  # Find OpenVR
  include(FetchContent)
  FetchContent_Declare(openvr
    GIT_REPOSITORY https://github.com/ValveSoftware/openvr.git
    GIT_TAG v1.14.15
  )
  FetchContent_MakeAvailable(openvr)

  set(OPENVR_DIR "${FETCHCONTENT_BASE_DIR}/openvr-src")
  set(OPENVR_INCLUDE_DIR "${OPENVR_DIR}/headers")
  include_directories("${OPENVR_INCLUDE_DIR}")

  if(WIN32)
    set(PATH_SUFFIXES win64 Win64 x64)
    # Find SRAnipal
    set(SRANI_DIR sranipal)
    find_library(
      SRANI_LIBRARIES
      NAMES SRAnipal
      PATHS "${SRANI_DIR}/lib"
      NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    include_directories("${SRANI_DIR}/include")
  else()
    set(PATH_SUFFIXES linux64 Linux64 x64)
  endif()

  find_library(
    OPENVR_LIBRARIES
    NAMES openvr_api
    PATHS "${OPENVR_DIR}/bin" "${OPENVR_DIR}/lib"
    PATH_SUFFIXES ${PATH_SUFFIXES}
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

  add_library(
    VRRendererContext MODULE glad/gl.cpp cpp/vr_mesh_renderer.cpp
                             cpp/glfw_mesh_renderer.cpp cpp/mesh_renderer.cpp)
  if(WIN32)
    target_link_libraries(
      VRRendererContext
      PRIVATE pybind11::module
              ${CMAKE_DL_LIBS}
              glfw
              ${GLFW_LIBRARIES}
              ${OPENGL_LIBRARIES}
              ${OPENVR_LIBRARIES}
              ${SRANI_LIBRARIES}
              cryptopp-static)
  else()
    target_link_libraries(
      VRRendererContext
      PRIVATE pybind11::module
              ${CMAKE_DL_LIBS}
              glfw
              ${GLFW_LIBRARIES}
              ${OPENGL_LIBRARIES}
              ${OPENVR_LIBRARIES}
              ${CUDA_LIBRARIES}
              cryptopp-static)
  endif()
  set_target_properties(
    VRRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                 SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

if(USE_EGL)
  if(USE_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
    cuda_add_library(EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp
                     cpp/mesh_renderer.cpp cpp/egl_mesh_renderer.cpp)
  else()
    add_library(
      EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/mesh_renderer.cpp
                                cpp/egl_mesh_renderer.cpp)
  endif()

  if(USE_GLAD)
    target_link_libraries(EGLRendererContext PRIVATE pybind11::module dl
                                                     pthread cryptopp-static)
  else()
    target_link_libraries(
      EGLRendererContext PRIVATE pybind11::module dl pthread EGL
                                 ${OPENGL_LIBRARIES} cryptopp-static)
  endif()

  set_target_properties(
    EGLRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                  SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

if(USE_GLFW)
  add_library(GLFWRendererContext MODULE glad/gl.cpp cpp/glfw_mesh_renderer.cpp
                                         cpp/mesh_renderer.cpp)
  if(WIN32)
    if(USE_GLAD)
      target_link_libraries(
        GLFWRendererContext PRIVATE pybind11::module glfw ${GLFW_LIBRARIES}
                                    cryptopp-static)
    else()
      target_link_libraries(
        GLFWRendererContext PRIVATE pybind11::module glfw ${GLFW_LIBRARIES}
                                    ${OPENGL_LIBRARIES} cryptopp-static)
    endif()
  else()
    if(USE_GLAD)
      target_link_libraries(
        GLFWRendererContext PRIVATE pybind11::module dl glfw ${GLFW_LIBRARIES}
                                    cryptopp-static)
    else()
      target_link_libraries(
        GLFWRendererContext PRIVATE pybind11::module dl glfw ${GLFW_LIBRARIES}
                                    ${OPENGL_LIBRARIES} cryptopp-static)
    endif()
  endif()
  set_target_properties(
    GLFWRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                   SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

add_library(
  tinyobjloader MODULE
  cpp/tinyobjloader/tiny_obj_loader.cc
  cpp/tinyobjloader/tiny_obj_loader_decrypt.cc cpp/tinyobjloader/bindings.cc)
target_link_libraries(tinyobjloader PRIVATE pybind11::module cryptopp-static)
set_target_properties(
  tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                           SUFFIX "${PYTHON_MODULE_EXTENSION}")

if(UNIX)
  add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
  add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)
  if(USE_GLAD)
    target_link_libraries(query_devices dl pthread)
    target_link_libraries(test_device dl pthread)
  else()
    target_link_libraries(query_devices dl pthread EGL ${OPENGL_LIBRARIES})
    target_link_libraries(test_device dl pthread EGL ${OPENGL_LIBRARIES})
  endif()
endif()
