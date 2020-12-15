cmake_minimum_required(VERSION 2.8.12)
project(CppMeshRenderer)

set(USE_GLAD TRUE)

if (MAC_PLATFORM)
    set(USE_CUDA FALSE)
    set(USE_GLFW TRUE)
    set(USE_VR FALSE)
elseif (WIN32)
    set(USE_CUDA FALSE)
    set(USE_GLFW TRUE)
    # iGibson on Windows is always in VR mode
    set(USE_VR TRUE)
else ()
    set(USE_CUDA TRUE)
    set(USE_GLFW FALSE)
    set(USE_VR FALSE)
endif ()

include_directories(glad)

if (NOT USE_GLAD)
    find_package(OpenGL)
else ()
    add_definitions(-DUSE_GLAD)
endif ()

if (USE_VR)
    find_package(OpenGL)
    # OpenGL is needed for vr-gl interoperation
endif()

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(OpenGL)
    #OpenGL is still needed for cuda-gl interoperation
endif ()

add_subdirectory(pybind11)
# Add GLM include directory
include_directories("glm")

if (USE_GLFW)
    set(GLFW_DIR glfw)
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
    set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
    add_subdirectory("${GLFW_DIR}")
    include_directories("${GLFW_DIR}/include")
endif ()

if (USE_VR)
    # Find OpenVR
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

    # Find SRAnipal
    set(SRANI_DIR sranipal)
    find_library(SRANI_LIBRARIES
	NAMES
	    SRAnipal
	PATHS
	    "${SRANI_DIR}/lib"
	NO_DEFAULT_PATH
	NO_CMAKE_FIND_ROOT_PATH
    )
    include_directories("${SRANI_DIR}/include")
endif()

add_library(tinyobjloader MODULE cpp/tinyobjloader/tiny_obj_loader.cc cpp/tinyobjloader/bindings.cc)
if (USE_VR)
    add_library(VRRendererContext MODULE glad/gl.cpp cpp/vr_mesh_renderer.cpp cpp/glfw_mesh_renderer.cpp cpp/mesh_renderer.cpp)
else()
    if (USE_CUDA)
        find_package(CUDA REQUIRED)
        set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
        cuda_add_library(EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/mesh_renderer.cpp cpp/egl_mesh_renderer.cpp)
    else ()
        add_library(EGLRendererContext MODULE glad/egl.cpp glad/gl.cpp cpp/mesh_renderer.cpp cpp/egl_mesh_renderer.cpp)
    endif ()
    if (USE_GLFW)
        add_library(GLFWRendererContext MODULE glad/gl.cpp cpp/glfw_mesh_renderer.cpp cpp/mesh_renderer.cpp)
    endif ()
endif()

if (USE_VR)
    target_link_libraries(VRRendererContext PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OPENVR_LIBRARIES} ${SRANI_LIBRARIES})
else ()
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
    endif()
endif ()

target_link_libraries(tinyobjloader PRIVATE pybind11::module)

if (USE_VR) 
    set_target_properties(VRRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
else()
    set_target_properties(EGLRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
    if (USE_GLFW)
        set_target_properties(GLFWRendererContext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}")
    endif ()
endif()

set_target_properties(tinyobjloader PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

if (NOT USE_VR)
    add_executable(query_devices glad/egl.cpp glad/gl.cpp cpp/query_devices.cpp)
    add_executable(test_device glad/egl.cpp glad/gl.cpp cpp/test_device.cpp)
    if (USE_GLAD)
        target_link_libraries(query_devices dl pthread)
        target_link_libraries(test_device dl pthread)
    else ()
        target_link_libraries(query_devices dl pthread EGL ${OPENGL_LIBRARIES})
        target_link_libraries(test_device dl pthread EGL ${OPENGL_LIBRARIES})
    endif ()
endif()
