# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#project(audiopluginresonanceaudio)


if (WIN32)
    set(STATIC_LIB_EXT ".lib")
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(THIRD_PARTY_INSTALL_DIR "build64")
        set(EMBREE_BUILD_DIR "build64/Release")
    else ()
        set(THIRD_PARTY_INSTALL_DIR "install32")
        set(EMBREE_BUILD_DIR "build32/Release")
    endif ()
else ()
    set(STATIC_LIB_EXT ".a")
    set(THIRD_PARTY_INSTALL_DIR "install")
    set(EMBREE_BUILD_DIR "build")
endif ()

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

set(LIBOGG_DIR "${PROJECT_SOURCE_DIR}/third_party/ogg/${THIRD_PARTY_INSTALL_DIR}" CACHE PATH "Path to libOgg installation")
set(LIBOGG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/ogg/include")
set(LIBVORBIS_DIR "${PROJECT_SOURCE_DIR}/third_party/vorbis/${THIRD_PARTY_INSTALL_DIR}" CACHE PATH "Path to libVorbis installation")
set(LIBVORBIS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/vorbis/include")
set(LIBEMBREE_DIR "${PROJECT_SOURCE_DIR}/third_party/embree" CACHE PATH "Path to libEmbree installation")
set(LIBEMBREE_INCLUDE_DIR "${LIBEMBREE_DIR}/include")

if (WIN32)
    set(OGG_VORBIS_LIBRARY_IMPORTS
            "${LIBOGG_DIR}/Release/ogg.lib"
            "${LIBVORBIS_DIR}/lib/Release/vorbis.lib"
            #"${LIBVORBIS_DIR}/vorbisenc.lib"
            "${LIBVORBIS_DIR}/lib/Release/vorbisfile.lib")
else ()
    set(OGG_VORBIS_LIBRARY_IMPORTS
            "${LIBOGG_DIR}/lib/libogg.a"
            "${LIBVORBIS_DIR}/lib/libvorbis.a"
            "${LIBVORBIS_DIR}/lib/libvorbisenc.a"
            "${LIBVORBIS_DIR}/lib/libvorbisfile.a")
endif ()

add_subdirectory("${GIBSON2_DIR}/render/pybind11" "${GIBSON2_DIR}/render/pybind11")

# Create STATIC IMPORTED library targets for static 3P dependencies.

# Embree's CMake config doesn't export all static libraries in its install target.
# (see https://github.com/embree/embree/issues/130)
# To statically link embree into the Unity plugin, all static libraries from
# ../third_party/embree/build are manually added to the "${TARGET}" target.
set(EMBREE_LIB_DIR "${LIBEMBREE_DIR}/${EMBREE_BUILD_DIR}")

if (NOT (IOS_DETECTED OR ANDROID))
    file(GLOB EXTERN_LIBS ABSOLUTE ${OGG_VORBIS_LIBRARY_IMPORTS} "${EMBREE_LIB_DIR}/*${STATIC_LIB_EXT}")
endif ()

foreach (EXTERN_LIB ${EXTERN_LIBS})
    #message( "Adding external library at ${EXTERN_LIB}")
    get_filename_component(EXTERN_LIB_FILENAME ${EXTERN_LIB} NAME)
    string(REPLACE "." "_" EXTERN_LIB_TARGETNAME ${EXTERN_LIB_FILENAME})
    string(TOUPPER ${EXTERN_LIB_TARGETNAME} EXTERN_LIB_TARGETNAME)
    add_library(${EXTERN_LIB_TARGETNAME} STATIC IMPORTED)
    set_target_properties(${EXTERN_LIB_TARGETNAME} PROPERTIES
            IMPORTED_LOCATION "${EXTERN_LIB}"
            )
endforeach ()

function(TargetLinkExternal3PStaticLibs TARGET)
    set(WHOLE_ARCHIVE_FLAGS "")
    foreach (EXTERN_LIB ${EXTERN_LIBS})
        get_filename_component(EXTERN_LIB_FILENAME ${EXTERN_LIB} NAME)
        string(REPLACE "." "_" EXTERN_LIB_TARGETNAME ${EXTERN_LIB_FILENAME})
        string(TOUPPER ${EXTERN_LIB_TARGETNAME} EXTERN_LIB_TARGETNAME)
        message(STATUS "Statically linking to: " ${EXTERN_LIB})
        if (MSVC)
            target_link_libraries("${TARGET}" PUBLIC ${EXTERN_LIB_TARGETNAME})
            set(WHOLE_ARCHIVE_FLAGS "${WHOLE_ARCHIVE_FLAGS} /WHOLEARCHIVE:\"${EXTERN_LIB_FILENAME}\"")
        else ()
            SET(WHOLE_ARCHIVE_BRACKET -Wl,--whole-archive ${EXTERN_LIB_TARGETNAME} -Wl,--no-whole-archive)
            target_link_libraries("${TARGET}" PUBLIC ${WHOLE_ARCHIVE_BRACKET})
        endif ()
    endforeach ()

    if (APPLE)
        set_target_properties("${TARGET}" PROPERTIES
                LINK_FLAGS "-Wl,-all_load"
                )
        if (NOT IOS_DETECTED)
            set_target_properties("${TARGET}" PROPERTIES
                    PREFIX "" SUFFIX ".bundle"
                    )
        endif (NOT IOS_DETECTED)
    elseif (MSVC)
        set_target_properties("${TARGET}" PROPERTIES
                LINK_FLAGS "/MANIFEST:NO ${WHOLE_ARCHIVE_FLAGS}"
                )
    endif ()
endfunction()

set(RA_GEO_ACOUSTICS_SOURCES
        ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_listener.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_ray.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_ray.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_source.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/collection_kernel.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/collection_kernel.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/estimating_rt60.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/estimating_rt60.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/impulse_response_computer.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/impulse_response_computer.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/mesh.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/parallel_for.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/parallel_for.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/path.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/path_tracer.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/path_tracer.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/proxy_room_estimator.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/proxy_room_estimator.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/reflection_kernel.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/reflection_kernel.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/sampling.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/scene_manager.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/scene_manager.h
        ${RA_SOURCE_DIR}/geometrical_acoustics/sphere.cc
        ${RA_SOURCE_DIR}/geometrical_acoustics/sphere.h
        ${RA_SOURCE_DIR}/utils/task_thread_pool.cc
        ${RA_SOURCE_DIR}/utils/task_thread_pool.h
        )

add_library(GeometricalAcousticsObj OBJECT ${RA_GEO_ACOUSTICS_SOURCES})
target_include_directories(GeometricalAcousticsObj PRIVATE ${RA_SOURCE_DIR})
target_include_directories(GeometricalAcousticsObj PRIVATE ${LIBEMBREE_INCLUDE_DIR})
target_compile_options(GeometricalAcousticsObj PRIVATE "-DEMBREE_STATIC_LIB")

set(RA_OGG_RECORDER_SOURCES
        ${RA_SOURCE_DIR}/utils/ogg_vorbis_recorder.cc
        ${RA_SOURCE_DIR}/utils/ogg_vorbis_recorder.h
        ${RA_SOURCE_DIR}/utils/vorbis_stream_encoder.cc
        ${RA_SOURCE_DIR}/utils/vorbis_stream_encoder.h
        )
#add_library(OggRecorderObj OBJECT ${RA_OGG_RECORDER_SOURCES})
#target_include_directories(OggRecorderObj PRIVATE ${RA_SOURCE_DIR})
#target_include_directories(OggRecorderObj PRIVATE ${LIBVORBIS_INCLUDE_DIR})
#target_include_directories(OggRecorderObj PRIVATE ${LIBOGG_INCLUDE_DIR})

set(UNITY_PLUGIN_SDK_DIR "${PROJECT_SOURCE_DIR}/third_party/nativeaudioplugins" CACHE PATH "Path to Unity Native Audio Plugins SDK")

set(RA_GIB_SOURCE_DIR "${PROJECT_SOURCE_DIR}/platforms/iGibson")
if (IOS_DETECTED OR ANDROID)
    set(RA_GIB_SOURCES
            $<TARGET_OBJECTS:ResonanceAudioObj>
            $<TARGET_OBJECTS:SadieHrtfsObj>
            $<TARGET_OBJECTS:PffftObj>
            ${RA_GIB_SOURCE_DIR}/bindings.h
            ${RA_GIB_SOURCE_DIR}/bindings.cc
            ${RA_GIB_SOURCE_DIR}/igibson_nativeaudioplugins.h
            ${RA_GIB_SOURCE_DIR}/igibson_nativeaudioplugins.cc
            ${RA_GIB_SOURCE_DIR}/igibson.h
            ${RA_GIB_SOURCE_DIR}/igibson.cc
            )
else ()
    set(RA_GIB_SOURCES
            $<TARGET_OBJECTS:ResonanceAudioObj>
            $<TARGET_OBJECTS:SadieHrtfsObj>
            $<TARGET_OBJECTS:PffftObj>
            $<TARGET_OBJECTS:GeometricalAcousticsObj>
            ${RA_GIB_SOURCE_DIR}/igibson_nativeaudioplugins.h
            ${RA_GIB_SOURCE_DIR}/igibson_nativeaudioplugins.cc
            ${RA_GIB_SOURCE_DIR}/igibson_reverb_computer.h
            ${RA_GIB_SOURCE_DIR}/igibson_reverb_computer.cc
            ${RA_GIB_SOURCE_DIR}/igibson.h
            ${RA_GIB_SOURCE_DIR}/igibson.cc
            ${RA_GIB_SOURCE_DIR}/bindings.h
            ${RA_GIB_SOURCE_DIR}/bindings.cc
            )
endif ()

#add_library(audiopluginresonanceaudio SHARED ${RA_GIB_SOURCES})
add_library(audio MODULE ${RA_GIB_SOURCES})

set(PFFFT_INCLUDE_DIR ${PFFFT_DIR})
target_include_directories(audio PRIVATE ${UNITY_PLUGIN_SDK_DIR})
target_include_directories(audio PRIVATE ${RA_SOURCE_DIR})
target_include_directories(audio PRIVATE ${LIBEMBREE_INCLUDE_DIR})
target_include_directories(audio PRIVATE ${EIGEN3_INCLUDE_DIR}/)
target_include_directories(audio PRIVATE ${PFFFT_INCLUDE_DIR}/)
target_include_directories(audio PRIVATE ${LIBVORBIS_INCLUDE_DIR})
target_include_directories(audio PRIVATE ${LIBOGG_INCLUDE_DIR})
#if (MSVC)
#    target_sources(audiopluginresonanceaudio PUBLIC ${RA_GIB_SOURCE_DIR}/igibson_win.def)
#endif (MSVC)

#add_library(audiobindings MODULE  ${RA_GIB_SOURCE_DIR}/bindings.h ${RA_GIB_SOURCE_DIR}/bindings.cc)

TargetLinkExternal3PStaticLibs(audio)
if (MSVC)
    # Embree pulls in WinSock32
    target_link_libraries(audio PUBLIC wsock32 ws2_32 PRIVATE pybind11::module)
else()
    target_link_libraries(audio PRIVATE pybind11::module)
endif ()

set_target_properties(audio PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

set(IGIB_INSTALL_DIR "${INSTALL_DIR}/iGibson/")
set(INSTALL_BIN_DIR "")
if (MSVC)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(INSTALL_BIN_DIR "x86_64")
    else ()
        set(INSTALL_BIN_DIR "x86")
    endif ()
elseif (ANDROID)
    if ("${ANDROID_ABI}" MATCHES "armeabi-v7a")
        set(INSTALL_BIN_DIR "Android/libs/armeabi-v7a")
    elseif ("${ANDROID_ABI}" MATCHES "x86")
        set(INSTALL_BIN_DIR "Android/libs/x86")
    elseif ("${ANDROID_ABI}" MATCHES "arm64-v8a")
        set(INSTALL_BIN_DIR "Android/libs/arm64-v8a")
    elseif ("${ANDROID_ABI}" MATCHES "x86_64")
        set(INSTALL_BIN_DIR "Android/libs/x86_64")
    endif ()
elseif (APPLE)
    if (IOS_DETECTED)
        set(INSTALL_BIN_DIR "iOS")
    else ()
        set(INSTALL_BIN_DIR "x86_64")
    endif ()
elseif (UNIX)
    set(INSTALL_BIN_DIR "x86_64")
endif ()


install(TARGETS audio
        LIBRARY DESTINATION "${AUDIO_DIR}"
        ARCHIVE DESTINATION "${AUDIO_DIR}"
        RUNTIME DESTINATION "${AUDIO_DIR}")
#install(DIRECTORY ${RA_UNITY_SOURCE_DIR}/UnityIntegration/Assets DESTINATION "${INSTALL_DIR}/unity")
#install(DIRECTORY ${RA_UNITY_SOURCE_DIR}/UnityIntegration/ProjectSettings DESTINATION "${INSTALL_DIR}/unity")

if (BUILD_GEOMETRICAL_ACOUSTICS_TESTS)
    set(RA_GEO_ACOUSTICS_TESTS
            ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_ray_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/acoustic_source_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/collection_kernel_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/estimating_rt60_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/impulse_response_computer_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/path_tracer_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/proxy_room_estimator_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/reflection_kernel_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/scene_manager_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/sphere_test.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/test_util.cc
            ${RA_SOURCE_DIR}/geometrical_acoustics/test_util.h
            )

    # Unit test target
    add_executable(RAGeometricAcousticsUnitTests ${RA_GEO_ACOUSTICS_TESTS}
            $<TARGET_OBJECTS:gtest>
            $<TARGET_OBJECTS:ResonanceAudioObj>
            $<TARGET_OBJECTS:SadieHrtfsObj>
            $<TARGET_OBJECTS:PffftObj>
            $<TARGET_OBJECTS:GeometricalAcousticsObj>)
    # Link test executable against gtest & gtest_main
    TargetLinkExternal3PStaticLibs(RAGeometricAcousticsUnitTests)

    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/resonance_audio/)
    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE ${LIBEMBREE_INCLUDE_DIR})
    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE "${GTEST_DIR}/googlemock/")
    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE "${GTEST_DIR}/googlemock/include")
    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE "${GTEST_DIR}/googletest/")
    target_include_directories(RAGeometricAcousticsUnitTests PRIVATE "${GTEST_DIR}/googletest/include")

    if (NOT WIN32)
        find_package(Threads REQUIRED)
        target_link_libraries(RAGeometricAcousticsUnitTests pthread)
    endif (NOT WIN32)
    if (MSVC)
        # Embree pulls in WinSock32
        target_link_libraries(RAGeometricAcousticsUnitTests wsock32 ws2_32)
    elseif (UNIX)
        target_link_libraries(RAGeometricAcousticsUnitTests ${CMAKE_DL_LIBS})
    endif ()

    add_test(NAME runRAGeometricAcousticsUnitTest COMMAND $<TARGET_FILE:RAGeometricAcousticsUnitTests>)
    add_custom_command(
            TARGET RAGeometricAcousticsUnitTests
            COMMENT "Run geometric acoustics tests"
            POST_BUILD
            COMMAND RAGeometricAcousticsUnitTests
    )
endif (BUILD_GEOMETRICAL_ACOUSTICS_TESTS)
