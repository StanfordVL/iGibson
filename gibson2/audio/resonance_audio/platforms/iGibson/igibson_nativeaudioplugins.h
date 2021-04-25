/*
Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_NATIVEAUDIOPLUGINS_H_
#define RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_NATIVEAUDIOPLUGINS_H_

// Unity's code is broken for 64-bit Android platforms; they don't define the
// necessary types. So we'll compensate until it's fixed. Once it's fixed, this
// code should break the build in AudioPluginInterface.h.
#if defined(__ANDROID__) && defined(__LP64__)
#include "base/integral_types.h"

typedef int32 SInt32;
typedef uint32 UInt32;
typedef int64 SInt64;
typedef uint64 UInt64;
#endif

#include "NativeCode/AudioPluginInterface.h"
#include "platforms/iGibson/igibson.h"


namespace vraudio {
namespace igibson {

// Callback function for when the renderer plugin instance is created.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
RendererCreateCallback(UnityAudioEffectState* state);

// Callback function for when the renderer plugin instance is destroyed.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
RendererReleaseCallback(UnityAudioEffectState* state);

// Callback function for audio process of the renderer plugin.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK RendererProcessCallback(
    UnityAudioEffectState* state, float* inbuffer, float* outbuffer,
    unsigned int length, int inchannels, int outchannels);

// Callback function for when the spatializer plugin instance is created.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerCreateCallback(UnityAudioEffectState* state);

// Callback function for when the spatializer plugin instance is destroyed.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerReleaseCallback(UnityAudioEffectState* state);

// Callback function for distance attenuation calculation of the spatializer
// plugin.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerDistanceAttenuationCallback(UnityAudioEffectState* state,
                                       float distance_in, float attenuation_in,
                                       float* attenuation_out);

// Callback function for audio process of the spatializer plugin.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK SpatializerProcessCallback(
    UnityAudioEffectState* state, float* inbuffer, float* outbuffer,
    unsigned int length, int inchannels, int outchannels);
// Callback function for when a spatializer plugin parameter value is to be set.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerSetFloatParameterCallback(UnityAudioEffectState* state, int index,
                                     float value);

// Callback function for when a spatializer plugin parameter value is to be get.
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerGetFloatParameterCallback(UnityAudioEffectState* state, int index,
                                     float* value, char* valuestr);

// NOTE: all exported symbols must be added to "unity_android.lds".
extern "C" {

// Defines & registers the spatializer plugin to Unity.
int EXPORT_API
UnityGetAudioEffectDefinitions(UnityAudioEffectDefinition*** definitionptr);

}  // extern C


}
}  // namespace igibson

#endif  // RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_NATIVEAUDIOPLUGINS_H_
