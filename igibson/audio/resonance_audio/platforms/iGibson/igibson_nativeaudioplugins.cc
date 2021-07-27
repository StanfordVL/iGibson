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

#include "platforms/iGibson/igibson_nativeaudioplugins.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

#include "api/resonance_audio_api.h"
#include "base/constants_and_types.h"
#include "base/logging.h"
#include "base/misc_math.h"
#include "platforms/common/utils.h"


namespace vraudio {
namespace igibson {

namespace {


// External effect data of the spatializer.
struct EffectData {
  // Defines the parameters to be exposed to Unity.
  enum {
    // Source id.
    kId = 0,
    // Minimum distance for distance-based attenuation.
    kDistanceAttenuation = 1,
    // Room effects gain.
    kRoomEffectsGain = 2,
    // Linear gain.
    kGain = 3,
    // Source directivity alpha.
    kDirectivityAlpha = 4,
    // Source directivity sharpness.
    kDirectivitySharpness = 5,
    // Listener directivity alpha.
    kListenerDirectivityAlpha = 6,
    // Listener directivity sharpness.
    kListenerDirectivitySharpness = 7,
    // Occlusion intensity.
    kOcclusion = 8,
    // Rendering quality.
    kQuality = 9,
    // Near field effect gain.
    kNearFieldEffectGain = 10,
    // Source volume.
    kVolume = 11,
    // Number of parameters to be exposed.
    kNumParams = 12
  };

  std::array<float, kNumParams> p;
};


// There are two native audio plugins - the renderer and the spatializer.
static const int kNumAudioPlugins = 2;

// Unity audio effect definitions for the plugins.
static UnityAudioEffectDefinition* definitions[kNumAudioPlugins];
static UnityAudioEffectDefinition definition[kNumAudioPlugins];
static UnityAudioParameterDefinition parameters[EffectData::kNumParams];

// This is set to true after the first initialization in order to avoid
// redundant computation on |UnityGetAudioEffectDefinitions| calls from Unity.
static bool plugins_initialized = false;

// This is set to true if the host Unity native audio plugin interface supports
// the additional properties needed in native plugin data, namely mute, volume
// and minimum distance properties of an audio source, that were introduced in
// UNITY_AUDIO_PLUGIN_API_VERSION 0x010401.
static const UInt32 kNativePluginDataMinApiVersion = 0x010401;
static bool use_native_plugin_data = false;

// Listener transformation matrix in Unity space.
static const int kTransformMatrixSize = 16;
static float unity_listener_transform[kTransformMatrixSize] = {
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

// Native audio plugin names.
const char* kRendererName = "Resonance Audio Renderer";
const char* kSpatializerName = "Resonance Audio";

// Helper function to populate and return a |UnityAudioParameterDefinition|
// struct for the paramaters of a native audio plugin.
UnityAudioParameterDefinition GetUnityAudioParameterDefinition(
    const std::string& name, float min_value, float max_value,
    float default_value) {
  UnityAudioParameterDefinition parameter;
  memset(&parameter, 0, sizeof(parameter));

  std::copy(name.begin(), name.end(), parameter.name);
  parameter.min = min_value;
  parameter.max = max_value;
  parameter.defaultval = default_value;
  parameter.description = parameter.name;

  return parameter;
}

// Helper function to populate and return a |UnityAudioEffectDefinition| struct
// for a native audio plugin.
UnityAudioEffectDefinition GetUnityAudioEffectDefinition(
    const std::string& name, UnityAudioEffect_CreateCallback createcallback,
    UnityAudioEffect_ReleaseCallback releasecallback,
    UnityAudioEffect_ProcessCallback processcallback,
    UnityAudioEffect_SetFloatParameterCallback setfloatparametercallback,
    UnityAudioEffect_GetFloatParameterCallback getfloatparametercallback,
    size_t num_parameters, UnityAudioParameterDefinition* parameters,
    bool is_spatializer) {
  UnityAudioEffectDefinition definition;
  memset(&definition, 0, sizeof(definition));

  std::copy(name.begin(), name.end(), definition.name);
  definition.structsize = sizeof(UnityAudioEffectDefinition);
  definition.paramstructsize = sizeof(UnityAudioParameterDefinition);
  definition.apiversion = UNITY_AUDIO_PLUGIN_API_VERSION;
  definition.pluginversion = 0x010000;
  definition.create = createcallback;
  definition.release = releasecallback;
  definition.process = processcallback;
  definition.setfloatparameter = setfloatparametercallback;
  definition.getfloatparameter = getfloatparametercallback;
  definition.numparameters = static_cast<UInt32>(num_parameters);
  definition.paramdefs = parameters;
  if (is_spatializer) {
    definition.flags |=
        UnityAudioEffectDefinitionFlags_IsSpatializer |
        UnityAudioEffectDefinitionFlags_IsAmbisonicDecoder |
        UnityAudioEffectDefinitionFlags_AppliesDistanceAttenuation;
  }

  return definition;
}

// Returns the corresponding |RenderingMode| equivalent of the given
// |rendering_mode| value.
RenderingMode GetRenderingModeEnum(int rendering_mode) {
  RenderingMode rendering_mode_enum;
  switch (rendering_mode) {
    case 0:  // Stereo panning.
      rendering_mode_enum = RenderingMode::kStereoPanning;
      break;
    case 1:  // Binaural low quality.
      rendering_mode_enum = RenderingMode::kBinauralLowQuality;
      break;
    case 2:  // Binaural high quality.
      rendering_mode_enum = RenderingMode::kBinauralHighQuality;
      break;
    default:
      LOG(WARNING) << "Invalid rendering quality mode specified: "
                   << rendering_mode << ", using binaural high quality";
      rendering_mode_enum = RenderingMode::kBinauralHighQuality;
      break;
  }
  return rendering_mode_enum;
}

}  // namespace

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
RendererCreateCallback(UnityAudioEffectState* state) {
  CHECK(state);
  const int sample_rate = static_cast<int>(state->samplerate);
  const size_t frames_per_buffer = static_cast<size_t>(state->dspbuffersize);
  Initialize(sample_rate, kNumStereoChannels, frames_per_buffer);
  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
RendererReleaseCallback(UnityAudioEffectState* state) {
  Shutdown();
  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK RendererProcessCallback(
    UnityAudioEffectState* state, float* inbuffer, float* outbuffer,
    unsigned int length, int inchannels, int outchannels) {
  CHECK(state);
  CHECK(inbuffer);
  CHECK(outbuffer);
  CHECK_GE(inchannels, 0);
  CHECK_GE(outchannels, 0);

  if (inchannels != kNumStereoChannels || outchannels != kNumStereoChannels) {
    // Skip processing if the audio channels are not stereo.
    const size_t buffer_size_per_channel_bytes = length * sizeof(float);
    CHECK(!DoesIntegerMultiplicationOverflow<size_t>(
        length, sizeof(float), buffer_size_per_channel_bytes));
    const size_t buffer_size_bytes =
        buffer_size_per_channel_bytes * outchannels;
    CHECK(!DoesIntegerMultiplicationOverflow<size_t>(
        buffer_size_per_channel_bytes, outchannels, buffer_size_bytes));

    memcpy(outbuffer, inbuffer, buffer_size_bytes);

    return UNITY_AUDIODSP_OK;
  }

  // Update the listener transformation.
  // First, invert the transformation matrix, as Unity provides an inverse
  // camera transformation matrix for the listener. For more information:
  // https://docs.unity3d.com/Manual/AudioSpatializerSDK.html
  Eigen::Matrix4f listener_transform =
      Eigen::Matrix4f(unity_listener_transform).inverse();
  // Switch to right-handed coordinate system (Unity is left-handed).
  FlipZAxis(&listener_transform);
  // Set listener transform.
  const Eigen::Vector3f position = GetPosition(listener_transform);
  const Eigen::Quaternionf rotation = GetQuaternion(listener_transform);
  SetListenerTransform(position.x(), position.y(), position.z(), rotation.x(),
                       rotation.y(), rotation.z(), rotation.w());
  // Process the next buffer.
  DCHECK_EQ(inchannels, outchannels);
  ProcessListener(length, outbuffer);

  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerCreateCallback(UnityAudioEffectState* state) {
  CHECK(state);
  // Verify if the native plugin data can be used for additional source
  // properties.
  use_native_plugin_data =
      state->structsize >= sizeof(UnityAudioEffectState) &&
      state->hostapiversion >= kNativePluginDataMinApiVersion;
  // Initialize the effect data.
  EffectData* data = new EffectData();
  data->p[EffectData::kId] =
      static_cast<float>(ResonanceAudioApi::kInvalidSourceId);
  data->p[EffectData::kGain] = 1.0f;
  data->p[EffectData::kDistanceAttenuation] = 1.0f;
  state->effectdata = data;
  // Override distance attenuation callback.
  if (state->spatializerdata != nullptr) {
    data->p[EffectData::kRoomEffectsGain] = 1.0f;
    data->p[EffectData::kDirectivityAlpha] = 0.0f;
    data->p[EffectData::kDirectivitySharpness] = 1.0f;
    data->p[EffectData::kNearFieldEffectGain] = 0.0f;
    data->p[EffectData::kListenerDirectivityAlpha] = 0.0f;
    data->p[EffectData::kListenerDirectivitySharpness] = 1.0f;
    data->p[EffectData::kOcclusion] = 0.0f;
    data->p[EffectData::kQuality] = 2.0f;  // Binaural high quality.
    state->spatializerdata->distanceattenuationcallback =
        &SpatializerDistanceAttenuationCallback;
  }
  if (state->ambisonicdata != nullptr) {
    data->p[EffectData::kRoomEffectsGain] = 0.0f;
    data->p[EffectData::kVolume] = 1.0f;
    state->ambisonicdata->distanceattenuationcallback =
        &SpatializerDistanceAttenuationCallback;
  }
  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerReleaseCallback(UnityAudioEffectState* state) {
  CHECK(state);
  EffectData* data = state->GetEffectData<EffectData>();
  const auto id =
      static_cast<ResonanceAudioApi::SourceId>(data->p[EffectData::kId]);
  if (id != ResonanceAudioApi::kInvalidSourceId) {
    DestroySource(id);
    data->p[EffectData::kId] =
        static_cast<float>(ResonanceAudioApi::kInvalidSourceId);
  }
  delete data;

  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerDistanceAttenuationCallback(UnityAudioEffectState* state,
                                       float distance_in, float attenuation_in,
                                       float* attenuation_out) {
  CHECK(state);
  EffectData* data = state->GetEffectData<EffectData>();
  data->p[EffectData::kDistanceAttenuation] = attenuation_in;

  if (state->ambisonicdata != nullptr) {
    // Pass the incoming attenuation value as-is for the ambisonic decoder
    // plugin.
    *attenuation_out = attenuation_in;
  } else {
    // Pass *no attenuation* back to Unity to bypass the distance attenuation
    // process in Unity side for the spatializer plugin.
    *attenuation_out = 1.0f;
  }

  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK SpatializerProcessCallback(
    UnityAudioEffectState* state, float* inbuffer, float* outbuffer,
    unsigned int length, int inchannels, int outchannels) {
  CHECK(state);
  CHECK(inbuffer);
  CHECK(outbuffer);
  CHECK_GE(inchannels, 0);
  CHECK_GE(outchannels, 0);

  const size_t buffer_size_per_channel_bytes = length * sizeof(float);
  CHECK(!DoesIntegerMultiplicationOverflow<size_t>(
      length, sizeof(float), buffer_size_per_channel_bytes));
  const size_t buffer_size_bytes = buffer_size_per_channel_bytes * outchannels;
  CHECK(!DoesIntegerMultiplicationOverflow<size_t>(
      buffer_size_per_channel_bytes, outchannels, buffer_size_bytes));

  // Get the spatializer data.
  EffectData* data = state->GetEffectData<EffectData>();
  DCHECK(data != nullptr);

  const UnityAudioAmbisonicData* ambisonicdata = state->ambisonicdata;
  const UnityAudioSpatializerData* spatializerdata = state->spatializerdata;
  const bool is_ambisonic_decoder = ambisonicdata != nullptr;

  if (!is_ambisonic_decoder &&
      (inchannels != kNumStereoChannels || outchannels != kNumStereoChannels)) {
    // Skip processing if the spatializer audio channels are not stereo.
    return UNITY_AUDIODSP_OK;
  }

  // Store the current listener transformation.
  const auto& listenermatrix = is_ambisonic_decoder
                                   ? ambisonicdata->listenermatrix
                                   : spatializerdata->listenermatrix;
  std::memcpy(unity_listener_transform, listenermatrix,
              kTransformMatrixSize * sizeof(float));

  DCHECK_EQ(inchannels, outchannels);

  // Update the source properties and process the next buffer.
  auto id = static_cast<ResonanceAudioApi::SourceId>(data->p[EffectData::kId]);
  if (id == ResonanceAudioApi::kInvalidSourceId) {
    // Create a new source.
    if (is_ambisonic_decoder) {
      id = CreateSoundfield(inchannels);
    } else {
      const auto rendering_mode =
          GetRenderingModeEnum(static_cast<int>(data->p[EffectData::kQuality]));
      id = CreateSoundObject(rendering_mode, 0.0f, 0.0f);
    }
    data->p[EffectData::kId] = static_cast<float>(id);
  }
  if (id != ResonanceAudioApi::kInvalidSourceId) {
    // Compute the source transformation.
    const auto& sourcematrix = is_ambisonic_decoder
                                   ? ambisonicdata->sourcematrix
                                   : spatializerdata->sourcematrix;
    Eigen::Matrix4f transform(sourcematrix);
    // Switch to right-handed coordinate system (Unity is left-handed).
    FlipZAxis(&transform);
    const Eigen::Vector3f position = GetPosition(transform);
    const Eigen::Quaternionf rotation = GetQuaternion(transform);
    SetSourceTransform(id, position.x(), position.y(), position.z(),
                       rotation.x(), rotation.y(), rotation.z(), rotation.w());

    // Final source attenuation is calculated manually with respect to the
    // spatial blend of the spatializer.
    const float gain = data->p[EffectData::kGain];
    const float attenuation = data->p[EffectData::kDistanceAttenuation];
    const float spatialblend = is_ambisonic_decoder
                                   ? ambisonicdata->spatialblend
                                   : spatializerdata->spatialblend;
    const float weighted_attenuation =
        1.0f - spatialblend + spatialblend * attenuation;
    SetSourceDistanceAttenuation(id, weighted_attenuation);

    SetSourceRoomEffectsGain(id, data->p[EffectData::kRoomEffectsGain]);

    if (is_ambisonic_decoder) {
      // Apply audio source volume manually to workaround the process order in
      // Unity's ambisonic decoder pipeline.
      const float volume = use_native_plugin_data
                               ? ambisonicdata->volume
                               : data->p[EffectData::kVolume];
      SetSourceGain(id, volume * gain);
    } else {
      // There is an additional |kSqrtTwo| applied to compensate the stereo
      // upmix multiplier for mono sound sources.
      SetSourceGain(id, kSqrtTwo * gain);

      const float directivity_alpha = data->p[EffectData::kDirectivityAlpha];
      const float directivity_order =
          data->p[EffectData::kDirectivitySharpness];
      SetSourceDirectivity(id, directivity_alpha, directivity_order);

      const float listener_directivity_alpha =
          data->p[EffectData::kListenerDirectivityAlpha];
      const float listener_directivity_order =
          data->p[EffectData::kListenerDirectivitySharpness];
      SetSourceListenerDirectivity(id, listener_directivity_alpha,
                                   listener_directivity_order);

      const float near_field_effect_gain =
          data->p[EffectData::kNearFieldEffectGain];
      SetSourceNearFieldEffectGain(id, near_field_effect_gain);

      const float occlusion_intensity = data->p[EffectData::kOcclusion];
      SetSourceOcclusionIntensity(id, occlusion_intensity);

      SetSourceSpread(id, spatializerdata->spread);
    }
    ProcessSource(id, static_cast<size_t>(inchannels), length, inbuffer);

    // Fill the output buffer with the raw input buffer to allow post
    // processing/analysis features in Unity.
    memcpy(outbuffer, inbuffer, buffer_size_bytes);
  } else {
    // Source not initialized, fill the output buffer with zeros.
    memset(outbuffer, 0, buffer_size_bytes);
  }

  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerSetFloatParameterCallback(UnityAudioEffectState* state, int index,
                                     float value) {
  CHECK_GE(index, 0);

  if (index >= EffectData::kNumParams) {
    return UNITY_AUDIODSP_ERR_UNSUPPORTED;
  }

  EffectData* data = state->GetEffectData<EffectData>();
  if (index == static_cast<int>(EffectData::kQuality) &&
      value != data->p[index]) {
    // Destroy the source when the rendering mode is changed, to be re-created
    // in the next process callback.
    const auto id =
        static_cast<ResonanceAudioApi::SourceId>(data->p[EffectData::kId]);
    if (id != ResonanceAudioApi::kInvalidSourceId) {
      DestroySource(id);
      data->p[EffectData::kId] =
          static_cast<float>(ResonanceAudioApi::kInvalidSourceId);
    }
  }
  data->p[index] = value;

  return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK
SpatializerGetFloatParameterCallback(UnityAudioEffectState* state, int index,
                                     float* value, char* valuestr) {
  CHECK_GE(index, 0);

  if (index >= EffectData::kNumParams) {
    return UNITY_AUDIODSP_ERR_UNSUPPORTED;
  }

  EffectData* data = state->GetEffectData<EffectData>();
  if (value != nullptr) {
    *value = data->p[index];
  }

  return UNITY_AUDIODSP_OK;
}

int UnityGetAudioEffectDefinitions(
    UnityAudioEffectDefinition*** definitionptr) {
  CHECK(definitionptr);

  if (!plugins_initialized) {
    plugins_initialized = true;
    // Initialize the renderer plugin.
    definition[0] = GetUnityAudioEffectDefinition(
        kRendererName, RendererCreateCallback, RendererReleaseCallback,
        RendererProcessCallback, nullptr, nullptr, 0, nullptr, false);
    definitions[0] = &definition[0];
    // Initialize the spatializer plugin.
    // Note that the parameter names must be less than 16 characters long.
    parameters[EffectData::kId] = GetUnityAudioParameterDefinition(
        "Id", static_cast<float>(ResonanceAudioApi::kInvalidSourceId),
        std::numeric_limits<float>::max(),
        static_cast<float>(ResonanceAudioApi::kInvalidSourceId));
    parameters[EffectData::kDistanceAttenuation] =
        GetUnityAudioParameterDefinition("Distance attn", 0.0f, 1.0f, 1.0f);
    parameters[EffectData::kRoomEffectsGain] =
        GetUnityAudioParameterDefinition("Room fx gain", 0.0f, 1.0f, 1.0f);
    parameters[EffectData::kGain] = GetUnityAudioParameterDefinition(
        "Gain", 0.0f, std::numeric_limits<float>::max(), 1.0f);
    parameters[EffectData::kDirectivityAlpha] =
        GetUnityAudioParameterDefinition("Src dir alpha", 0.0f, 1.0f, 0.0f);
    parameters[EffectData::kDirectivitySharpness] =
        GetUnityAudioParameterDefinition("Src dir sharp", 1.0f, 10.0f, 1.0f);
    parameters[EffectData::kListenerDirectivityAlpha] =
        GetUnityAudioParameterDefinition("Lis dir alpha", 0.0f, 1.0f, 0.0f);
    parameters[EffectData::kListenerDirectivitySharpness] =
        GetUnityAudioParameterDefinition("Lis dir sharp", 1.0f, 10.0f, 1.0f);
    parameters[EffectData::kOcclusion] = GetUnityAudioParameterDefinition(
        "Occlusion", 0.0f, std::numeric_limits<float>::max(), 0.0f);
    parameters[EffectData::kQuality] = GetUnityAudioParameterDefinition(
        "Quality", 0.0f /* stereo panning */, 2.0f /* binaural high quality */,
        2.0f /* binaural high quality */);
    parameters[EffectData::kNearFieldEffectGain] =
        GetUnityAudioParameterDefinition("Near-field gain", 0.0f, 9.0f, 1.0f);
    parameters[EffectData::kVolume] =
        GetUnityAudioParameterDefinition("Volume", 0.0f, 1.0f, 1.0f);
    definition[1] = GetUnityAudioEffectDefinition(
        kSpatializerName, SpatializerCreateCallback, SpatializerReleaseCallback,
        SpatializerProcessCallback, SpatializerSetFloatParameterCallback,
        SpatializerGetFloatParameterCallback, EffectData::kNumParams,
        parameters, true);
    definitions[1] = &definition[1];
  }
  *definitionptr = definitions;

  return kNumAudioPlugins;
}

}
}  // namespace igibson
