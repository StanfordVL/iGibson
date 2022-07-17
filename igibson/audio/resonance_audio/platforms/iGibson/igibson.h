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

#ifndef RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_H_
#define RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_H_

#include "api/resonance_audio_api.h"
#include "platforms/common/room_properties.h"
#include "base/audio_buffer.h"
#include "base/constants_and_types.h"
#include "geometrical_acoustics/scene_manager.h"

namespace vraudio {
namespace igibson {


extern std::unique_ptr<SceneManager> scene_manager;

// Output channels must be stereo for the ResonanceAudio system to run properly.
const size_t kNumOutputChannels = 2;

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
    // Maximum number of buffers allowed to record a soundfield, which is set to ~5
    // minutes (depending on the sampling rate and the number of frames per buffer).
    const size_t kMaxNumRecordBuffers = 15000;

    // Record compression quality.
    const float kRecordQuality = 1.0f;
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))



// Stores the necessary components for the ResonanceAudio system. Methods called
// from the native implementation below must check the validity of this
// instance.
struct ResonanceAudioSystem {
    ResonanceAudioSystem(int sample_rate, size_t num_channels,
        size_t frames_per_buffer)
        : api(CreateResonanceAudioApi(num_channels, frames_per_buffer,
            sample_rate)) {
        is_recording_soundfield = false;
    }

    ~ResonanceAudioSystem() {
        std::cout << "Destroying Audio System" << std::endl;
    }

    // ResonanceAudio API instance to communicate with the internal system.
    std::unique_ptr<ResonanceAudioApi> api;

    // Default room properties, which effectively disable the room effects.
    ReflectionProperties null_reflection_properties;
    ReverbProperties null_reverb_properties;

    std::unordered_map<std::string, std::pair<ReflectionProperties,ReverbProperties>> room_to_reflection_and_reverb;

    // Denotes whether the soundfield recording is currently in progress.
    bool is_recording_soundfield;
};


// Initializes the ResonanceAudio system
void Initialize(int sample_rate, size_t num_channels, size_t frames_per_buffer);

// Shuts down the ResonanceAudio system.
void Shutdown();

// Processes the next output buffer and stores the processed buffer in |output|.
// This method must be called from the audio thread.
void RenderAmbisonics(size_t num_frames, std::vector<std::vector<float>>* ambisonicData);

void ProcessListener(size_t num_frames, float* output);

void ProcessListener(size_t num_frames, int16* output);

// Updates the listener's position and rotation.
void SetListenerTransform(float px, float py, float pz, float qx, float qy,
                          float qz, float qw);

// Creates an ambiX format soundfield and connects it to the audio manager.
ResonanceAudioApi::SourceId CreateSoundfield(int num_channels);

// Creates a sound object sub-graph and connects it to the audio manager.
ResonanceAudioApi::SourceId CreateSoundObject(RenderingMode rendering_mode, float min_distance, float max_distance);

// Disconnects the source with |id| from the pipeline and releases its
// resources.
void DestroySource(ResonanceAudioApi::SourceId id);

// Passes the next input buffer of the source to the system. This method must be
// called from the audio thread.
void ProcessSource(ResonanceAudioApi::SourceId id, size_t num_channels,
                   size_t num_frames, float* input);

void ProcessSource(ResonanceAudioApi::SourceId id, size_t num_channels,
                   size_t num_frames, const int16* input);

// Updates the directivity parameters of the source.
void SetSourceDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                          float order);

// Sets the computed distance attenuation of a source.
void SetSourceDistanceAttenuation(ResonanceAudioApi::SourceId id,
                                  float distance_attenuation);

// Updates the gain of the source.
void SetSourceGain(ResonanceAudioApi::SourceId id, float gain);

// Updates the listener directivity parameters of the source.
void SetSourceListenerDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                                  float order);

// Updates the near field effect gain for the source.
void SetSourceNearFieldEffectGain(ResonanceAudioApi::SourceId id,
                                  float near_field_effect_gain);

// Updates the occlusion intensity of the source.
void SetSourceOcclusionIntensity(ResonanceAudioApi::SourceId id,
                                 float intensity);

void EstimateAndUpdateOcclusion(int id);

// Sets the room effects gain for the source.
void SetSourceRoomEffectsGain(ResonanceAudioApi::SourceId id,
                              float room_effects_gain);

// Updates the spread of the source.
void SetSourceSpread(ResonanceAudioApi::SourceId id, float spread_deg);

// Updates the position, rotation and scale of the source.
void SetSourceTransform(ResonanceAudioApi::SourceId id, float px, float py,
                        float pz, float qx, float qy, float qz, float qw);

extern "C" {

// Updates the listener's master gain.
void EXPORT_API SetListenerGain(float gain);

// Updates the stereo speaker mode.
void EXPORT_API SetListenerStereoSpeakerMode(bool enable_stereo_speaker_mode);

// Updates the properties of the room.
void EXPORT_API SetRoomProperties(RoomProperties* room_properties,
                                  float* rt60s);
void EXPORT_API SetRoomReflectionAndReverb(const ReflectionProperties& reflection_properties, const ReverbProperties& reverb_properties);


}  // extern C

}
}  // namespace vraudio

#endif  // RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_H_
