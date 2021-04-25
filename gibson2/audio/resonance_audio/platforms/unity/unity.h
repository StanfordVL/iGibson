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

#ifndef RESONANCE_AUDIO_PLATFORM_UNITY_UNITY_H_
#define RESONANCE_AUDIO_PLATFORM_UNITY_UNITY_H_

#include "api/resonance_audio_api.h"
#include "platforms/common/room_properties.h"

namespace vraudio {
namespace unity {

// Initializes the ResonanceAudio system with Unity audio engine settings.
void Initialize(int sample_rate, size_t num_channels, size_t frames_per_buffer);

// Shuts down the ResonanceAudio system.
void Shutdown();

// Processes the next output buffer and stores the processed buffer in |output|.
// This method must be called from the audio thread.
void ProcessListener(size_t num_frames, float* output);

// Updates the listener's position and rotation.
void SetListenerTransform(float px, float py, float pz, float qx, float qy,
                          float qz, float qw);

// Creates an ambiX format soundfield and connects it to the audio manager.
ResonanceAudioApi::SourceId CreateSoundfield(int num_channels);

// Creates a sound object sub-graph and connects it to the audio manager.
ResonanceAudioApi::SourceId CreateSoundObject(RenderingMode rendering_mode);

// Disconnects the source with |id| from the pipeline and releases its
// resources.
void DestroySource(ResonanceAudioApi::SourceId id);

// Passes the next input buffer of the source to the system. This method must be
// called from the audio thread.
void ProcessSource(ResonanceAudioApi::SourceId id, size_t num_channels,
                   size_t num_frames, float* input);

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

// Sets the room effects gain for the source.
void SetSourceRoomEffectsGain(ResonanceAudioApi::SourceId id,
                              float room_effects_gain);

// Updates the spread of the source.
void SetSourceSpread(ResonanceAudioApi::SourceId id, float spread_deg);

// Updates the position, rotation and scale of the source.
void SetSourceTransform(ResonanceAudioApi::SourceId id, float px, float py,
                        float pz, float qx, float qy, float qz, float qw);

// NOTE: all exported symbols must be added to "unity_android.lds".
extern "C" {

// Updates the listener's master gain.
void EXPORT_API SetListenerGain(float gain);

// Updates the stereo speaker mode.
void EXPORT_API SetListenerStereoSpeakerMode(bool enable_stereo_speaker_mode);

// Updates the properties of the room.
void EXPORT_API SetRoomProperties(RoomProperties* room_properties,
                                  float* rt60s);

// Starts the soundfield recorder.
bool EXPORT_API StartSoundfieldRecorder();

// Stops the soundfield recorder and writes the recorded data into file.
bool EXPORT_API StopSoundfieldRecorderAndWriteToFile(const char* file_path,
                                                     bool seamless);

}  // extern C

}  // namespace unity
}  // namespace vraudio

#endif  // RESONANCE_AUDIO_PLATFORM_UNITY_UNITY_H_
