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

#include "platforms/unity/unity.h"

#include <algorithm>
#include <memory>

#include "base/audio_buffer.h"
#include "base/constants_and_types.h"
#include "base/logging.h"
#include "base/misc_math.h"
#include "graph/resonance_audio_api_impl.h"
#include "platforms/common/room_effects_utils.h"

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
#include "utils/ogg_vorbis_recorder.h"
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))

namespace vraudio {
namespace unity {

namespace {

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
#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
    is_recording_soundfield = false;
    soundfield_recorder.reset(
        new OggVorbisRecorder(sample_rate, kNumFirstOrderAmbisonicChannels,
                              frames_per_buffer, kMaxNumRecordBuffers));
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  }

  // ResonanceAudio API instance to communicate with the internal system.
  std::unique_ptr<ResonanceAudioApi> api;

  // Default room properties, which effectively disable the room effects.
  ReflectionProperties null_reflection_properties;
  ReverbProperties null_reverb_properties;

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  // Denotes whether the soundfield recording is currently in progress.
  bool is_recording_soundfield;

  // First-order ambisonic soundfield recorder.
  std::unique_ptr<OggVorbisRecorder> soundfield_recorder;
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
};

// Singleton |ResonanceAudioSystem| instance to communicate with the internal
// API.
static std::shared_ptr<ResonanceAudioSystem> resonance_audio = nullptr;

}  // namespace

void Initialize(int sample_rate, size_t num_channels,
                size_t frames_per_buffer) {
  CHECK_GE(sample_rate, 0);
  CHECK_EQ(num_channels, kNumOutputChannels);
  CHECK_GE(frames_per_buffer, 0);
  resonance_audio = std::make_shared<ResonanceAudioSystem>(
      sample_rate, num_channels, frames_per_buffer);
}

void Shutdown() { resonance_audio.reset(); }

void ProcessListener(size_t num_frames, float* output) {
  CHECK(output != nullptr);

  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return;
  }

  if (!resonance_audio_copy->api->FillInterleavedOutputBuffer(
          kNumOutputChannels, num_frames, output)) {
    // No valid output was rendered, fill the output buffer with zeros.
    const size_t buffer_size_samples = kNumOutputChannels * num_frames;
    CHECK(!vraudio::DoesIntegerMultiplicationOverflow<size_t>(
        kNumOutputChannels, num_frames, buffer_size_samples));

    std::fill(output, output + buffer_size_samples, 0.0f);
  }

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  if (resonance_audio_copy->is_recording_soundfield) {
    // Record output into soundfield.
    auto* const resonance_audio_api_impl =
        static_cast<ResonanceAudioApiImpl*>(resonance_audio_copy->api.get());
    const auto* soundfield_buffer =
        resonance_audio_api_impl->GetAmbisonicOutputBuffer();
    std::unique_ptr<AudioBuffer> record_buffer(
        new AudioBuffer(kNumFirstOrderAmbisonicChannels, num_frames));
    if (soundfield_buffer != nullptr) {
      for (size_t ch = 0; ch < kNumFirstOrderAmbisonicChannels; ++ch) {
        (*record_buffer)[ch] = (*soundfield_buffer)[ch];
      }
    } else {
      // No output received, fill the record buffer with zeros.
      record_buffer->Clear();
    }
    resonance_audio_copy->soundfield_recorder->AddInput(
        std::move(record_buffer));
  }
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
}

void SetListenerGain(float gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetMasterVolume(gain);
  }
}

void SetListenerStereoSpeakerMode(bool enable_stereo_speaker_mode) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetStereoSpeakerMode(enable_stereo_speaker_mode);
  }
}

void SetListenerTransform(float px, float py, float pz, float qx, float qy,
                          float qz, float qw) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetHeadPosition(px, py, pz);
    resonance_audio_copy->api->SetHeadRotation(qx, qy, qz, qw);
  }
}

ResonanceAudioApi::SourceId CreateSoundfield(int num_channels) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    return resonance_audio_copy->api->CreateAmbisonicSource(num_channels);
  }
  return ResonanceAudioApi::kInvalidSourceId;
}

ResonanceAudioApi::SourceId CreateSoundObject(RenderingMode rendering_mode) {
  SourceId id = ResonanceAudioApi::kInvalidSourceId;
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    id = resonance_audio_copy->api->CreateSoundObjectSource(rendering_mode);
    resonance_audio_copy->api->SetSourceDistanceModel(
        id, DistanceRolloffModel::kNone, 0.0f, 0.0f);
  }
  return id;
}

void DestroySource(ResonanceAudioApi::SourceId id) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->DestroySource(id);
  }
}

void ProcessSource(ResonanceAudioApi::SourceId id, size_t num_channels,
                   size_t num_frames, float* input) {
  CHECK(input != nullptr);

  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetInterleavedBuffer(id, input, num_channels,
                                                    num_frames);
  }
}

void SetSourceDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                          float order) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectDirectivity(id, alpha, order);
  }
}

void SetSourceDistanceAttenuation(ResonanceAudioApi::SourceId id,
                                  float distance_attenuation) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceDistanceAttenuation(
        id, distance_attenuation);
  }
}

void SetSourceGain(ResonanceAudioApi::SourceId id, float gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceVolume(id, gain);
  }
}

void SetSourceListenerDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                                  float order) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectListenerDirectivity(id, alpha,
                                                                 order);
  }
}

void SetSourceNearFieldEffectGain(ResonanceAudioApi::SourceId id,
                                  float near_field_effect_gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectNearFieldEffectGain(
        id, near_field_effect_gain);
  }
}

void SetSourceOcclusionIntensity(ResonanceAudioApi::SourceId id,
                                 float intensity) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectOcclusionIntensity(id, intensity);
  }
}

void SetSourceRoomEffectsGain(ResonanceAudioApi::SourceId id,
                              float room_effects_gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceRoomEffectsGain(id, room_effects_gain);
  }
}

void SetSourceSpread(int id, float spread_deg) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectSpread(id, spread_deg);
  }
}

void SetSourceTransform(int id, float px, float py, float pz, float qx,
                        float qy, float qz, float qw) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourcePosition(id, px, py, pz);
    resonance_audio_copy->api->SetSourceRotation(id, qx, qy, qz, qw);
  }
}

void SetRoomProperties(RoomProperties* room_properties, float* rt60s) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return;
  }
  if (room_properties == nullptr) {
    resonance_audio_copy->api->SetReflectionProperties(
        resonance_audio_copy->null_reflection_properties);
    resonance_audio_copy->api->SetReverbProperties(
        resonance_audio_copy->null_reverb_properties);
    return;
  }

  const auto reflection_properties =
      ComputeReflectionProperties(*room_properties);
  resonance_audio_copy->api->SetReflectionProperties(reflection_properties);
  const auto reverb_properties =
      (rt60s == nullptr)
          ? ComputeReverbProperties(*room_properties)
          : ComputeReverbPropertiesFromRT60s(
                rt60s, room_properties->reverb_brightness,
                room_properties->reverb_time, room_properties->reverb_gain);
  resonance_audio_copy->api->SetReverbProperties(reverb_properties);
}

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
bool StartSoundfieldRecorder() {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return false;
  }
  if (resonance_audio_copy->is_recording_soundfield) {
    LOG(ERROR) << "Another soundfield recording already in progress";
    return false;
  }

  resonance_audio_copy->is_recording_soundfield = true;
  return true;
}

bool StopSoundfieldRecorderAndWriteToFile(const char* file_path,
                                          bool seamless) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return false;
  }
  if (!resonance_audio_copy->is_recording_soundfield) {
    LOG(ERROR) << "No recorded soundfield found";
    return false;
  }

  resonance_audio_copy->is_recording_soundfield = false;
  if (file_path == nullptr) {
    resonance_audio_copy->soundfield_recorder->Reset();
    return false;
  }
  resonance_audio_copy->soundfield_recorder->WriteToFile(
      file_path, kRecordQuality, seamless);
  return true;
}
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))

}  // namespace unity
}  // namespace vraudio
