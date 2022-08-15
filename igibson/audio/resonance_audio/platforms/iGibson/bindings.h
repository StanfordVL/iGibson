
#ifndef RESONANCE_AUDIO_PLATFORM_IGIBSON_BINDINGS_H_
#define RESONANCE_AUDIO_PLATFORM_IGIBSON_BINDINGS_H_

#include "api/resonance_audio_api.h"
#include "platforms/common/room_properties.h"
#include "igibson.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vraudio {
namespace igibson {

extern std::shared_ptr<ResonanceAudioSystem> resonance_audio;

py::array_t<int16> InitializeFromMeshAndTest(int num_vertices, int num_triangles,
    py::array_t<float> vertices, py::array_t<int> triangles,
    py::array_t<int> material_indices,
    float scattering_coefficient, const char* fName, py::array_t<float> source_location, py::array_t<float> head_pos);

void InitializeSystem(int frames_per_buffer, int sample_rate);

int InitializeSource(py::array_t<float> source_pos, float min_distance, float max_distance, float source_gain, float near_field_gain, float room_effects_gain);

void SetSourcePosition(int source_id, py::array_t<float> source_pos); 

void SetListenerPosition(py::array_t<float> listener_pos);

void SetNearFieldEffectGain(int source_id, float gain);


py::array_t<int16> ProcessSourceAndListener(int source_id, size_t num_frames, py::array_t<int16> input_arr);


}  // namespace vraudio
}
#endif  // RESONANCE_AUDIO_PLATFORM_IGIBSON_BINDINGS_H_