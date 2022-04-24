
#include <algorithm>
#include <memory>

#include "platforms/iGibson/bindings.h"
#include "platforms/iGibson/igibson_reverb_computer.h"
#include "platforms/common/room_properties.h"


#include "base/audio_buffer.h"
#include "base/constants_and_types.h"
#include "base/logging.h"
#include "base/misc_math.h"
#include "graph/resonance_audio_api_impl.h"
#include "platforms/common/room_effects_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <iostream>
#include <fstream>
#include <chrono>

#include "utils/wav.h"

namespace py = pybind11;

namespace vraudio {
namespace igibson {


    py::array_t<int16> InitializeFromMeshAndTest(int num_vertices, int num_triangles,
        py::array_t<float> vertices, py::array_t<int> triangles,
        py::array_t<int> material_indices,
        float scattering_coefficient, const char * fName, py::array_t<float> source_location, py::array_t<float> head_pos) {

        py::print("Entering Function");

        // Number of frames per buffer.
        const size_t kFramesPerBuffer = 16;

        // Sampling rate.
        const int kSampleRate = 48000;

        Initialize(kSampleRate, 2, kFramesPerBuffer);

        py::print("Post-Initialize");

        py::buffer_info mi_buf = material_indices.request();
        int* material_indices_arr = (int*)mi_buf.ptr;

        py::buffer_info v_buf = vertices.request();
        float* verts = (float*)v_buf.ptr;

        py::buffer_info t_buf = triangles.request();
        int* tris = (int*)t_buf.ptr;

        py::print("Converted arrays");
        auto time1 = std::chrono::high_resolution_clock::now();
        InitializeReverbComputer(num_vertices, num_triangles,
            verts, tris,
            material_indices_arr,
            scattering_coefficient);
        auto time2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> fp_ms = time2 - time1;
        py::print("Reverb Computer Initialized in ", fp_ms.count(), "ms");

        // Ray-tracing related fields.
        const int kNumRays = 200000;//20000;
        const int kNumRaysPerBatch = 20000;//2000;
        const int kMaxDepth = 3;
        const float kEnergyThresold = 1e-6f;
        const float listener_sphere_radius = 0.1f;
        const size_t impulse_response_num_samples = 1000000;//96000;

        RoomProperties proxy_room_properties;
        float rt60s [kNumReverbOctaveBands];

        py::buffer_info sl_buf = source_location.request();
        float* source_location_arr = (float*)sl_buf.ptr;

        py::print("Pre-compute rt60s");
        time1 = std::chrono::high_resolution_clock::now();
        if (!ComputeRt60sAndProxyRoom(kNumRays, kNumRaysPerBatch,
            kMaxDepth, kEnergyThresold,
            source_location_arr,
            listener_sphere_radius, kSampleRate,
            impulse_response_num_samples,
            rt60s,
            &proxy_room_properties)) assert(1);

        time2 = std::chrono::high_resolution_clock::now();

        fp_ms = time2 - time1;
        py::print("Computed rt60s in ", fp_ms.count(), "ms");

        time1 = std::chrono::high_resolution_clock::now();
        SetRoomProperties(&proxy_room_properties, rt60s);
        time2 = std::chrono::high_resolution_clock::now();

        fp_ms = time2 - time1;

        py::print("Room properties set in ", fp_ms.count(), "ms");

        std::filebuf fb;
        time1 = std::chrono::high_resolution_clock::now();
        if (!fb.open(fName, std::ios::in)) assert(1);

        std::istream is(&fb);
        std::unique_ptr<const Wav> wav = Wav::CreateOrNull(&is);
        fb.close();
        time2 = std::chrono::high_resolution_clock::now();

        fp_ms = time2 - time1;
        py::print("Finished reading file in ", fp_ms.count(), "ms");

        assert(wav != nullptr);

        py::print("Creating source");
        time1 = std::chrono::high_resolution_clock::now();
        ResonanceAudioApi::SourceId source_id = CreateSoundObject(RenderingMode::kBinauralHighQuality, 0.0f, 0.0f);

        py::print("Set source position");
        resonance_audio->api->SetSourcePosition(source_id, source_location_arr[0], source_location_arr[1], source_location_arr[2]);
        

        py::buffer_info hp_buf = head_pos.request();
        int* head_pos_arr = (int*)hp_buf.ptr;

        py::print("Set head position");
        resonance_audio->api->SetHeadPosition(head_pos_arr[0], head_pos_arr[1], head_pos_arr[2]);
        time2 = std::chrono::high_resolution_clock::now();

        SetListenerStereoSpeakerMode(true);

        fp_ms = time2 - time1;
        py::print("Source and Head initialization in ", fp_ms.count(), "ms");

        int num_frames = (int) wav->interleaved_samples().size() / wav->GetNumChannels();

        time1 = std::chrono::high_resolution_clock::now();
        
        // Process the next buffer.
        ProcessSource(source_id, 1, kFramesPerBuffer, &(wav->interleaved_samples()[0]));

        int sum = 0;
        for (auto n : wav->interleaved_samples()) {
            sum += n;
        }
        std::cout << "Sum of source buffer " << sum << std::endl;
        time2 = std::chrono::high_resolution_clock::now();

        fp_ms = time2 - time1;
        py::print("Set buffer in ", fp_ms.count(), "ms");

        time1 = std::chrono::high_resolution_clock::now();
        py::array_t<int16> output_py = py::array_t<int16>(kNumOutputChannels * kFramesPerBuffer);
        py::buffer_info out_buf = output_py.request();
        int16* output = static_cast<int16*>(out_buf.ptr);

        ProcessListener(kFramesPerBuffer, output);
        time2 = std::chrono::high_resolution_clock::now();

        fp_ms = time2 - time1;
        py::print("Got output in ", fp_ms.count(), "ms");

        //py::print("Deleting Scene Manager explicitly to avoid mutex bug");
        //DeleteSceneManager();

        py::print("Returning");

        int sum1 = 0;
        for (int i = 0; i < out_buf.size; ++i) {
            sum1 += output[i];
        }
        std::cout << "Sum of output buffer " << sum1 << std::endl;

        return output_py; 
    }

    void InitializeSystem(int frames_per_buffer, int sample_rate) {
        Initialize(sample_rate, 2, frames_per_buffer);
    }

    void DisableRoomEffects() {
        resonance_audio->api->EnableRoomEffects(false);
    }

    void ShutdownSystem() {
        DeleteSceneManager();
        Shutdown();
    }

    void DeleteMesh() {
        DeleteSceneManager();
    }


    void LoadMesh(int num_vertices, int num_triangles,
        py::array_t<float> vertices, py::array_t<int> triangles,
        py::array_t<int> material_indices,
        float scattering_coefficient) {

        py::buffer_info mi_buf = material_indices.request();
        int* material_indices_arr = (int*)mi_buf.ptr;

        py::buffer_info v_buf = vertices.request();
        float* verts = (float*)v_buf.ptr;

        py::buffer_info t_buf = triangles.request();
        int* tris = (int*)t_buf.ptr;

        auto time1 = std::chrono::high_resolution_clock::now();
        InitializeReverbComputer(num_vertices, num_triangles,
            verts, tris,
            material_indices_arr,
            scattering_coefficient);
        auto time2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> fp_ms = time2 - time1;
        py::print("Reverb Computer Initialized in ", fp_ms.count(), "ms");
    }

    void RegisterReverbProbe(const std::string &room,
                             py::array_t<float> sample_pos,
                             int kSampleRate,
                             int kNumRays,
                             int kNumRaysPerBatch,
                             int kMaxDepth,
                             float kEnergyThresold,
                             float listener_sphere_radius,
                             size_t impulse_response_num_samples) {
        RoomProperties proxy_room_properties;
        float rt60s [kNumReverbOctaveBands];

        py::buffer_info sp_buf = sample_pos.request();
        float* sample_pos_arr = (float*)sp_buf.ptr;
        
        ComputeRt60sAndProxyRoom(kNumRays, kNumRaysPerBatch,
            kMaxDepth, kEnergyThresold,
            sample_pos_arr,
            listener_sphere_radius, kSampleRate,
            impulse_response_num_samples,
            rt60s,
            &proxy_room_properties);

        //SetRoomProperties(&proxy_room_properties, rt60s);
        const auto reflection_properties = ComputeReflectionProperties(proxy_room_properties);
        const auto reverb_properties = (rt60s == nullptr) ? ComputeReverbProperties(proxy_room_properties)
          : ComputeReverbPropertiesFromRT60s(
                rt60s, proxy_room_properties.reverb_brightness,
                proxy_room_properties.reverb_time, proxy_room_properties.reverb_gain);

        resonance_audio->room_to_reflection_and_reverb[room] = std::make_pair(reflection_properties, reverb_properties);

        py::print("Reverb and reflection properties computed for probe [", room, "]");
    }

    void SetRoomPropertiesFromProbe(const std::string &room) {
        const auto reflection_and_reverb = resonance_audio->room_to_reflection_and_reverb[room];
        SetRoomReflectionAndReverb(reflection_and_reverb.first, reflection_and_reverb.second);
    }

    int InitializeSource(py::array_t<float> source_pos, float min_distance, float max_distance, float source_gain, float near_field_gain, float room_effects_gain) {
        ResonanceAudioApi::SourceId source_id = CreateSoundObject(RenderingMode::kBinauralHighQuality, min_distance, max_distance);

        py::buffer_info sl_buf = source_pos.request();
        float* source_location_arr = (float*)sl_buf.ptr;

        SetSourcePosition(source_id, source_pos);
        SetSourceGain(source_id, source_gain);
        SetNearFieldEffectGain(source_id, near_field_gain);
        SetSourceRoomEffectsGain(source_id, room_effects_gain);
        SetSourceSpread(source_id, 0);

        return source_id;
    }

    void SetNearFieldEffectGain(int source_id, float gain) {
        resonance_audio->api->SetSoundObjectNearFieldEffectGain(source_id, gain);
    }

    void SetSourcePosition(int source_id, py::array_t<float> source_pos) {
        py::buffer_info sp_buf = source_pos.request();
        float* source_pos_arr = (float*)sp_buf.ptr;

        resonance_audio->api->SetSourcePosition(source_id, source_pos_arr[0], source_pos_arr[1], source_pos_arr[2]);
    }

    void SetListenerPosition(py::array_t<float> listener_pos) {
        py::buffer_info hp_buf = listener_pos.request();
        float* head_pos_arr = (float*)hp_buf.ptr;

        resonance_audio->api->SetHeadPosition(head_pos_arr[0], head_pos_arr[1], head_pos_arr[2]);
    }

    void SetListenerPositionAndRotation(py::array_t<float> listener_pos, py::array_t<float> listener_rot) {
        py::buffer_info hp_buf = listener_pos.request();
        float* head_pos_arr = (float*)hp_buf.ptr;

        py::buffer_info hr_buf = listener_rot.request();
        float* head_rot_arr = (float*)hr_buf.ptr;

        SetListenerTransform(head_pos_arr[0], head_pos_arr[1], head_pos_arr[2], head_rot_arr[0], head_rot_arr[1], head_rot_arr[2], head_rot_arr[3]);
    }



    void SetSourceOcclusion(int source_id, float occl) {
        resonance_audio->api->SetSoundObjectOcclusionIntensity(source_id, occl);
    }
    

    void ProcessSourceBind(int source_id, size_t num_frames, py::array_t<int16> input_arr) {

        py::buffer_info in_buf = input_arr.request();
        int16* input = static_cast<int16*>(in_buf.ptr);

        // Process the next buffer.
        ProcessSource(source_id, 1, num_frames, input);
    }

    py::array_t<int16> ProcessListenerBind(size_t num_frames) {
        py::array_t<int16> output_py = py::array_t<int16>(kNumOutputChannels * num_frames);
        py::buffer_info out_buf = output_py.request();
        int16* output = static_cast<int16*>(out_buf.ptr);

        ProcessListener(num_frames, output);
        return output_py; 
    }

    std::vector<std::vector<float>> RenderAmbisonicsBind(size_t num_frames) {
        std::vector<float> audioData(num_frames, 0);
        std::vector<std::vector<float>> ambisonicData(kNumThirdOrderAmbisonicChannels, audioData);

        RenderAmbisonics(num_frames, &ambisonicData);
        return ambisonicData; 
    }

    py::array_t<int16> ProcessSourceAndListener(int source_id, size_t num_frames, py::array_t<int16> input_arr) {
        
        ProcessSourceBind(source_id, num_frames, input_arr);
        EstimateAndUpdateOcclusion(source_id);

        return ProcessListenerBind(num_frames);
    }

    void DeleteSource(int source_id) {
        resonance_audio->api->DestroySource(source_id);
    }

    PYBIND11_MODULE(audio, m) {
        //m.def<decltype(&InitializeFromMeshAndTest)>("InitializeFromMeshAndTest", &InitializeFromMeshAndTest);
        m.def("InitializeFromMeshAndTest", &InitializeFromMeshAndTest, py::return_value_policy::automatic, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("InitializeSystem", &InitializeSystem, py::return_value_policy::automatic, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("DisableRoomEffects", &DisableRoomEffects, py::return_value_policy::automatic, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("LoadMesh", &LoadMesh, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("ShutdownSystem", &ShutdownSystem, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("DeleteMesh", &DeleteMesh, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("RegisterReverbProbe", &RegisterReverbProbe, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("InitializeSource", &InitializeSource, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("SetNearFieldEffectGain", &SetNearFieldEffectGain, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("SetSourcePosition", &SetSourcePosition, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("SetListenerPosition", &SetListenerPosition, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        
        m.def("SetListenerPositionAndRotation", &SetListenerPositionAndRotation, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("ProcessSource", &ProcessSourceBind, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("ProcessListener", &ProcessListenerBind, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("RenderAmbisonics", &RenderAmbisonicsBind, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("SetSourceOcclusion", &SetSourceOcclusion, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("RegisterReverbProbe", &RegisterReverbProbe, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("SetRoomPropertiesFromProbe", &SetRoomPropertiesFromProbe, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());

        m.def("ProcessSourceAndListener", &ProcessSourceAndListener, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        
        m.def("SetSourceListenerDirectivity", &SetSourceListenerDirectivity, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
        m.def("DestroySource", &DestroySource, py::call_guard<py::scoped_ostream_redirect,
                py::scoped_estream_redirect>());
    }
}
}  // namespace vraudio


