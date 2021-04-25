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

#ifndef RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_REVERB_COMPUTER_H_
#define RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_REVERB_COMPUTER_H_

#include "api/resonance_audio_api.h"
#include "platforms/common/room_properties.h"

namespace vraudio {
namespace igibson {





extern "C" {

//Workaround to avoid embree mutex issue when not explicitly deleting this before returning
void EXPORT_API DeleteSceneManager();

// Initializes the scene and necessary data for ray tracing. The input vertices
// are represented as an array of floating points, with three floats per vertex
// lined up one after another. The triangles are similarly represented by
// an array of vertex indices, with every three indices per triangle.
// For example, for a scene of four vertices and two triangles sharing an edge:
//   Vertices: v1 = {v1_x, v1_y, v1_z}
//             v2 = {v2_x, v2_y, v2_z}
//             v3 = {v3_x, v3_y, v3_z}
//             v4 = {v4_x, v4_y, v4_z}
//   Triangles: t1 = {v1, v2, v3}
//              t2 = {v2, v3, v4}
//
// Then the |vertices| is an array of 12 elements:
//     {v1_x, v1_y, vy_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z, v4_x, v4_y, v4_z}
// and |triangles| is an array of 6 elements:
//     {v1, v2, v3, v2, v3, v4}.
//
// @param num_vertices Number of vertices in the scene.
// @param num_triangles Number of triangles in the scene.
// @param vertices Vertices as an array of floating points.
// @param triangles Triangles as an array of indices.
// @param material_indices Surface material indices for each triangle.
// @param scattering_coefficient Scattering coefficient of all triangles.
void EXPORT_API InitializeReverbComputer(int num_vertices, int num_triangles,
                                         float* vertices, int* triangles,
                                         int* material_indices,
                                         float scattering_coefficient);

// Computes the RT60s and proxy room using ray tracing.
//
// @param total_num_paths Total number of ray paths to be traced.
// @param num_paths_per_batch Number of paths per batch.
// @param max_depth Maximum depth of tracing performed along a path.
// @param energy_threshold Energy threshold below which the tracing stops.
// @param sample_position Sample position to shoot and collect rays from.
// @param listener_sphere_radius Radius of listener spheres (m).
// @param sampling_rate Sampling rate (Hz).
// @param impulse_response_num_samples Number of samples in the energy impulse
//     response for each frequency band.
// @param output_rt60s Output estimated RT60s.
// @param output_proxy_room Output estimated proxy room.
// @return True if the computation succeeded.
bool EXPORT_API ComputeRt60sAndProxyRoom(
    int total_num_paths, int num_paths_per_batch, int max_depth,
    float energy_threshold, float sample_position[3],
    float listener_sphere_radius, float sampling_rate,
    int impulse_response_num_samples, float* output_rt60s,
    RoomProperties* output_proxy_room);

}  // extern C

}
}  // namespace igibson

#endif  // RESONANCE_AUDIO_PLATFORM_IGIBSON_IGIBSON_REVERB_COMPUTER_H_
