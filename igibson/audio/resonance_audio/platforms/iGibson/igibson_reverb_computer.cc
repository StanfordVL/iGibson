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

#include "platforms/iGibson/igibson_reverb_computer.h"

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "api/resonance_audio_api.h"
#include "base/constants_and_types.h"
#include "base/logging.h"
#include "geometrical_acoustics/estimating_rt60.h"
#include "geometrical_acoustics/impulse_response_computer.h"
#include "geometrical_acoustics/path_tracer.h"
#include "geometrical_acoustics/proxy_room_estimator.h"
#include "geometrical_acoustics/scene_manager.h"
#include "platforms/common/room_effects_utils.h"

namespace vraudio {
namespace igibson {

std::unique_ptr<SceneManager> scene_manager = nullptr;

namespace {


static std::function<float()> random_number_generator = nullptr;
static std::unique_ptr<PathTracer> path_tracer = nullptr;

// Copy a float array to a vertex buffer (a vector of vertices). Every three
// floats make up one vertex.
std::vector<Vertex> CopyFloatArrayToVertexBuffer(int input_num_vertices,
                                                 const float* vertices) {
  CHECK_GE(input_num_vertices, 0);
  const size_t num_vertices = static_cast<size_t>(input_num_vertices);
  std::vector<Vertex> vertex_buffer(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    vertex_buffer[i].x = vertices[3 * i + 0];
    vertex_buffer[i].y = vertices[3 * i + 1];
    vertex_buffer[i].z = vertices[3 * i + 2];
  }
  return vertex_buffer;
}

// Copy an int array to a triangle buffer (a vector of triangles). Every three
// ints make up one triangle.
std::vector<Triangle> CopyIntArrayToTriangleBuffer(int input_num_triangles,
                                                   const int* triangles) {
  CHECK_GE(input_num_triangles, 0);
  const size_t num_triangles = static_cast<size_t>(input_num_triangles);
  std::vector<Triangle> triangle_buffer(num_triangles);
  for (size_t i = 0; i < num_triangles; ++i) {
    triangle_buffer[i].v0 = triangles[3 * i + 0];
    triangle_buffer[i].v1 = triangles[3 * i + 1];
    triangle_buffer[i].v2 = triangles[3 * i + 2];
  }
  return triangle_buffer;
}

// Create a scene with |num_vertices| vertices, |num_triangles| triangles,
// and the vertices and triangles are represented as a float array and an int
// array respectively.
void CreateScene(int num_vertices, int num_triangles, float* vertices,
                 int* triangles) {
  const std::vector<Vertex> vertex_buffer =
      CopyFloatArrayToVertexBuffer(num_vertices, vertices);
  const std::vector<Triangle> triangle_buffer =
      CopyIntArrayToTriangleBuffer(num_triangles, triangles);
  scene_manager.reset(new SceneManager());
  scene_manager->BuildScene(vertex_buffer, triangle_buffer);

  LOG(INFO) << "Scene built with " << scene_manager->num_triangles()
            << " triangles and " << scene_manager->num_vertices()
            << " vertices";
}

// Set materials (described as an int array of indices, |material_indices|) to
// |num_triangles| triangles, with all scattering coefficients being the same
// |scattering_coefficient|.
void SetMaterialsToTriangles(float scattering_coefficient, size_t num_triangles,
                             int* material_indices) {
  // First group triangles according to material indices.
  std::unordered_map<int, std::unordered_set<unsigned int>>
      triangle_indices_from_material;
  for (size_t i = 0; i < num_triangles; ++i) {
    const int material_index = material_indices[i];
    const unsigned int triangle_index = static_cast<unsigned int>(i);
    triangle_indices_from_material[material_index].insert(triangle_index);
  }

  // Set each material to the corresponding group of triangles.
  for (const auto& material_triangle_indices_pair :
       triangle_indices_from_material) {
    const int material_index = material_triangle_indices_pair.first;
    const auto& triangle_indices = material_triangle_indices_pair.second;
    const auto& room_material = GetRoomMaterial(material_index);

    std::array<float, kNumReverbOctaveBands> absorption_coefficients;
    for (size_t band = 0; band < kNumReverbOctaveBands; ++band) {
      absorption_coefficients[band] =
          room_material.absorption_coefficients[band];
    }
    ReflectionKernel reflection_kernel(absorption_coefficients,
                                       scattering_coefficient,
                                       random_number_generator);
    scene_manager->AssociateReflectionKernelToTriangles(reflection_kernel,
                                                        triangle_indices);
  }
}

// Initialize the unique pointers of a AcousticSource and a vector of one
// AcousticListener. The positions are described as a float array,
// |sample_position_array|, and each listener will contain
// |impulse_response_num_samples| samples of impulse responses.
void InitializeSourceAndListeners(
    const float sample_position_array[3], size_t impulse_response_num_samples,
    std::unique_ptr<AcousticSource>* source,
    std::unique_ptr<std::vector<AcousticListener>>* listeners) {
  const Eigen::Vector3f sample_position(sample_position_array[0],
                                        sample_position_array[1],
                                        sample_position_array[2]);
  std::array<float, kNumReverbOctaveBands> energies;
  energies.fill(1.0f);
  source->reset(
      new AcousticSource(sample_position, energies, random_number_generator));
  listeners->reset(new std::vector<AcousticListener>);
  (*listeners)->emplace_back(sample_position, impulse_response_num_samples);
}

}  // namespace


//Workaround to avoid embree mutex issue when not explicitly deleting this before returning
void DeleteSceneManager(){
  if (scene_manager) {
    scene_manager.reset();
  }
}


void InitializeReverbComputer(int num_vertices, int num_triangles,
                              float* vertices, int* triangles,
                              int* material_indices,
                              float scattering_coefficient) {
  static std::default_random_engine random_engine;
  static std::uniform_real_distribution<float> distribution;
  if (random_number_generator == nullptr) {
    random_number_generator = []() { return distribution(random_engine); };
  }

  CreateScene(num_vertices, num_triangles, vertices, triangles);
  SetMaterialsToTriangles(scattering_coefficient,
                          static_cast<size_t>(num_triangles), material_indices);
  path_tracer.reset(new PathTracer(*scene_manager));
}

bool ComputeRt60sAndProxyRoom(int total_num_paths, int num_paths_per_batch,
                              int max_depth, float energy_threshold,
                              float sample_position[3],
                              float listener_sphere_radius, float sampling_rate,
                              int impulse_response_num_samples,
                              float* output_rt60s,
                              RoomProperties* output_proxy_room) {
  if (scene_manager == nullptr || path_tracer == nullptr ||
      random_number_generator == nullptr) {
    LOG(ERROR) << "InitializeReverbComputer must be called first";
    return false;
  }

  // Initialize the source and listener at the sample position.
  std::unique_ptr<AcousticSource> source;
  std::unique_ptr<std::vector<AcousticListener>> listeners;
  InitializeSourceAndListeners(
      sample_position, static_cast<size_t>(impulse_response_num_samples),
      &source, &listeners);

  // Impulse response computer.
  ImpulseResponseComputer impulse_response_computer(
      listener_sphere_radius, sampling_rate, std::move(listeners),
      scene_manager.get());

  // Proxy room estimator.
  ProxyRoomEstimator proxy_room_estimator;

  // Tracing rays in batches.
  const size_t num_batches =
      static_cast<size_t>(std::ceil(static_cast<float>(total_num_paths) /
                                    static_cast<float>(num_paths_per_batch)));
  for (size_t batch_index = 0; batch_index < num_batches; ++batch_index) {
    const size_t min_num_paths =
        std::min(static_cast<size_t>(num_paths_per_batch),
                 static_cast<size_t>(total_num_paths) -
                     batch_index * num_paths_per_batch);
    const auto& paths = path_tracer->TracePaths(*source, min_num_paths,
                                                static_cast<size_t>(max_depth),
                                                energy_threshold);

    // For estimating RT60s.
    impulse_response_computer.CollectContributions(paths);

    // For estimating a proxy room.
    proxy_room_estimator.CollectHitPointData(paths);
  }

  // Estimate RT60s.
  const auto& energy_impulse_responses =
      impulse_response_computer.GetFinalizedListeners()
          .at(0)
          .energy_impulse_responses;
  for (size_t band_index = 0; band_index < kNumReverbOctaveBands;
       ++band_index) {
    output_rt60s[band_index] =
        EstimateRT60(energy_impulse_responses[band_index], sampling_rate);
    //LOG(INFO) << "Estimiated RT60 for band[" << band_index
    //          << "]= " << output_rt60s[band_index];
  }

  // Estimate a proxy room. Use a default room when failed.
  const float outlier_portion = 0.1f;
  if (!proxy_room_estimator.EstimateCubicProxyRoom(outlier_portion,
                                                   output_proxy_room)) {
    LOG(WARNING) << "Proxy room estimation failed; a default room is used";
    *output_proxy_room = RoomProperties();
  }

  return true;
}

}
}  // namespace igibson
