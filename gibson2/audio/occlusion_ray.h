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

#ifndef RESONANCE_AUDIO_GEOMETRICAL_ACOUSTICS_OCCLUSION_RAY_H_
#define RESONANCE_AUDIO_GEOMETRICAL_ACOUSTICS_OCCLUSION_RAY_H_

#include <array>
#include <vector>

#include "embree2/rtcore.h"
#include "embree2/rtcore_ray.h"
#include "embree2/rtcore_scene.h"
#include "base/constants_and_types.h"
#include "base/logging.h"

namespace vraudio {

#define HIT_LIST_LENGTH 16

// A class extending Embree's RTCRay (https://embree.github.io/api.html) with
// data needed for acoustic computations.
// It exposes useful fields through accessors.
class RTCORE_ALIGN(16) OcclusionRay : public RTCRay {
 public:
  // A constant used to indicate that the ray extends to infinity, if
  // ray.t_far() == AcousticRay::kInfinity.
  static const float kInfinity;

  // Used to offset a ray's origin slightly so that it will not
  // intersect with the same geometry/primitive that it was generated from
  // (by reflection, transmission, diffraction, etc.).
  static const float kRayEpsilon = 1e-4f;

  // Constructor.
  //
  // @param origin Origin of the ray.
  // @param direction Direction of the ray.
  // @param t_near Ray parameter corresponding to the start of the ray.
  // @param t_far Ray parameter corresponding to the end of the ray. Pass in
  //     AcousticRay::kInfinity if there is no end point.
  // @param energies Ray energies for all frequency bands.
  // @param ray_type Type of ray.
  // @param prior_distance Distance traveled before this ray.
  OcclusionRay(const float origin[3], const float direction[3], float t_near, float t_far) {
    org[0] = origin[0];
    org[1] = origin[1];
    org[2] = origin[2];
    dir[0] = direction[0];
    dir[1] = direction[1];
    dir[2] = direction[2];
    tnear = t_near;
    tfar = t_far;
    Ng[0] = 0.0f;
    Ng[1] = 0.0f;
    Ng[2] = 0.0f;
    geomID = RTC_INVALID_GEOMETRY_ID;

    // Members in RTCRay that we do not use (or whose initial values we do not
    // care) are not initialized:
    // align0, align1, align2, time, mask, u, v, primID, instID.

    // we remember up to 16 hits to ignore duplicate hits
    unsigned int hit_geomIDs[HIT_LIST_LENGTH];
    unsigned int hit_primIDs[HIT_LIST_LENGTH];
    unsigned int hitSlot;

  }

  // Ray origin.
  const float* origin() const { return org; }
  void set_origin(const float origin[3]) {
    org[0] = origin[0];
    org[1] = origin[1];
    org[2] = origin[2];
  }

  // Ray direction.
  const float* direction() const { return dir; }
  void set_direction(const float direction[3]) {
    dir[0] = direction[0];
    dir[1] = direction[1];
    dir[2] = direction[2];
  }

  // Ray parameter t corresponding to the start of the ray segment.
  const float t_near() const { return tnear; }
  void set_t_near(float t_near) { tnear = t_near; }

  // Ray parameter t corresponding to the end of the ray segment.
  const float t_far() const { return tfar; }
  void set_t_far(float t_far) { tfar = t_far; }

  const size_t num_hits() const {return hitSlot;}

  // Finds the first intersection between this ray and a scene. Some fields
  // will be filled/mutated, which can be examined by the following functions:
  // - t_far()
  // - intersected_geometry_normal()
  // - intersected_geometry_id()
  // - intersected_primitive_id()
  //
  // @param scene An RTCScene to test the intersection.
  // @return True if an intersection is found.
  bool Intersect(RTCScene scene) {
    rtcIntersect(scene, *this);
    return geomID != RTC_INVALID_GEOMETRY_ID;
  }

};


/* occlusion filter function */
void occlusionFilter(void* ptr, RTCRay& ray_i);

/* gathers hits in a single pass */
void RayHits(const OcclusionRay& ray_i, RTCScene scene);



}  // namespace vraudio

#endif  // RESONANCE_AUDIO_GEOMETRICAL_ACOUSTICS_OCCLUSION_RAY_H_
