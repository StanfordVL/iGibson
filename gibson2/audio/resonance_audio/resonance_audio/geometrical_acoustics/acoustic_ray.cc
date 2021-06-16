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

#include "geometrical_acoustics/acoustic_ray.h"

#include <iostream>
#include <limits>

namespace vraudio {

// Static member initialization.
const float AcousticRay::kInfinity = std::numeric_limits<float>::infinity();
const float AcousticRay::kRayEpsilon = 1e-4f;



/* occlusion filter function */
void occlusionFilter(void* ptr, RTCRay& ray_i)
{
  AcousticRay& ray = (AcousticRay&) ray_i;
  
  if (ray.hitSlot >= HIT_LIST_LENGTH) return;
  /* The occlusion filter function may be called multiple times with
   * the same hit. We remember the last N hits, and skip duplicates. */

  
  for (size_t slot=0; slot<ray.hitSlot; slot++) {
      //API does not guarantee validitiy of anything except geomID = 0
    if (ray.hit_tfars[slot] == ray.t_far()) { //} && ray.hit_geomIDs[slot] == ray.geomID && ray.hit_primIDs[slot] == ray.primID) {
      ray.geomID = RTC_INVALID_GEOMETRY_ID;
      return;
    }
  }

  /* store hit in hit list */
  //ray.hit_geomIDs[ray.hitSlot] = ray.geomID;
  //ray.hit_primIDs[ray.hitSlot] = ray.primID;
  ray.hit_tfars[ray.hitSlot] = ray.t_far();
  ray.hitSlot++;
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
}


}  // namespace vraudio
