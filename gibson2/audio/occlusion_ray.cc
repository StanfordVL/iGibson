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

#include "geometrical_acoustics/occlusion_ray.h"

namespace vraudio {

/* occlusion filter function */
void occlusionFilter(void* ptr, RTCRay& ray_i)
{
  OcclusionRay& ray = (OcclusionRay&) ray_i;
  if (ray.hitSlot >= HIT_LIST_LENGTH) return;
  /* The occlusion filter function may be called multiple times with
   * the same hit. We remember the last N hits, and skip duplicates. */
  for (size_t slot=0; i<ray.hitSlot; i++) {
    if (ray.hit_geomIDs[slot] == ray.geomID && ray.hit_primIDs[slot] == ray.primID) {
      ray.geomID = RTC_INVALID_GEOMETRY_ID;
      return;
    }
  }

  /* store hit in hit list */
  ray.hit_geomIDs[ray.hitSlot] = ray.geomID;
  ray.hit_primIDs[ray.hitSlot] = ray.primID;
  ray.hitSlot++;
}

/* gathers hits in a single pass */
void RayHits(const OcclusionRay& ray_i, HitList& hits_o, RTCScene scene)
{
    /* trace ray to gather all hits */
    auto ray = ray_i;
    IntersectContext context(hits_o);
    rtcInitIntersectContext(&context.context);
    context.context.filter = gather_all_hits;
    rtcIntersect1(scene, &context.context, RTCRayHit_(ray));
    //RayStats_addRay(stats);

    /* sort hits by extended order */
    std::sort(&context.hits.hits[context.hits.begin], &context.hits.hits[context.hits.end]);

    /* ignore duplicated hits that can occur for tesselated primitives */
    if (hits_o.size())
    {
        unsigned int i = 0;
        while (hits_o.hits[i].t == 0) {
            i = i + 1;
            LOG(WARNING) << "Hit at t=0" << std::endl;
        }
        unsigned int j = i + 1;
        for (; j < hits_o.size(); j++) {
            if (hits_o.hits[i] == hits_o.hits[j]) continue;
            hits_o.hits[++i] = hits_o.hits[j];
            LOG(WARNING) << "Duplicate hit" << std::endl;
        }
        hits_o.end = i + 1;
    }
}



}  // namespace vraudio

#endif  // RESONANCE_AUDIO_GEOMETRICAL_ACOUSTICS_OCCLUSION_RAY_H_
