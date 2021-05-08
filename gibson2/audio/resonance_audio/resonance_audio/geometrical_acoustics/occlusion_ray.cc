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

/* Filter callback function that gathers all hits */
void gather_all_hits(const struct RTCFilterFunctionNArguments* args)
{
    assert(*args->valid == -1);
    IntersectContext* context = (IntersectContext*)args->context;
    HitList& hits = context->hits;
    RTCRay* ray = (RTCRay*)args->ray;
    RTCHit* hit = (RTCHit*)args->hit;
    assert(args->N == 1);
    args->valid[0] = 0; // ignore all hits

    /* avoid overflow of hits array */
    if (hits.end >= MAX_TOTAL_HITS) return;

    /* add hit to list */
    hits.hits[hits.end++] = HitList::Hit(ray->tfar, hit->primID, hit->geomID, hit->instID[0]);
}

/* gathers hits in a single pass */
void RayHits(const AcousticRay& ray_i, HitList& hits_o, RTCScene scene)
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
