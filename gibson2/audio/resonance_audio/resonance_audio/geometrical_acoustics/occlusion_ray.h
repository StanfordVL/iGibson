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

#define MAX_TOTAL_HITS 16*1024

/* extended ray structure that gathers all hits along the ray */
struct HitList
{
    HitList()
        : begin(0), end(0) {}

    /* Hit structure that defines complete order over hits */
    struct Hit
    {
        Hit() {}

        Hit(float t, unsigned int primID = 0xFFFFFFFF, unsigned int geomID = 0xFFFFFFFF, unsigned int instID = 0xFFFFFFFF)
            : t(t), primID(primID), geomID(geomID), instID(instID) {}

        /* lexicographical order (t,instID,geomID,primID) */
        __forceinline friend bool operator < (const Hit& a, const Hit& b)
        {
            if (a.t == b.t) {
                if (a.instID == b.instID) {
                    if (a.geomID == b.geomID) return a.primID < b.primID;
                    else                      return a.geomID < b.geomID;
                }
                else return a.instID < b.instID;
            }
            return a.t < b.t;
        }

        __forceinline friend bool operator == (const Hit& a, const Hit& b) {
            return a.t == b.t && a.primID == b.primID && a.geomID == b.geomID && a.instID == b.instID;
        }

        __forceinline friend bool operator <= (const Hit& a, const Hit& b)
        {
            if (a == b) return true;
            else return a < b;
        }

        __forceinline friend bool operator != (const Hit& a, const Hit& b) {
            return !(a == b);
        }

        friend std::ostream& operator<<(std::ostream& cout, const Hit& hit) {
            return cout << "Hit { opaque = " << hit.opaque << ", t = " << hit.t << ", instID = " << hit.instID << ", geomID = " << hit.geomID << ", primID = " << hit.primID << " }";
        }

    public:
        float t;
        unsigned int primID;
        unsigned int geomID;
        unsigned int instID;
    };

    /* return number of gathered hits */
    unsigned int size() const {
        return end - begin;
    }

    /* returns the last hit */
    const Hit& last() const {
        assert(end);
        return hits[end - 1];
    }

public:
    unsigned int begin;   // begin of hit list
    unsigned int end;     // end of hit list
    Hit hits[MAX_TOTAL_HITS];   // array to store all found hits to
};

/* we store the Hit list inside the intersection context to access it from the filter functions */
struct IntersectContext
{
    IntersectContext(HitList& hits): hits(hits) {}
    RTCIntersectContext context;
    HitList& hits;
};



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
void RayHits(const Ray& ray_i, HitList& hits_o, RTCScene scene)
{
    /* trace ray to gather all hits */
    Ray ray = ray_i;
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
