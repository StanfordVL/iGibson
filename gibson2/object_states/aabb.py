from gibson2.object_states.object_state_base import CachingEnabledObjectState
from gibson2.external.pybullet_tools.utils import aabb_union, get_aabb, get_all_links
import numpy as np
import pybullet as p


class AABB(CachingEnabledObjectState):
    def _compute_value(self):
        body_id = self.obj.get_body_id()
        all_links = get_all_links(body_id)
        if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
            all_links.remove(-1)
        aabbs = [get_aabb(body_id, link=link)
                 for link in all_links]
        aabb_low, aabb_hi = aabb_union(aabbs)

        return np.array(aabb_low), np.array(aabb_hi)

    def set_value(self, new_value):
        raise NotImplementedError("AABB state currently does not support setting.")
