# Code adopted and modified from https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/sensors/noise_models/redwood_depth_noise_model.py
from os import path as osp

import numba
import numpy as np

from gibson2.core.sensors.noise_models.sensor_noise_model import SensorNoiseModel

# Read about the noise model here: http://www.alexteichman.com/octo/clams/
# Original source code: http://redwood-data.org/indoor/data/simdepth.py


@numba.jit(nopython=True, fastmath=True)
def _undistort(x, y, z, model):
    i2 = int((z + 1) / 2)
    i1 = int(i2 - 1)
    a = (z - (i1 * 2.0 + 1.0)) / 2.0
    x = x // 8
    y = y // 6
    f = (1 - a) * model[y, x, min(max(i1, 0), 4)] + a * model[y, x, min(i2, 4)]

    if f < 1e-5:
        return 0
    else:
        return z / f


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _simulate(gt_depth, model, noise_multiplier):
    noisy_depth = np.empty_like(gt_depth)
    H, W = gt_depth.shape

    ymax, xmax = H - 1.0, W - 1.0

    rand_nums = np.random.randn(H, W, 3).astype(np.float32)

    # Parallelize just the outer loop.  This doesn't change the speed
    # noticably but reduces CPU usage compared to two parallel loops
    for j in numba.prange(H):
        for i in range(W):

            y = int(
                min(max(j + rand_nums[j, i, 0] *
                        0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] *
                        0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = _undistort(
                    int(x / xmax * 639.0 + 0.5),
                    int(y / ymax * 479.0 + 0.5),
                    d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth


class RedwoodNoiseModelCPUImpl:
    def __init__(self, model, noise_multiplier):
        self.model = model.reshape(model.shape[0], -1, 4)
        self.noise_multiplier = noise_multiplier

    def simulate(self, gt_depth):
        return _simulate(gt_depth, self.model, self.noise_multiplier)


class RedwoodDepthNoiseModel(SensorNoiseModel):
    def __init__(self, noise_multiplier=1.0):
        self.noise_multiplier = noise_multiplier
        dist = np.load(
            osp.join(osp.dirname(__file__), "data",
                     "redwood-depth-dist-model.npy")
        )
        self._impl = RedwoodNoiseModelCPUImpl(dist, self.noise_multiplier)

    def apply(self, gt_depth):
        return self._impl.simulate(gt_depth)


if __name__ == '__main__':
    import time
    model = RedwoodDepthNoiseModel(noise_multiplier=0.5)
    depth = np.ones((512, 512))
    start = time.time()
    depth_noise = model(depth)
    print(time.time() - start)
    print(depth)
    print(depth_noise)
