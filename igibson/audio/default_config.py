# Default System Configuration - can be overwritten by AudioSystem callers!
SAMPLE_RATE = 44100
# Determines number of reverb/reflections probes in the scene. Actual number is num_probes ^ 2
NUM_REVERB_PROBES = 10
OCCLUSION_MULTIPLIER = 1.0


# Default Source Configuration - can be overwritten when calling RegisterSource!
DEFAULT_MIN_FALLOFF_DISTANCE = 0.1
DEFAULT_MAX_FALLOFF_DISTANCE = 10
DEFAULT_SOURCE_GAIN = 1.0
DEFAULT_NEAR_FIELD_GAIN = 1.0
DEFAULT_ROOM_EFFECTS_GAIN = 2.0


# Reverb probe ray-tracing fields
REV_PROBE_SAMPLE_RATE = 48000
REV_PROBE_NUM_RAYS = 200000
REV_PROBE_NUM_RAYS_PER_BATCH = 20000
REV_PROBE_MAX_DEPTH = 3
REV_PROBE_E_THRESHOLD = 1e-6
REV_PROBE_LISTENER_RADIUS = 0.1
REV_PROBE_IMPULSE_N_SAMPLES = 1000000

REV_PROBE_PARAMS = [REV_PROBE_SAMPLE_RATE, \
                    REV_PROBE_NUM_RAYS, \
                    REV_PROBE_NUM_RAYS_PER_BATCH, \
                    REV_PROBE_MAX_DEPTH, \
                    REV_PROBE_E_THRESHOLD, \
                    REV_PROBE_LISTENER_RADIUS, \
                    REV_PROBE_IMPULSE_N_SAMPLES, \
                    ]