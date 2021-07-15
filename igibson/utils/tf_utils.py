import tensorflow as tf    # pylint: ignore-module


# TF related
def make_gpu_session(num_gpu=1):
    if num_gpu == 1:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()
    return sess
