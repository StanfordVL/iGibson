import subprocess
import os


def get_available_devices():
    executable_path = os.path.join(os.path.dirname(__file__), 'build')

    num_devices = int(subprocess.check_output(["{}/query_devices".format(executable_path)]))

    available_devices = []
    for i in range(num_devices):
        try:
            if b"NVIDIA" in subprocess.check_output(
                ["{}/test_device".format(executable_path),
                 str(i)]):
                available_devices.append(i)
        except subprocess.CalledProcessError as e:
            print(e)
    return (available_devices)


def get_cuda_device(idx):
    output = subprocess.check_output(["nvidia-smi", '-q', '-i', str(idx)])
    output_list = output.decode("utf-8").split('\n')
    output_list = [item for item in output_list if 'Minor' in item]
    num = int(output_list[0].split(':')[-1])
    return num


if __name__ == '__main__':
    print(get_available_devices())
    print(get_cuda_device(0))
