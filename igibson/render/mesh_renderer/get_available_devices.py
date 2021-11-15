import logging
import os
import subprocess


def get_available_devices():
    """
    Find available devices to run EGL on. It will return the minor numbers,
    The minor number for the device is such that the Nvidia device node
    file for each GPU will have the form /dev/nvidia[minor number]. Avail-
    able only on Linux platform.

    :return: Minor number
    """
    executable_path = os.path.join(os.path.dirname(__file__), "build")
    try:
        num_devices = int(subprocess.check_output(["{}/query_devices".format(executable_path)]))
    except subprocess.CalledProcessError as e:
        return [0], {0: {}}

    FNULL = open(os.devnull, "w")

    device_list = []
    device_info = {}
    for i in range(num_devices):
        try:
            if b"NVIDIA" in subprocess.check_output(["{}/test_device".format(executable_path), str(i)], stderr=FNULL):
                device_list.append(i)
                device_info[i] = {"device_type": "nvidia"}
                logging.info("Device {} is available for rendering".format(i))
            elif b"Intel" in subprocess.check_output(["{}/test_device".format(executable_path), str(i)], stderr=FNULL):
                device_list.append(i)
                device_info[i] = {"device_type": "intel"}
                logging.info("Device {} is available for rendering".format(i))
        except subprocess.CalledProcessError as e:
            logging.info(e)
            logging.info("Device {} is not available for rendering".format(i))
    FNULL.close()

    return device_list, device_info


def get_cuda_device(minor_idx):
    """
    Get the device index to use in pytorch

    The minor number for the device is such that the Nvidia device node
    file for each GPU will have the form /dev/nvidia[minor number]. Avail-
    able only on Linux platform.

    :param minor_idx: Minor index for a GPU
    :return: index to use in torch.cuda.device()
    """

    executable_path = os.path.join(os.path.dirname(__file__), "build")
    try:
        num_devices = int(subprocess.check_output(["{}/query_devices".format(executable_path)]))
    except subprocess.CalledProcessError as e:
        return 0

    for i in range(num_devices):
        output = subprocess.check_output(["nvidia-smi", "-q", "-i", str(i)])
        output_list = output.decode("utf-8").split("\n")
        output_list = [item for item in output_list if "Minor" in item]
        num = int(output_list[0].split(":")[-1])
        if num == minor_idx:
            return i
    return 0


if __name__ == "__main__":
    graphics_devices, device_info = get_available_devices()
    logging.info("Graphics Devices: {}".format(graphics_devices))
    logging.info("Graphics Device Ids: {}".format([get_cuda_device(item) for item in graphics_devices]))
