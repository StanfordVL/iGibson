from gibson2.core.simulator_flex import Simulator_Flex
import numpy as np

config_path = '/scr/yxjin/nvidia/sw/devrel/libdev/flex/dev/rbd/demo/gym/cfg/gibson.yaml'
bin_path = '/scr/yxjin/nvidia/sw/devrel/libdev/flex/dev/rbd/bin'

simulator = Simulator_Flex(config_path, bin_path)

while True:
	simulator.step(np.array([[0,0,0,0,0,0,0,0]]))
