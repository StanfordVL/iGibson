#  iGibson: the Interactive Gibson Environment

<img src="https://github.com/fxia22/iGibson/blob/readme/docs/images/igibsonlogo.png" width="600"> <img src="https://github.com/fxia22/iGibson/blob/readme/docs/images/igibson.gif" width="250"> 

### Large Scale Interactive Simulation Environments for Robot Learning

iGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. iGibson allows researchers to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and manipulation tasks such as opening doors, picking and placing objects, or searching in cabinets.

### Latest Updates
[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@article{xia2020interactive,
         title={Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments},
         author={Xia, Fei and Shen, William B and Li, Chengshu and Kasimbeg, Priya and Tchapmi, Micael Edmond and Toshev, Alexander and Mart{\'\i}n-Mart{\'\i}n, Roberto and Savarese, Silvio},
         journal={IEEE Robotics and Automation Letters},
         volume={5},
         number={2},
         pages={713--720},
         year={2020},
         publisher={IEEE}
}
```


### Release
This is the repository for iGibson (gibson2) 0.0.4 release. Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. The support for our previous version of the environment, [Gibson v1](http://github.com/StanfordVL/GibsonEnv/), will be moved to this repository.

### Documentation
The documentation for this repository can be found here: [iGibson Environment Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson), [our RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442) and [our (outdated) technical report](http://svl.stanford.edu/igibson/assets/gibsonv2paper.pdf).

### Dowloading Dataset of 3D Environments
There are several datasets of 3D reconstructed large real-world environments (homes and offices) that you can download and use with iGibson. All of them will be accessible once you fill in this [form](https://forms.gle/q5Ygkw3ijxD5WC5U8).

You will have access to ten environments with annotated instances of furniture (chairs, tables, desks, doors, sofas) that can be interacted with, and to the original 572 reconstructed 3D environments without annotated objects from [Gibson v1](http://github.com/StanfordVL/GibsonEnv/).

You will also have access to a [fully annotated environment: Rs_interactive](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz) where close to 200 articulated objects are placed in their original locations of a real house and ready for interaction. ([The original environment: Rs](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) is also available). More info can be found in the [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

