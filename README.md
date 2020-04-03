#  iGibson: the Interactive Gibson Environment
**Large Scale Virtualized Interactive Environments for Robot Learning**

iGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. iGibson allows to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets. 


#### Citation
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


Release
=================
This is the repository for iGibson (gibson2) 0.0.1 release. Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. [change log file](misc/CHANGELOG.md).  

The support for our previous version of the environment, [Gibson v1](http://github.com/StanfordVL/GibsonEnv/), will be moved to this repository.

Documentation
=================
The documentation for this repository, including multiple examples, can be found here: [iGibson Environment Documentation](http://svl.stanford.edu/igibson/docs/).

If you want to know more about the simulator, you can find more details in our [RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442) and our (outdated at this point) [technical report](http://svl.stanford.edu/igibson/assets/gibsonv2paper.pdf). Please, consider citing them if you use iGibson or its assets.

More information about the previous version of Gibson can be found in [its project webpage](http://gibsonenv.stanford.edu/).


Dowloading Dataset of 3D Environments
=================
There are several datasets of 3D reconstructed real-world large environments (homes and offices) that you can download and use with iGibson, all of them accessible once you fill in this [form](https://forms.gle/q5Ygkw3ijxD5WC5U8). You will gain access to the ten environments with annotated instances of furniture (chairs, tables, desks, doors, sofas) that can be interacted, and to the original 572 reconstructed 3D environments without interactions. 

You can also download a [fully annotated environment](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz) where the interactive objects replaced the original spatial arrangenment of a real house ([original 3D model](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) also available). 