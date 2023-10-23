#  iGibson: A Simulation Environment to train Robots in Large Realistic Interactive Scenes

<img src="./docs/images/igibsonlogo.png" width="500"> <img src="./docs/images/igibson.gif" width="250"> 

iGibson is a simulation environment providing fast visual rendering and physics simulation based on Bullet. iGibson is equipped with fifteen fully interactive high quality scenes, hundreds of large 3D scenes reconstructed from real homes and offices, and compatibility with datasets like CubiCasa5K and 3D-Front, providing 8000+ additional interactive scenes. Some of the features of iGibson include domain randomization, integration with motion planners and easy-to-use tools to collect human demonstrations. With these scenes and features, iGibson allows researchers to train and evaluate robotic agents that use visual signals to solve navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets.

### Latest Updates
[8/9/2021] Major update to iGibson to reach iGibson 2.0, for details please refer to our [arxiv preprint](https://arxiv.org/abs/2108.03272). 

-  iGibson 2.0 supports object states, including temperature, wetness level, cleanliness level, and toggled and sliced states, necessary to cover a wider range of tasks. 
- iGibson 2.0 implements a set of predicate logic functions that map the simulator states to logic states like Cooked or Soaked.
- iGibson 2.0 includes a virtual reality (VR) interface to immerse humans in its scenes to collect demonstrations. 


[12/1/2020] Major update to iGibson to reach iGibson 1.0, for details please refer to our [arxiv preprint](https://arxiv.org/abs/2012.02924). 

- Release of iGibson dataset that includes 15 fully interactive scenes and 500+ object models annotated with materials and physical attributes on top of [existing 3D articulated models](https://cs.stanford.edu/~kaichun/partnet/).
- Compatibility to import [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) and [3D-Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) scene descriptions leading to more than 8000 extra interactive scenes!
- New features in iGibson: Physically based rendering, 1-beam and 16-beam LiDAR, domain randomization, motion planning integration, tools to collect human demos and more!
- Code refactoring, better class structure and cleanup. 

[05/14/2020] Added dynamic light support :flashlight:

[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@inproceedings{li2022igibson,
  title = 	 {iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks},
  author =       {Li, Chengshu and Xia, Fei and Mart\'in-Mart\'in, Roberto and Lingelbach, Michael and Srivastava, Sanjana and Shen, Bokui and Vainio, Kent Elliott and Gokmen, Cem and Dharan, Gokul and Jain, Tanish and Kurenkov, Andrey and Liu, Karen and Gweon, Hyowon and Wu, Jiajun and Fei-Fei, Li and Savarese, Silvio},
  booktitle = 	 {Proceedings of the 5th Conference on Robot Learning},
  pages = 	 {455--465},
  year = 	 {2022},
  editor = 	 {Faust, Aleksandra and Hsu, David and Neumann, Gerhard},
  volume = 	 {164},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {08--11 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v164/li22b/li22b.pdf},
  url = 	 {https://proceedings.mlr.press/v164/li22b.html},
} 
```

```
@inproceedings{shen2021igibson,
      title={iGibson 1.0: a Simulation Environment for Interactive Tasks in Large Realistic Scenes}, 
      author={Bokui Shen and Fei Xia and Chengshu Li and Roberto Mart\'in-Mart\'in and Linxi Fan and Guanzhi Wang and Claudia P\'erez-D'Arpino and Shyamal Buch and Sanjana Srivastava and Lyne P. Tchapmi and Micael E. Tchapmi and Kent Vainio and Josiah Wong and Li Fei-Fei and Silvio Savarese},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      pages={accepted},
      organization={IEEE}
}
```

### Documentation
The documentation for iGibson can be found here: [iGibson Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download instructions), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson),  [iGibson 2.0 arxiv preprint](https://arxiv.org/abs/2108.03272) and [iGibson 1.0 arxiv preprint](https://arxiv.org/abs/2012.02924).

### Dowloading the Dataset of 3D Scenes

For instructions to install iGibson and download dataset, you can visit [installation guide](http://svl.stanford.edu/igibson/docs/installation.html) and [dataset download guide](http://svl.stanford.edu/igibson/docs/dataset.html).

There are other datasets we link to iGibson. We include support to use CubiCasa5K and 3DFront scenes, adding up more than 10000 extra interactive scenes to use in iGibson! Check our [documentation](https://github.com/StanfordVL/iGibson/tree/master/igibson/utils/data_utils/ext_scene) on how to use those.

We also maintain compatibility with datasets of 3D reconstructed large real-world scenes (homes and offices) that you can download and use with iGibson. For Gibson Dataset and Stanford 2D-3D-Semantics Dataset, please fill out this [form](https://forms.gle/36TW9uVpjrE1Mkf9A). For Matterport3D Dataset, please fill in this [form](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf) and send it to [matterport3d@googlegroups.com](mailto:matterport3d@googlegroups.com). Please put "use with iGibson simulator" in your email. Check our [dataset download guide](http://svl.stanford.edu/igibson/docs/dataset.html) for more details.

### Using iGibson with VR
If you want to use iGibson VR interface, please visit the [VR guide (TBA)].


### Contributing
This is the github repository for iGibson (pip package `igibson`) 2.0 release. (For iGibson 1.0, please use `1.0` branch.) Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. Please, consider creating an issue or sending us an email. 

The support for our previous version of the environment, Gibson, can be found in the [following repository](http://github.com/StanfordVL/GibsonEnv/).

### Acknowledgments

iGibson uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations under permissable copyleft licenses), iGibson would not be possible. We thanks these authors for their efforts!

- Syoyo Fujita: [tinyobjloader](https://github.com/syoyo/tinyobjloader)
- Erwin Coumans: [egl_example](https://github.com/erwincoumans/egl_example)
- Caelan Garrett: [ss-pybullet](https://github.com/caelan/ss-pybullet)
- Sean Barrett: [stb](https://github.com/nothings/stb)
- Michał Siejak: [PBR](https://github.com/Nadrin/PBR)
- [CryptoPP](https://www.cryptopp.com/)
