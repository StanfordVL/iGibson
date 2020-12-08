#  iGibson: A Simulation Environment to Train AI Agents in Large Realistic Scenes

<img src="./docs/images/igibsonlogo.png" width="500"> <img src="./docs/images/igibson.gif" width="250"> 

### Similation of Large Scale Interactive Scenes for Robot Learning

iGibson is a simulation environment providing fast visual rendering and physics simulation based on Bullet. iGibson is equipped with fifteen fully interactive high quality scenes, hundreds of large 3D scenes reconstructed from real homes and offices, and compatibility with datasets like CubiCasa5K and 3D-Front, providing 8000+ additional interactive scenes. Some of the features of iGibson include domain randomization, integration with motion planners and easy-to-use tools to collect human demonstrations. With these scenes and features, iGibson allows researchers to train and evaluate robotic agents that use visual signals to solve navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets.

### Latest Updates

[12/1/2020] Major update to iGibson to reach iGibson v1.0, for details please refer to our [arxiv preprint](https://arxiv.org/abs/2012.02924). 

- Release of iGibson dataset that includes 15 fully interactive scenes and 500+ object models annotated with materials and physical attributes.
- Compatibility to import CubiCasa and 3D-Front scene layouts for more than 8000 extra interactive scenes!
- New features in iGibson: Physically based rendering, 1-beam and 16-beam LiDAR, domain randomization, motion planning integration, tools to collect human demos and more!
- Code refactoring, better class structure and cleanup. 

[05/14/2020] Added dynamic light support :flashlight:

[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@article{shenigibson,
  title={iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes},
  author={Shen*, Bokui and Xia*, Fei and Li*, Chengshu and Mart{\'i}n-Mart{\'i}n*, Roberto and Fan, Linxi and Wang, Guanzhi and Buch, Shyamal and D’Arpino, Claudia and Srivastava, Sanjana and Tchapmi, Lyne P and  Vainio, Kent and Fei-Fei, Li and Savarese, Silvio},
  journal={arXiv preprint arXiv:2012.02924},
  year={2020}
}
```

### Documentation
The documentation for iGibson can be found here: [iGibson Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download instructions), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson),  [our
 updated arxiv preprint](https://arxiv.org/abs/2012.02924) and [our previous RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442).

### Dowloading Dataset of 3D Environments

With iGibson v1.0 release, you will have access to 15 fully interactive scenes (100+ rooms) that can be
 used in simulation. As a highlight, here
 are the features we support. We also include 500+ object models.  

- Scenes are the
 result of converting 3D reconstructions of real homes into fully interactive simulatable environments.
- Each scene corresponds to one floor of a real-world home.
The scenes are annotated with bounding box location and size of different objects, mostly furniture, e.g. cabinets, doors, stoves, tables, chairs, beds, showers, toilets, sinks...
- Scenes include layout information (occupancy, semantics)
- Each scene's lighting effect is designed manually, and the texture of the building elements (walls, floors, ceilings
) is baked offline with high-performant ray-tracing
- Scenes are defined in iGSDF (iGibson Scene Definition Format), an extension of URDF, and shapes are OBJ files with
 associated materials
 
For instructions to install iGibson and download dataset, you can visit [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

There are other datasets we link to iGibson. We include support to use CubiCasa5K and 3DFront scenes, adding up more than 8000 extra interactive scenes to use in iGibson! Check our documentation on how to use those.

We also maintain compatibility with datasets of 3D reconstructed large real-world environments (homes and offices) that you can download and use with iGibson, for example from our previous simulator, Gibson. All of them will be accessible once you fill in this <a href="https://forms.gle/36TW9uVpjrE1Mkf9A" target="_blank">[form]</a>.

### Contributing
This is the github repository for iGibson (pip package `gibson2`) 1.0 release. Bug reports, suggestions for improvement, as
 well as community
 developments are encouraged and appreciated. Please, consider creating an issue or sending us an email. 

The support for our previous version of the environment, Gibson, can be found in the [following repository
 ](http://github.com/StanfordVL/GibsonEnv/).
