#  iGibson: the Interactive Gibson Environment

### Large Scale Interactive Simulation Environments for Robot Learning

iGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. iGibson allows researchers to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and manipulation tasks such as opening doors, picking and placing objects, or searching in cabinets.


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

### Code Release
The GitHub repository of iGibson can be found here: [iGibson GitHub Repo](https://github.com/StanfordVL/iGibson). Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. The support for our previous version of the environment, [Gibson v1](http://github.com/StanfordVL/GibsonEnv/), will be moved there.


### Documentation
The documentation for iGibson can be found here: [iGibson Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download instructions), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson),  [our
 updated arxiv preprint](https://arxiv.org/abs/2012.02924) and [our previous RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442).

### Dowloading the Dataset of 3D Scenes

With iGibson v1.0 release, you will have access to 15 fully interactive scenes (100+ rooms) that can be
 used in simulation. As a highlight, here
 are the features we support. We also include 500+ object models.  

- Scenes are the
 result of converting 3D reconstructions of real homes into fully interactive simulatable models.
- Each scene corresponds to one floor of a real-world home.
The scenes are annotated with bounding box location and size of different objects, mostly furniture, e.g. cabinets, doors, stoves, tables, chairs, beds, showers, toilets, sinks...
- Scenes include layout information (occupancy, semantics)
- Each scene's lighting effect is designed manually, and the texture of the building elements (walls, floors, ceilings
) is baked offline with high-performant ray-tracing
- Scenes are defined in iGSDF (iGibson Scene Definition Format), an extension of URDF, and shapes are OBJ files with
 associated materials
 
For instructions to install iGibson and download dataset, you can visit [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

There are other datasets we link to iGibson. We include support to use CubiCasa5K and 3DFront scenes, adding up more than 8000 extra interactive scenes to use in iGibson! Check our documentation on how to use those.

We also maintain compatibility with datasets of 3D reconstructed large real-world scenes (homes and offices) that you can download and use with iGibson, for example from our previous simulator, Gibson. All of them will be accessible once you fill in this <a href="https://forms.gle/36TW9uVpjrE1Mkf9A" target="_blank">[form]</a>.
