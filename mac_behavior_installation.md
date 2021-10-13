## Instructions for installing behavior on mac

1. `pip install cmake`
2. `git clone git@github.com:Learning-and-Intelligent-Systems/iGibson.git --recursive`
3. `git clone git@github.com:Learning-and-Intelligent-Systems/bddl.git`
4. Fill out https://forms.gle/GXAacjpnotKkM2An7 -- ask Nishanth for the signed form if you don't have it already. After submitting the form, make sure that you open each of the links in a new tab, otherwise you will be navigated away and will have to fill out the form again. I downloaded all of them but I'm going to ignore the third link for now. The second and third downloads are very large (40GB) and will take a long time.
5. `mkdir iGibson/igibson/data`
6. Put each of the three downloads in that new data dir.
7. unzip behavior_data_bundle.zip (about 27 GB unzipped)
8. `pip install -e ./iGibson` (this will also take a long time)
9. `pip install -e ./bddl`
10. `python -m igibson.utils.assets_utils --download_assets`

## Hack around an issue
Issue: https://github.com/StanfordVL/iGibson/issues/122

To see if this issue is still present, try this:
`python iGibson/igibson/examples/behavior/behavior_env_metrics.py`

If that results in this error: `ERROR:root:Optimized renderer is not supported on Mac` then you will need a hack.

In `iGibson/igibson/envs/env_base.py`, in the call to `MeshRendererSettings`, change `optimized=True` to `optimized=False`.

## Check if it worked
This should now run with a GUI:
`python iGibson/igibson/examples/behavior/behavior_env_metrics.py -m pbgui`
or a different GUI:
`python iGibson/igibson/examples/behavior/behavior_env_metrics.py -m iggui`
