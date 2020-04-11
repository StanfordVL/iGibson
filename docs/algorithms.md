# Algorithms

### Overview

iGibson can be used with any algorithms (from optimal control to model-free reinforcement leanring) that accommodate OpenAI gym interface. Feel free to use your favorite algorithms and deep learning frameworks.

### Examples

#### TF-Agents

In this example, we show an environment wrapper of [TF-Agents](https://github.com/tensorflow/agents) for iGibson and an example training code for [SAC agent](https://arxiv.org/abs/1801.01290). The code can be found in [our fork of TF-Agents](https://github.com/StanfordVL/agents/): [agents/blob/gibson_sim2real/tf_agents/environments/suite_gibson.py](https://github.com/StanfordVL/agents/blob/gibson_sim2real/tf_agents/environments/suite_gibson.py) and [agents/blob/gibson_sim2real/tf_agents/agents/sac/examples/v1/train_single_env.sh](https://github.com/StanfordVL/agents/blob/gibson_sim2real/tf_agents/agents/sac/examples/v1/train_single_env.sh).

```python
def load(config_file,
         model_id=None,
         env_type='gibson',
         sim2real_track='static',
         env_mode='headless',
         action_timestep=1.0 / 5.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         random_position=False,
         random_height=False,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    if env_type == 'gibson':
        if random_position:
            env = NavigateRandomEnv(config_file=config_file,
                                    mode=env_mode,
                                    action_timestep=action_timestep,
                                    physics_timestep=physics_timestep,
                                    device_idx=device_idx,
                                    random_height=random_height)
        else:
            env = NavigateEnv(config_file=config_file,
                              mode=env_mode,
                              action_timestep=action_timestep,
                              physics_timestep=physics_timestep,
                              device_idx=device_idx)
    elif env_type == 'gibson_sim2real':
        env = NavigateRandomEnvSim2Real(config_file=config_file,
                                        mode=env_mode,
                                        action_timestep=action_timestep,
                                        physics_timestep=physics_timestep,
                                        device_idx=device_idx,
                                        track=sim2real_track)
    else:
        assert False, 'unknown env_type: {}'.format(env_type)

    discount = env.discount_factor
    max_episode_steps = env.max_step

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
```
