import mujoco
import numpy as np
import gym
import itertools
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from time import time
from dynamics_util import change_dynamics, get_dynamics, scale_dynamics

joint_damping_ratios = (0.98, 1.0)
body_mass_ratios = (0.98, 1.0)

class HalfCheetahEnv_RandomDynamics(HalfCheetahEnv):
    def __init__(
        self, 
        xml_file = "half_cheetah.xml", 
        forward_reward_weight = 1, 
        ctrl_cost_weight = 0.1, 
        reset_noise_scale = 0.1, 
        exclude_current_positions_from_observation = True,
    ):
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation)

        self.original_joint_damping = get_dynamics(self.model, 'joint', 'damping')
        self.original_body_mass = get_dynamics(self.model, 'body', 'mass')

        print('Create HalfCheetahEnv_RandomDynamics successfully!') 


    def set_dynamics(self, component_name, dynamics_name, dynamics_dict):
        change_dynamics(self.model, component_name, dynamics_name, dynamics_dict)


    def reset_model(self):
        return super().reset_model()


# if __name__ == '__main__':
#     envs = [HalfCheetahEnv_RandomDynamics() for _ in range(4)]

#     scaled_joint_damping = [scale_dynamics(envs[0].original_joint_damping, r) for r in joint_damping_ratios]
#     scaled_body_mass = [scale_dynamics(envs[0].original_body_mass, r) for r in body_mass_ratios]

#     for i, (sjd, sbm) in enumerate(itertools.product(scaled_joint_damping, scaled_body_mass)):
#         envs[i].set_dynamics('joint', 'damping', sjd)
#         envs[i].set_dynamics('body', 'mass', sbm)

#     for env in envs:
#         env.reset()
#         print(get_dynamics(env.model, 'joint', 'damping'))
#         print(get_dynamics(env.model, 'body', 'mass'))
#         print('--')

if __name__ == '__main__':
    try:
        gym.envs.register(
            id='HalfCheetahRandomDynamics-v0',
            entry_point='halfcheetah:HalfCheetahEnv_RandomDynamics',
            max_episode_steps=1000,
        )
        print('Ok!')
        print(gym.envs.registry.all())
    except:
        print('Failed!')