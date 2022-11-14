import mujoco
import numpy as np
import copy
from collections import defaultdict


def get_dynamics(model, component_name, dynamics_name):
    if component_name == 'joint':
        component_type = mujoco.mjtObj.mjOBJ_JOINT
        num_components = model.njnt
    elif component_name == 'body':
        component_type = mujoco.mjtObj.mjOBJ_BODY
        num_components = model.nbody
    elif component_name == 'geom':
        component_type = mujoco.mjtObj.mjOBJ_GEOM
        num_components = model.ngeom
    else:
        raise ValueError

    dynamics_dict = {}

    for i in range(num_components):
        component_i_name = mujoco.mj_id2name(model, component_type, i)
        component_i_obj = getattr(model, component_name)(component_i_name)
        dynamics_dict[component_i_name] = np.copy(getattr(component_i_obj, dynamics_name))

    return dynamics_dict


def scale_dynamics(dynamics_dict, ratio):
    new_dynamics_dict = copy.deepcopy(dynamics_dict)
    for _, val in new_dynamics_dict.items():
        val *= ratio
    return new_dynamics_dict


def change_dynamics(model, component_name, dynamics_name, dynamics_dict = {}):
    '''
        Warning: Calling env.reset() doesn't reset the dynamics parameters to the default value! Need to call change_dynamics() taking the original parameters as input.
    '''
    for n, val in dynamics_dict.items():
        obj = getattr(model, component_name)(n)
        setattr(obj, dynamics_name, val)


# class RandomDynamicsEnv(object):
#     def __init__(self) -> None:
#         pass