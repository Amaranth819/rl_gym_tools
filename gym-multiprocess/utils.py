import torch
import numpy as np
from gym import spaces

def get_device(device_str):
    assert device_str in ['auto', 'cpu', 'cuda']
    if device_str == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(device_str)


def tensor_to_np(tensor : torch.Tensor):
    return tensor.detach().cpu().numpy()


def np_to_tensor(arr : np.array, device : torch.device):
    return torch.from_numpy(arr).to(device)


def get_space_shape(space):
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        return (space.n)
    else:
        raise NotImplementedError