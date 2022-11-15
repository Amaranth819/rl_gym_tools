import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def predict(self, obs, deterministic = False):
        raise NotImplementedError


    def update(self, batch):
        raise NotImplementedError