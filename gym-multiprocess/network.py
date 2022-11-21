import torch
import torch.nn as nn


def create_mlp(in_dim, out_dim, hidden_dim_list = [256], act_fn = None, last_act_fn = None):
    layer_dim_list = [in_dim] + hidden_dim_list
    layers = []

    for i in range(len(layer_dim_list) - 1):
        layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
        if act_fn is not None:
            layers.append(act_fn())

    layers.append(nn.Linear(layer_dim_list[-1], out_dim))
    if last_act_fn is not None:
        layers.append(last_act_fn())

    return nn.Sequential(*layers)


def create_distribution(p, mu_act_fn = None, min_logstd = -2, max_logstd = 0.5):
    mu, logstd = torch.chunk(p, 2, -1)
    if mu_act_fn is not None:
        mu = mu_act_fn(mu)
    std = torch.clip(logstd, min_logstd, max_logstd).exp()
    dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
    return dist, (mu, std)


class GaussianPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_list = [256], act_fn = None, mu_act_fn = torch.tanh) -> None:
        super().__init__()

        self.main = create_mlp(in_dim, out_dim * 2, hidden_dim_list, act_fn, None)
        self.mu_act_fn = mu_act_fn


    def forward(self, x):
        return create_distribution(self.main(x), self.mu_act_fn)



if __name__ == '__main__':
    print(create_mlp(4, 2, [64], act_fn = nn.ReLU, last_act_fn = nn.Tanh))