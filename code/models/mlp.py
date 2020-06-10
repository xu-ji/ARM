import numpy as np
import torch.nn as nn


# Consistent with https://github.com/optimass/Maximally_Interfered_Retrieval

class mlp(nn.Module):
  def __init__(self, config):
    super(mlp, self).__init__()

    self.nf = 400

    self.input_size = np.prod(config.task_in_dims)
    self.hidden = nn.Sequential(nn.Linear(self.input_size, self.nf),
                                nn.ReLU(True),
                                nn.Linear(self.nf, self.nf),
                                nn.ReLU(True))

    self.linear = nn.Linear(self.nf, np.prod(config.task_out_dims))

  def forward(self, x):
    x = x.view(-1, self.input_size)
    x = self.hidden(x)
    return self.linear(x)
