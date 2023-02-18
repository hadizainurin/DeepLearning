import copy
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import math

class SEUnit(nn.Module):
  """Squeeze-Excitation Unit
  paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper
  """
  def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
      super(SEUnit, self).__init__()
      hidden_dim = in_channel // reduction_ratio
      self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
      self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
      self.act1 = act1()
      self.act2 = act2()

     # Programatically
  def forward(self, x):
  # How to make y
    y = self.avg_pool(x)
    y = self.fc(y)
    y = self.act1(y)
    y = self.fc2(y)
    y = self.act2(y)
    return x*y
    #return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


# Basic Structure
class MBConv(nn.Module):
  def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SEUnitit(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


  def forward(self, x):
      if self.identity:
          return x + self.conv(x)
      else:
          return self.conv(x)