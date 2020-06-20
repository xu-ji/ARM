import torch
import torch.nn as nn

# Consistent with https://github.com/optimass/Maximally_Interfered_Retrieval


class BasicBlock(nn.Module):
  """Basic Block for resnet 18 and resnet 34
  """

  # BasicBlock and BottleNeck block
  # have different output size
  # we use class attribute expansion
  # to distinct
  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True, batchnorm_mom=None,
               batchnorm_dont_track=False):
    super().__init__()

    # residual function
    seq = [
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
    if use_batchnorm:
      if batchnorm_mom is None:
        seq += [nn.BatchNorm2d(out_channels, track_running_stats=(not batchnorm_dont_track))]
      else:
        seq += [nn.BatchNorm2d(out_channels, momentum=batchnorm_mom,
                               track_running_stats=(not batchnorm_dont_track))]

    seq += [
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1,
                bias=False),
    ]
    if use_batchnorm:
      if batchnorm_mom is None:
        seq += [nn.BatchNorm2d(out_channels * BasicBlock.expansion,
                               track_running_stats=(not batchnorm_dont_track))]
      else:
        seq += [nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=batchnorm_mom,
                               track_running_stats=(not batchnorm_dont_track))]

    self.residual_function = nn.Sequential(*seq)

    # shortcut
    self.shortcut = nn.Sequential()

    # the shortcut output dimension is not the same with residual function
    # use 1*1 convolution to match the dimension
    if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
      shortcut_seq = [
        nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                  bias=False)]
      if use_batchnorm:
        if batchnorm_mom is None:
          shortcut_seq += [nn.BatchNorm2d(out_channels * BasicBlock.expansion,
                                          track_running_stats=(not batchnorm_dont_track))]
        else:
          shortcut_seq += [
            nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=batchnorm_mom,
                           track_running_stats=(not batchnorm_dont_track))]

      self.shortcut = nn.Sequential(*shortcut_seq)

  def forward(self, x):
    return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
  def __init__(self, block, num_block, num_classes=100, use_batchnorm=True, batchnorm_mom=None,
               batchnorm_dont_track=False, in_channels=64, init=False, batchnorm_init=False,
               linear_sz=None):
    super().__init__()

    self.num_classes = num_classes
    self.use_batchnorm = use_batchnorm
    self.batchnorm_mom = batchnorm_mom

    self.in_channels = in_channels

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(in_channels, momentum=self.batchnorm_mom,
                     track_running_stats=(not batchnorm_dont_track)),
      nn.ReLU(inplace=True))
    # we use a different inputsize than the original paper
    # so conv2_x's stride is 1

    # make layer resets in_channels to be out channels
    self.conv2_x = self._make_layer(block, in_channels, num_block[0], 1,
                                    batchnorm_mom=self.batchnorm_mom,
                                    batchnorm_dont_track=batchnorm_dont_track)
    self.conv3_x = self._make_layer(block, in_channels * 2, num_block[1], 2,
                                    batchnorm_mom=self.batchnorm_mom,
                                    batchnorm_dont_track=batchnorm_dont_track)
    self.conv4_x = self._make_layer(block, in_channels * 4, num_block[2], 2,
                                    batchnorm_mom=self.batchnorm_mom,
                                    batchnorm_dont_track=batchnorm_dont_track)
    self.conv5_x = self._make_layer(block, in_channels * 8, num_block[3], 2,
                                    batchnorm_mom=self.batchnorm_mom,
                                    batchnorm_dont_track=batchnorm_dont_track)

    if self.num_classes == 10: # cifar10, legacy code
      self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    self.fc1 = nn.Linear(linear_sz, self.num_classes)

    """
    if init:  # else default, which is uniform
      print("calling _initialise")
      self._initialise()

    if batchnorm_init:  # else default, which is all 1s since 1.2
      print("calling _batchnorm_initialise")
      self._batchnorm_initialise()
    """

  def _initialise(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _batchnorm_initialise(self):
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):  # pre 1.2 default init
        nn.init.uniform_(m.weight, a=0.0, b=1.0)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, out_channels, num_blocks, stride, batchnorm_mom=None,
                  batchnorm_dont_track=False):
    """make resnet layers(by layer i didnt mean this 'layer' was the 
    same as a neuron netowork layer, ex. conv layer), one layer may 
    contain more than one residual block 

    Args:
        block: block type, basic block or bottle neck block
        out_channels: output depth channel number of this layer
        num_blocks: how many blocks per layer
        stride: the stride of the first block of this layer

    Return:
        return a resnet layer
    """

    # we have num_block blocks per layer, the first block
    # could be 1 or 2, other blocks would always be 1
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride,
                          use_batchnorm=self.use_batchnorm, batchnorm_mom=batchnorm_mom,
                          batchnorm_dont_track=batchnorm_dont_track))
      self.in_channels = out_channels * block.expansion

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)

    pool1 = self.avg_pool(x)
    pool2 = nn.functional.avg_pool2d(x, 4)
    print("diff: %.8f, %.8f" % ((pool1.mean() - pool2.mean()).item(), (pool1.max() - pool2.max()).item()))

    if self.num_classes == 10:
      x = self.avg_pool(x)
    else:
      x = nn.functional.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)

    return self.fc1(x)


def _batch_stats_hook(b, input):
  if isinstance(input, tuple):
    assert (len(input) == 1)
    input = input[0]

  assert (len(input.shape) == 4)

  stored_mean = b.running_mean
  stored_var = b.running_var

  curr_mean = input.mean(dim=(0, 2, 3))
  curr_var = input.var(dim=(0, 2, 3))

  assert (stored_mean.shape == (input.shape[1],))
  assert (stored_var.shape == (input.shape[1],))
  assert (curr_mean.shape == (input.shape[1],))
  assert (curr_var.shape == (input.shape[1],))

  b.batch_stats_loss = torch.norm(curr_mean - stored_mean, p=2) + torch.norm(curr_var - stored_var,
                                                                             p=2)
  assert (b.batch_stats_loss.shape == torch.Size([]))  # scalar


class resnet18(ResNet):
  def __init__(self, config):
    # 4 pool is only different to avgpool for large images. Newer resnet code uses avgpool but Aljundi uses 4 pool.
    if config.data == "miniimagenet":
      num_classes = 100
      linear_sz = 160 * 2 * 2
    elif config.data == "cifar10":
      num_classes = 10
      linear_sz = 160 * 1 * 1

    super(resnet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                                   in_channels=20,
                                   use_batchnorm=True,
                                   batchnorm_dont_track=False,
                                   linear_sz=linear_sz)


class resnet18_batch_stats(resnet18):
  def __init__(self, config):
    super(resnet18_batch_stats, self).__init__(config)

    self._compute_batch_stats_loss = False

  def compute_batch_stats_loss(self):
    self._compute_batch_stats_loss = True

    self.num_batchnorms = 0
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.register_forward_pre_hook(_batch_stats_hook)
        m.batch_stats_loss = None
        self.num_batchnorms += 1

  def forward(self, x):
    # return loss for batch stats

    x = self.conv1(x)
    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)

    if self.num_classes == 10:
      x = self.avg_pool(x)
    else:
      x = nn.functional.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)

    x = self.fc1(x)

    if self._compute_batch_stats_loss:
      batch_stats_losses = []
      for m in self.modules():
        if isinstance(m, nn.BatchNorm2d):
          assert (m.batch_stats_loss is not None)
          batch_stats_losses.append(m.batch_stats_loss)
          m.batch_stats_loss = None
      assert (len(batch_stats_losses) == self.num_batchnorms)
      return x, torch.mean(torch.stack(batch_stats_losses))
    else:
      return x