from .resnet18 import resnet18, resnet18_batch_stats

# todo: legacy class names are empty wrappers, remove for final version

class cifar10_resnet18(resnet18):
  def __init__(self, config):
    super(cifar10_resnet18, self).__init__(config)


class cifar10_resnet18_feat20(resnet18):
  def __init__(self, config):
    super(cifar10_resnet18_feat20, self).__init__(config)


class cifar10_resnet18_feat20_batch_stats(resnet18_batch_stats):
  def __init__(self, config):
    super(cifar10_resnet18_feat20_batch_stats, self).__init__(config)

