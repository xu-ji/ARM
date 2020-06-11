from .resnet18 import resnet18, resnet18_batch_stats

# todo: legacy class names are now empty wrappers, remove for final version

class cifar100_resnet34(resnet18):
  def __init__(self, config):
    super(cifar100_resnet34, self).__init__(config)