from .resnet18 import resnet18, resnet18_batch_stats, BasicBlock

# todo: legacy class names, remove

class aljundi_resnet(resnet18):
  def __init__(self, config):
    super(aljundi_resnet, self).__init__(config)


class aljundi_resnet_batch_stats(resnet18_batch_stats):
  def __init__(self, config):
    super(aljundi_resnet_batch_stats, self).__init__(config)