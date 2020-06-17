from .data import get_data
from .general import get_device, invert_dict
from code.models import *

def get_model_and_data(config):
  config.task_in_dims = {"mnist5k": (28 * 28,), "miniimagenet": (3, 84, 84), "cifar10": (3, 32, 32)}[config.data]
  config.task_out_dims = {"mnist5k": (10,), "miniimagenet": (100,), "cifar10": (10,)}[config.data]

  tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))

  trainloader, testloader, valloader = get_data(config)

  config.class_dict_tasks = invert_dict(trainloader.dataset.task_dict_classes)

  return tasks_model, trainloader, testloader, valloader
