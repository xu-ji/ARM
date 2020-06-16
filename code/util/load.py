from .data import get_data
from .general import get_device
from code.models import *

def get_model_and_data(config):
  tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))
  trainloader, testloader, valloader = get_data(config)

  return tasks_model, trainloader, testloader, valloader
