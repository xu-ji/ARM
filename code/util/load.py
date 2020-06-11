from .data import get_data
from .general import get_device
from code.models import *

# list(range(2262, 2262+5)) 
old_code_models = list(range(4821, 4821+5)) + list(range(4557, 4557+5))

def get_model_and_data(config):
  if config.model_ind in old_code_models: # reproduce random sampling of old code: remove in final version
    tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))
    trainloader, testloader, valloader = get_data(config)
  else:
    trainloader, testloader, valloader = get_data(config)
    tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))

  return tasks_model, trainloader, testloader, valloader
