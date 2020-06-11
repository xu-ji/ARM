from .data import get_data
from .general import get_device

def get_model_and_data(config):
  if config.data == "miniimagenet":
    tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))
    trainloader, testloader, valloader = get_data(config)
  else:
    trainloader, testloader, valloader = get_data(config)  # mnist needs data size fields set first
    tasks_model = globals()[config.task_model_type](config).to(get_device(config.cuda))

  return tasks_model, trainloader, testloader, valloader
