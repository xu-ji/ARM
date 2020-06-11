from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms

from code.data import *


def get_data(config):
  config.task_in_dims, config.task_out_dims = None, None
  dataloaders, datasets = globals()["get_%s_loaders" % config.data](config)
  num_train_samples = len(datasets[0])
  assert (num_train_samples % config.tasks_train_batch_sz) == 0
  config.batches_per_epoch = int(num_train_samples / config.tasks_train_batch_sz)

  assert ((config.task_in_dims is not None) and (config.task_out_dims is not None))
  print("length of training d, test d, val d: %d %d %d, batches_per_epoch %d" %
        (num_train_samples, len(datasets[1]), len(datasets[2]), config.batches_per_epoch))
  assert (config.store_model_freq == config.batches_per_epoch)  # once per epoch

  return dataloaders


def get_cifar10_loaders(config):
  assert (config.data == "cifar10")
  assert (config.task_in_dims == (3, 32, 32))
  assert (config.task_out_dims == (10,))
  two_classes_per_block = (config.classes_per_task == 2)

  train_fns = [transforms.ToTensor()]
  transform_train = transforms.Compose(train_fns)

  cifar10_training = cifar10(root=config.data_path, train=True, transform=transform_train,
                             non_stat=(not config.stationary),
                             two_classes_per_block=two_classes_per_block,
                             num_iterations=config.num_iterations)

  if not config.stationary:
    cifar10_training_loader = DataLoader(cifar10_training,
                                         sampler=SequentialSampler(cifar10_training), shuffle=False,
                                         batch_size=config.tasks_train_batch_sz)
  else:
    cifar10_training_loader = DataLoader(cifar10_training, shuffle=True,
                                         batch_size=config.tasks_train_batch_sz)

  test_fns = [transforms.ToTensor()]
  transform_test = transforms.Compose(test_fns)

  evals_non_stat = False  # does not make a difference, test time behavior independent of rest of
  #  batch

  cifar10_val = cifar10val(root=config.data_path, transform=transform_test,
                           non_stat=evals_non_stat, two_classes_per_block=two_classes_per_block)
  cifar10_val_loader = DataLoader(cifar10_val, shuffle=False, batch_size=config.tasks_eval_batch_sz)

  cifar10_test = cifar10(root=config.data_path, train=False, transform=transform_test,
                         num_iterations=1,
                         non_stat=evals_non_stat, two_classes_per_block=two_classes_per_block)
  cifar10_test_loader = DataLoader(cifar10_test, shuffle=False,
                                   batch_size=config.tasks_eval_batch_sz)

  return (cifar10_training_loader, cifar10_test_loader, cifar10_val_loader), \
         (cifar10_training, cifar10_test, cifar10_val)


def get_miniimagenet_loaders(config):
  assert (config.data == "miniimagenet")
  assert (config.task_in_dims == (3, 84, 84))
  assert (config.task_out_dims == (100,))

  train_fns = [
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
  ]

  transform_train = transforms.Compose(train_fns)

  miniimagenet_training = miniimagenet(root=config.data_path, data_type="train",
                                       transform=transform_train,
                                       non_stat=(not config.stationary),
                                       classes_per_task=config.classes_per_task,
                                       num_iterations=config.num_iterations)

  if not config.stationary:
    miniimagenet_training_loader = DataLoader(miniimagenet_training,
                                              sampler=SequentialSampler(miniimagenet_training),
                                              shuffle=False, batch_size=config.tasks_train_batch_sz)
  else:
    miniimagenet_training_loader = DataLoader(miniimagenet_training, shuffle=True,
                                              batch_size=config.tasks_train_batch_sz)

  test_fns = [
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
  ]

  transform_test = transforms.Compose(test_fns)

  evals_non_stat = False

  miniimagenet_val = miniimagenetval(root=config.data_path, transform=transform_test,
                                     non_stat=evals_non_stat,
                                     classes_per_task=config.classes_per_task)
  miniimagenet_val_loader = DataLoader(miniimagenet_val, shuffle=False,
                                       batch_size=config.tasks_eval_batch_sz)

  miniimagenet_test = miniimagenet(root=config.data_path, data_type="test",
                                   transform=transform_test, num_iterations=1,
                                   non_stat=evals_non_stat,
                                   classes_per_task=config.classes_per_task)
  miniimagenet_test_loader = DataLoader(miniimagenet_test, shuffle=False,
                                        batch_size=config.tasks_eval_batch_sz)

  return (miniimagenet_training_loader, miniimagenet_test_loader, miniimagenet_val_loader), \
         (miniimagenet_training, miniimagenet_test, miniimagenet_val)


def get_mnist5k_loaders(config):
  assert (config.data == "mnist5k")
  assert (config.task_in_dims == (28 * 28,))
  assert (config.task_out_dims == (10,))

  mnist5k_training = mnist5k(root=config.data_path, data_type="train",
                             non_stat=(not config.stationary), num_iterations=config.num_iterations,
                             classes_per_task=config.classes_per_task)

  if not config.stationary:
    mnist5k_training_loader = DataLoader(mnist5k_training,
                                         sampler=SequentialSampler(mnist5k_training),
                                         shuffle=False, batch_size=config.tasks_train_batch_sz)
  else:
    mnist5k_training_loader = DataLoader(mnist5k_training, shuffle=True,
                                         batch_size=config.tasks_train_batch_sz)

  evals_non_stat = False

  mnist5k_val = mnist5k(root=config.data_path, data_type="val", non_stat=evals_non_stat,
                        num_iterations=1, classes_per_task=config.classes_per_task)
  mnist5k_val_loader = DataLoader(mnist5k_val, shuffle=False, batch_size=config.tasks_eval_batch_sz)

  mnist5k_test = mnist5k(root=config.data_path, data_type="test", non_stat=evals_non_stat,
                         num_iterations=1, classes_per_task=config.classes_per_task)
  mnist5k_test_loader = DataLoader(mnist5k_test, shuffle=False,
                                   batch_size=config.tasks_eval_batch_sz)

  return (mnist5k_training_loader, mnist5k_test_loader, mnist5k_val_loader), \
         (mnist5k_training, mnist5k_test, mnist5k_val)
