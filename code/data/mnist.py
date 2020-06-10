from collections import defaultdict

import numpy as np
import torch
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset

from code.util.general import make_valid_from_train


# Reference: https://github.com/optimass/Maximally_Interfered_Retrieval/blob/master/data.py
# We use 1 dataloader rather than one per task

class mnist5k(VisionDataset):
  train_val_pc = 0.95

  def __init__(self, root, data_type=None, non_stat=False, num_iterations=None):
    super(mnist5k, self).__init__(root, transform=None, target_transform=None)

    self.data_type = data_type
    self.non_stat = non_stat
    self.classes_per_task = 2
    self.num_classes = 10
    self.orig_train_samples_per_class = 500

    self.num_iterations = num_iterations
    assert (num_iterations is not None)

    # Load data ------------------------------------------------------------------------------------
    # splits are deterministic

    # follows https://github.com/optimass/Maximally_Interfered_Retrieval/

    train = datasets.MNIST(root, train=True, download=False)
    test = datasets.MNIST(root, train=False, download=False)

    train_x, train_y = train.data, train.targets  # 60000, 28, 28; 60000
    test_x, test_y = test.data, test.targets

    # sort by label
    train_ds, test_ds = [], []  # doesn't really matter for test_ds because of batchnorm tracking
    #  stats
    task_i = 0
    current_train, current_test = None, None
    self.task_dict_classes = defaultdict(list)
    for i in range(self.num_classes):
      self.task_dict_classes[task_i].append(i)
      train_i = train_y == i
      test_i = test_y == i

      if current_train is None:
        current_train, current_test = (train_x[train_i], train_y[train_i]), (
        test_x[test_i], test_y[test_i])
      else:
        current_train = (torch.cat((current_train[0], train_x[train_i]), dim=0),
                         torch.cat((current_train[1], train_y[train_i]), dim=0))
        current_test = (torch.cat((current_test[0], test_x[test_i]), dim=0),
                        torch.cat((current_test[1], test_y[test_i]), dim=0))

      if i % self.classes_per_task == (self.classes_per_task - 1):
        train_ds += [current_train]
        test_ds += [current_test]
        current_train, current_test = None, None
        task_i += 1

    # separate validation set (randomised)
    train_ds, val_ds = make_valid_from_train(train_ds, cut=mnist5k.train_val_pc)

    # flatten into single list, and truncate training data into 500 per class
    data_summary = {"train": train_ds, "val": val_ds, "test": test_ds}[self.data_type]
    self.data = []  # list of tensors
    self.targets = []
    counts_per_class = torch.zeros(self.num_classes, dtype=torch.int)
    task_lengths = []
    for task_ds in data_summary:
      assert (len(task_ds[1]) == len(task_ds[0]))

      num_samples_task = 0
      for i in range(len(task_ds[1])):
        target = task_ds[1][i]
        if self.data_type == "train" and counts_per_class[
          target] == self.orig_train_samples_per_class:
          continue
        else:
          self.data.append(task_ds[0][i])
          self.targets.append(target)
          counts_per_class[target] += 1
          num_samples_task += 1

      task_lengths.append(num_samples_task)

    print(self.task_dict_classes)

    # if stationary, shuffle
    if not self.non_stat:
      perm = np.random.permutation(len(self.data))
      self.data, self.targets = [self.data[perm_i] for perm_i in perm], [self.targets[perm_i] for
                                                                         perm_i in perm]

    self.orig_len = len(self.data)
    self.actual_len = self.orig_len * self.num_iterations

    if self.non_stat:  # we need to care about looping over in task order
      assert (self.orig_len % self.num_classes == 0)
      self.orig_samples_per_task = int(
        self.orig_len / self.num_classes) * self.classes_per_task  # equally split among tasks
      self.actual_samples_per_task = self.orig_samples_per_task * self.num_iterations

      # sanity
      if self.data_type == "train": assert (self.orig_samples_per_task == 1000)

      print("orig samples per task: %d, actual samples per task: %d" % (
      self.orig_samples_per_task, self.actual_samples_per_task))

  def __len__(self):
    return self.actual_len

  def __getitem__(self, index):
    assert (index < self.actual_len)

    if not self.non_stat:
      index = index % self.orig_len  # looping over stationary data is arbitrary
    else:
      task_i, actual_offset = divmod(index, self.actual_samples_per_task)
      _, orig_offset = divmod(actual_offset, self.orig_samples_per_task)
      index = task_i * self.orig_samples_per_task + orig_offset

    sample, target = self.data[index], self.targets[index]
    sample = sample.view(-1).float() / 255.  # flatten and turn from uint8 (255) -> [0., 1.]

    assert (self.transform is None)
    assert (self.target_transform is None)

    return sample, target

  def __len__(self):
    return self.actual_len
