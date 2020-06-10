import os
import os.path
import sys

import numpy as np
from PIL import Image

from code.util.general import np_rand_seed
from code.util.check_data import *

import pickle

from torchvision.datasets.vision import VisionDataset

# Reference: https://github.com/optimass/Maximally_Interfered_Retrieval/blob/master/data.py
# We use 1 dataloader rather than one per task

__all__ = ["cifar10", "cifar10val"]

class cifar(VisionDataset):
  train_pc = 0.95

  def __init__(self, root, num_classes, train=True, transform=None, target_transform=None, non_stat=False,
               two_classes_per_block=False, is_val=False, shuffle_classes=False, num_iterations=None):
    print("initialising cifar%d, is val: %s..." % (num_classes, is_val))
    super(cifar, self).__init__(root, transform=transform, target_transform=target_transform)

    self.num_classes = num_classes

    self.train = train
    self.is_val = is_val
    self.non_stat = non_stat
    self.two_classes_per_block = two_classes_per_block
    self.shuffle_classes = shuffle_classes

    self.num_iterations = num_iterations
    assert(num_iterations is not None)

    if self.is_val:
      assert(self.train)

    if self.train:
      downloaded_list = self.train_list
    else:
      downloaded_list = self.test_list

    self.data = []
    self.targets = []

    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
      file_path = os.path.join(self.root, self.base_folder, file_name)
      with open(file_path, 'rb') as f:
        if sys.version_info[0] == 2:
          entry = pickle.load(f)
        else:
          entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        if 'labels' in entry:
          self.targets.extend(entry['labels'])
        else:
          self.targets.extend(entry['fine_labels'])

    self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
    self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    sz = self.data.shape[0]
    assert(len(self.targets) == sz)

    # Split train from val deterministically -------------------------------------------------------

    if self.train:
      if not self.is_val:
        per_class_sz = int((sz / self.num_classes) * cifar.train_pc)
        data_inds = range(sz) # from start
      else:
        per_class_sz = int((sz / self.num_classes) * (1 - cifar.train_pc))
        data_inds = reversed(range(sz)) # from back

      class_counts = [0 for _ in range(self.num_classes)]
      chosen_inds = []
      for i in data_inds:
        c = self.targets[i]
        if class_counts[c] < per_class_sz:
          chosen_inds.append(i)
          class_counts[c] += 1
      assert(len(chosen_inds) == per_class_sz * self.num_classes)

    else:
      chosen_inds = list(range(sz))

    # Deterministic shuffle
    np.random.seed(np_rand_seed())
    chosen_inds = np.array(chosen_inds)
    np.random.shuffle(chosen_inds)
    self.data = [self.data[i] for i in chosen_inds]
    self.targets = [self.targets[i] for i in chosen_inds]

    # Rearrange into contiguous format for non-stationary training ---------------------------------
    self.task_dict_classes = {}

    if non_stat:
      # organise data and targets by targets
      per_class = [[] for _ in range(self.num_classes)]
      for i, label in enumerate(self.targets):
        per_class[label].append(self.data[i])

      new_data = []
      new_targets = []
      if not two_classes_per_block: # classes contiguous
        for c in range(self.num_classes):
          new_data += per_class[c]
          new_targets += [c] * len(per_class[c])
          self.task_dict_classes[c] = [c]
      else:
        classes = np.arange(self.num_classes)
        assert(not self.shuffle_classes) # sanity
        if self.shuffle_classes:
          np.random.seed(np_rand_seed())
          np.random.shuffle(classes)

        num_tasks = int(np.ceil(self.num_classes / 2.))
        print("two_classes_per_block: num tasks %d" % num_tasks)
        for t in range(num_tasks):
          inds = [t * 2, t * 2 + 1]
          if (t * 2 + 1 >= self.num_classes): # odd number of tasks
            assert(t == num_tasks - 1 and num_tasks % 2 == 1)
            inds = [t * 2]

          self.task_dict_classes[t] = [classes[i] for i in inds]

          t_data = []
          t_targets = []
          for i in inds:
            c = classes[i]
            t_data += per_class[c]
            t_targets += [c] * len(per_class[c])

          order = np.arange(len(t_data))
          np.random.shuffle(order)
          new_data += [t_data[i] for i in order]
          new_targets += [t_targets[i] for i in order]

      self.data = new_data
      self.targets = new_targets

    class_lengths = []
    targets_np = np.array(self.targets)
    for c in range(num_classes):
      class_lengths.append((targets_np == c).sum())

    print("... finished initialising cifar train %s val %s non stat %s classes %d two_classes_per_block %s iterations %d shuffle %s" %
          (self.train, self.is_val, self.non_stat, self.num_classes, self.two_classes_per_block, self.num_iterations, self.shuffle_classes))

    self.orig_len = len(self.data)
    self.actual_len = self.orig_len * self.num_iterations

    if self.non_stat: # we need to care about looping over in task order
      assert(self.orig_len % self.num_classes == 0)

      self.orig_samples_per_task = int(self.orig_len / self.num_classes)

      if self.two_classes_per_block:
        self.orig_samples_per_task *= 2

      self.actual_samples_per_task = self.orig_samples_per_task * self.num_iterations
      print("orig samples per task: %d, actual samples per task: %d, orig len %d actual len %d" % (self.orig_samples_per_task, self.actual_samples_per_task, self.orig_len, self.actual_len))

  def __getitem__(self, index):
    assert(index < self.actual_len)

    if not self.non_stat:
      index = index % self.orig_len # looping over stationary data is arbitrary
    else:
      task_i, actual_offset = divmod(index, self.actual_samples_per_task)
      _, orig_offset = divmod(actual_offset, self.orig_samples_per_task)
      index = task_i * self.orig_samples_per_task + orig_offset

    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return self.actual_len


class cifar10(cifar):
  base_folder = 'cifar-10-batches-py'

  train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
  ]

  test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
  ]
  def __init__(self, root, train=True, transform=None, target_transform=None, non_stat=None,
               two_classes_per_block=False, is_val=False, num_iterations=None):
    assert(non_stat is not None)
    if not train:
      assert(num_iterations == 1)
    super(cifar10, self).__init__(root, num_classes=10, train=train, transform=transform, target_transform=target_transform,
                                   non_stat=non_stat, two_classes_per_block=two_classes_per_block, is_val=is_val, shuffle_classes=False,
                                  num_iterations=num_iterations)


class cifar10val(cifar10):
  # 5% validation set
  def __init__(self, root, transform=None, non_stat=False, two_classes_per_block=False):
    super(cifar10val, self).__init__(root, train=True, transform=transform,
                                     non_stat=non_stat, two_classes_per_block=two_classes_per_block,
                                     is_val=True, num_iterations=1)