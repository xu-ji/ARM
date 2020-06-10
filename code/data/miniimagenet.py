import os.path
import os.path
from collections import defaultdict

import numpy as np
from PIL import Image
from code.util.check_data import *
from torchvision.datasets.vision import VisionDataset

from code.util.general import make_valid_from_train


# Reference: https://github.com/optimass/Maximally_Interfered_Retrieval/blob/master/data.py
# We use 1 dataloader rather than one per task

# Can download from https://www.dropbox.com/s/ed1s1dgei9kxd2p/mini-imagenet.zip?dl=0

def get_data(setname, root_csv, root_images):
  csv_path = os.path.join(root_csv, setname + '.csv')
  lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

  data = []
  label = []
  lb = -1

  wnids = []

  for l in lines:
    name, wnid = l.split(',')
    path = os.path.join(root_images, name)
    if wnid not in wnids:
      wnids.append(wnid)
      lb += 1
    data.append(path)
    label.append(lb)

  return data, label


class MiniImagenetDatasetFolder(VisionDataset):
  train_val_pc = 0.95
  train_test_pc = 0.8

  def __init__(self, root, data_type=None, transform=None, target_transform=None,
               non_stat=False, classes_per_task=None, num_iterations=None):
    super(MiniImagenetDatasetFolder, self).__init__(root, transform=transform,
                                                    target_transform=target_transform)

    self.data_type = data_type
    self.non_stat = non_stat
    self.classes_per_task = classes_per_task

    self.num_classes = 100

    self.num_iterations = num_iterations
    assert (num_iterations is not None)

    # Load data ------------------------------------------------------------------------------------
    # splits are deterministic

    images_path = os.path.join(root, "images")
    train_data, train_label = get_data("train", root,
                                       images_path)  # zero indexed labels for all calls
    valid_data, valid_label = get_data("val", root, images_path)
    test_data, test_label = get_data("test", root, images_path)

    train_amt = np.unique(train_label).shape[0]
    valid_amt = np.unique(valid_label).shape[0]
    test_amt = np.unique(test_label).shape[0]

    assert (train_amt + valid_amt + test_amt == self.num_classes)
    valid_label = [x + train_amt for x in valid_label]
    test_label = [x + train_amt + valid_amt for x in test_label]

    all_data = np.array(train_data + valid_data + test_data)  # np array of strings!
    all_label = np.array(train_label + valid_label + test_label)

    train_ds, test_ds = [], []
    current_train, current_test = None, None

    cat = lambda x, y: np.concatenate((x, y), axis=0)

    self.task_dict_classes = defaultdict(list)
    task_i = 0
    for i in range(self.num_classes):
      self.task_dict_classes[task_i].append(i)

      class_indices = np.argwhere(all_label == i).reshape(-1)
      class_data = all_data[class_indices]
      class_label = all_label[class_indices]
      split = int(MiniImagenetDatasetFolder.train_test_pc * class_data.shape[0])  # train/test

      data_train, data_test = class_data[:split], class_data[split:]
      label_train, label_test = class_label[:split], class_label[split:]

      if current_train is None:
        current_train, current_test = (data_train, label_train), (
        data_test, label_test)  # multiple samples here
      else:
        current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
        current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)

      if i % self.classes_per_task == (self.classes_per_task - 1):
        train_ds += [current_train]
        test_ds += [current_test]
        current_train, current_test = None, None
        task_i += 1

    train_ds, val_ds = make_valid_from_train(train_ds,
                                             cut=MiniImagenetDatasetFolder.train_val_pc)  # uses
    # random split, but seed set in main script

    # now we have list of list of (path, label), one list per task
    # pick the right source, flatten into one list and load images

    data_summary = {"train": train_ds, "val": val_ds, "test": test_ds}[self.data_type]

    self.data = []
    self.targets = []
    task_lengths = []
    for task_ds in data_summary:
      num_samples_task = len(task_ds[0])
      assert (len(task_ds[1]) == num_samples_task)
      task_lengths.append(num_samples_task)
      for i in range(num_samples_task):
        img_path = task_ds[0][i]
        label = task_ds[1][i]
        self.data.append(img_path)
        self.targets.append(label)

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
      if self.data_type == "train":
        assert (self.orig_samples_per_task == (int(
          600 * MiniImagenetDatasetFolder.train_test_pc * MiniImagenetDatasetFolder.train_val_pc)
                                               * self.classes_per_task))

      if self.data_type == "val":
        assert (self.orig_samples_per_task == (int(600 * MiniImagenetDatasetFolder.train_test_pc * (
        1. - MiniImagenetDatasetFolder.train_val_pc)) * self.classes_per_task))

      if self.data_type == "test":
        assert (self.orig_samples_per_task == (
        int(600 * (1. - MiniImagenetDatasetFolder.train_test_pc)) * self.classes_per_task))

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

    sample_path, target = self.data[index], self.targets[index]

    with open(sample_path, "rb") as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target

  def __len__(self):
    return self.actual_len


class miniimagenet(MiniImagenetDatasetFolder):
  def __init__(self, root, data_type, transform=None, target_transform=None, non_stat=None,
               classes_per_task=None, num_iterations=None):
    assert (non_stat is not None)
    if data_type == "val" or data_type == "test":
      assert (num_iterations == 1)

    super(miniimagenet, self).__init__(root,
                                       data_type=data_type,
                                       transform=transform, target_transform=target_transform,
                                       non_stat=non_stat, classes_per_task=classes_per_task,
                                       num_iterations=num_iterations)


class miniimagenetval(miniimagenet):
  def __init__(self, root, transform=None, non_stat=False, classes_per_task=None):
    super(miniimagenetval, self).__init__(root, data_type="val",
                                          non_stat=non_stat, classes_per_task=classes_per_task,
                                          transform=transform, num_iterations=1)
