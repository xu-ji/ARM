import os.path as osp
import pickle

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy


def get_device(cuda):
  if cuda:
    return torch.device("cuda:0")
  else:
    return torch.device("cpu")


def store(config):
  # todo anonymization
  data_path = config.data_path
  out_root = config.out_root
  out_dir = config.out_dir

  config.data_path = ""
  config.out_root = ""
  config.out_dir = ""

  with open(osp.join(out_dir, "config.pickle"),
            'wb') as outfile:
    pickle.dump(config, outfile)

  with open(osp.join(out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)
    
  config.data_path = data_path
  config.out_root = out_root
  config.out_dir = out_dir


def get_avg_grads(model):
  total = None
  count = 0
  for p in model.parameters():
    sz = np.prod(p.grad.shape)
    grad_sum = p.grad.abs().sum()
    if total is None:
      total = grad_sum
      count = sz
    else:
      total += grad_sum
      count += sz

  return total / float(count)


def record_and_check(config, name, val, t):
  if hasattr(val, "item"):
    val = val.item()

  record(config, name, val, t)
  if not np.isfinite(val):
    print("value (probably loss) not finite, aborting:")
    print(name)
    print("t %d" % t)
    print(val)
    store(config)  # to store nan values
    exit(1)


def check(config, val, t):
  if not np.isfinite(val):
    print("value (probably loss) not finite, aborting:")
    print("t %d" % t)
    print(val)
    store(config)  # to store nan values
    exit(1)


def record(config, val_name, val, t, abs=False):
  if not hasattr(config, val_name):
    setattr(config, val_name, OrderedDict())

  storage = getattr(config, val_name)

  if "torch" in str(val.__class__):
    if abs:
      val = torch.abs(val)

    if val.dtype == torch.int64:
      assert (val.shape == torch.Size([]))
    else:
      val = torch.mean(val)

    storage[t] = val.item()  # either scalar loss or vector of grads
  else:
    if abs:
      val = abs(val)  # default abs

    storage[t] = val

  if not hasattr(config, "record_names"):
    config.record_names = []

  if not val_name in config.record_names:
    config.record_names.append(val_name)


def get_gpu_mem(nvsmi, gpu_ind):
  mem_stats = nvsmi.DeviceQuery('memory.free, memory.total')["gpu"][gpu_ind]["fb_memory_usage"]
  return mem_stats["total"] - mem_stats["free"]


def get_task_in_dims(config):
  return config.task_in_dims


def get_task_out_dims(config):
  return config.task_out_dims


def render_graphs(config):
  if not hasattr(config, "record_names"):
    return

  training_val_names = config.record_names
  fig0, axarr0 = plt.subplots(max(len(training_val_names), 2), sharex=False,
                              figsize=(8, len(training_val_names) * 4))

  for i, val_name in enumerate(training_val_names):
    if hasattr(config, val_name):
      storage = getattr(config, val_name)
      axarr0[i].clear()
      axarr0[i].plot(list(storage.keys()), list(storage.values()))  # ordereddict

      axarr0[i].set_title(val_name)

  fig0.suptitle("Model %d" % (config.model_ind), fontsize=8)
  fig0.savefig(osp.join(config.out_dir, "plots_0.png"))

  if hasattr(config, "val_accs"):
    fig1, axarr1 = plt.subplots(4, sharex=False, figsize=(8, 4 * 4))

    for pi, prefix in enumerate(["val", "test"]):
      accs_name = "%s_accs" % prefix
      axarr1[pi * 2].clear()
      axarr1[pi * 2].plot(list(getattr(config, accs_name).keys()),
                          list(getattr(config, accs_name).values()))  # ordereddict
      axarr1[pi * 2].set_title(accs_name)

      per_label_accs_name = "%s_per_label_accs" % prefix
      axarr1[pi * 2 + 1].clear()
      per_label_accs_t = getattr(config, per_label_accs_name).keys()
      per_label_accs_np = np.array(list(getattr(config, per_label_accs_name).values()))
      for c in range(int(np.prod(get_task_out_dims(config)))):
        axarr1[pi * 2 + 1].plot(list(per_label_accs_t), list(per_label_accs_np[:, c]), label=str(c))
      axarr1[pi * 2 + 1].legend()
      axarr1[pi * 2 + 1].set_title(per_label_accs_name)

    fig1.suptitle("Model %d" % (config.model_ind), fontsize=8)
    fig1.savefig(osp.join(config.out_dir, "plots_1.png"))

  # render predictions, if exist
  if hasattr(config, "aux_y_probs"):
    # time along x axis, classes along y axis
    fig2, ax2 = plt.subplots(1, figsize=(16, 8))  # width, height

    num_t = len(config.aux_y_probs)
    num_classes = int(np.prod(get_task_out_dims(config)))

    aux_y_probs = list(config.aux_y_probs.values())
    aux_y_probs = [aux_y_prob.numpy() for aux_y_prob in aux_y_probs]
    aux_y_probs = np.array(aux_y_probs)

    # print(aux_y_probs.shape)
    assert (aux_y_probs.shape == (len(config.aux_y_probs), int(np.prod(get_task_out_dims(config)))))

    aux_y_probs = aux_y_probs.transpose()  # now num classes, time
    min_val = aux_y_probs.min()
    max_val = aux_y_probs.max()

    # tile along y axis to make each class fatter. Should be same number of pixels altogether as
    # current t / 2
    scale = int(0.5 * float(num_t) / num_classes)
    if scale > 1:
      aux_y_probs = np.repeat(aux_y_probs, scale, axis=0)
      ax2.set_yticks(np.arange(num_classes) * scale)
      ax2.set_yticklabels(np.arange(num_classes))

    num_thousands = int(num_t / 1000)
    ax2.set_xticks(np.arange(num_thousands) * 1000)
    ax2.set_xticklabels(np.arange(num_thousands) * 1000 + list(config.aux_y_probs.keys())[0])

    im = ax2.imshow(aux_y_probs)
    fig2.colorbar(im, ax=ax2)
    # ax2.colorbar()

    fig2.suptitle("Model %d, max %f min %f" % (config.model_ind, max_val, min_val), fontsize=8)
    fig2.savefig(osp.join(config.out_dir, "plots_2.png"))

  plt.close("all")


def trim_config(config, next_t):
  # trim everything down to next_t numbers
  # we are starting at top of loop *before* eval step

  for val_name in config.record_names:
    storage = getattr(config, val_name)
    if isinstance(storage, list):
      assert (len(storage) >= (next_t))
      setattr(config, val_name, storage[:next_t])
    else:
      assert (isinstance(storage, OrderedDict))
      storage_copy = deepcopy(storage)
      for k, v in storage.items():
        if k >= next_t:
          del storage_copy[k]
      setattr(config, val_name, storage_copy)

  for prefix in ["val", "test"]:
    accs_storage = getattr(config, "%s_accs" % prefix)
    per_label_accs_storage = getattr(config, "%s_per_label_accs" % prefix)

    if isinstance(accs_storage, list):
      assert (isinstance(per_label_accs_storage, list))
      assert (len(accs_storage) >= (next_t) and len(per_label_accs_storage) >= (
      next_t))  # at least next_t stored

      setattr(config, "%s_accs" % prefix, accs_storage[:next_t])
      setattr(config, "%s_per_label_accs" % prefix, per_label_accs_storage[:next_t])
    else:
      assert (
      isinstance(accs_storage, OrderedDict) and isinstance(per_label_accs_storage, OrderedDict))
      for dn, d in [("accs", accs_storage), ("per_label_accs", per_label_accs_storage)]:
        d_copy = deepcopy(d)
        for k, v in d.items():
          if k >= next_t:
            del d_copy[k]
        setattr(config, dn, d_copy)

  # deal with window
  if config.long_window:
    # find index of first historical t for update >= next_t
    # set config.next_update_old_model_t = that t
    # trim history behind it, backing onto nest_t

    next_t_i = None
    for i, update_t in enumerate(config.next_update_old_model_t_history):
      if update_t > next_t:
        next_t_i = i
        break

    # there must be a t in update history that is greater than next_t
    # unless config.next_update_old_model_t >= next_t and we stopped before it was added to history
    # in which case we don't need to trim any update history

    if next_t_i is None:
      print("no trimming:")
      print(("config.next_update_old_model_t", config.next_update_old_model_t))
      print(("next_t", next_t))
      assert (config.next_update_old_model_t >= next_t)
    else:
      config.next_update_old_model_t = config.next_update_old_model_t_history[next_t_i]
      config.next_update_old_model_t_history = config.next_update_old_model_t_history[:next_t_i]


def sum_seq(seq):
  res = None
  for elem in seq:
    if res is None:
      res = elem
    else:
      res += elem
  return res


def np_rand_seed():  # fixed classes shuffling
  return 111


def reproc_settings(config):
  np.random.seed(0)  # set separately when shuffling data too
  if config.specific_torch_seed:
    torch.manual_seed(config.torch_seed)
  else:
    torch.manual_seed(config.model_ind)  # allow initialisations different per model

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def copy_parameter_values(from_model, to_model):
  to_params = list(to_model.named_parameters())
  assert (isinstance(to_params[0], tuple) and len(to_params[0]) == 2)

  to_params = dict(to_params)

  for n, p in from_model.named_parameters():
    to_params[n].data.copy_(p.data)  # not clone


def make_valid_from_train(dataset, cut):
  tr_ds, val_ds = [], []
  for task_ds in dataset:
    x_t, y_t = task_ds

    # shuffle before splitting
    perm = torch.randperm(len(x_t))
    x_t, y_t = x_t[perm], y_t[perm]

    split = int(len(x_t) * cut)
    x_tr, y_tr = x_t[:split], y_t[:split]
    x_val, y_val = x_t[split:], y_t[split:]

    tr_ds += [(x_tr, y_tr)]
    val_ds += [(x_val, y_val)]

  return tr_ds, val_ds
