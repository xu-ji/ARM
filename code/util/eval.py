from collections import OrderedDict

import numpy as np
import torch

from code.util.general import get_device


# Assumes same number of classes for each task

def evaluate_basic(config, tasks_model, data_loader, t, is_val, last_classes=None,
                   seen_classes=None,
                   compute_forgetting_metric=True, tag=""):
  if is_val:
    prefix = "val"
  else:
    prefix = "test"

  prefix = "%s%s" % (tag, prefix)

  tasks_model.eval()

  acc = 0.
  counts = 0

  num_out = int(np.prod(config.task_out_dims))
  per_label_acc = np.zeros(num_out)
  per_label_counts = np.zeros(num_out)

  for x, y in data_loader:
    x = x.to(get_device(config.cuda))
    y = y.to(get_device(config.cuda))

    with torch.no_grad():
      preds = tasks_model(x)

    preds_flat = torch.argmax(preds, dim=1)

    acc += (preds_flat == y).sum().item()
    counts += y.shape[0]

    for c in range(num_out):
      pos = (y == c)
      per_label_acc[c] += (pos * (preds_flat == c)).sum().item()
      per_label_counts[c] += pos.sum().item()

  acc /= counts
  per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
  per_label_acc /= per_label_counts

  if not hasattr(config, "%s_accs" % prefix):
    setattr(config, "%s_accs" % prefix, OrderedDict())
    setattr(config, "%s_per_label_accs" % prefix, OrderedDict())
    setattr(config, "%s_accs_avg_label" % prefix, OrderedDict())
    setattr(config, "%s_forgetting" % prefix, OrderedDict())

  getattr(config, "%s_accs" % prefix)[t] = acc
  getattr(config, "%s_per_label_accs" % prefix)[t] = per_label_acc
  getattr(config, "%s_accs_avg_label" % prefix)[t] = per_label_acc.mean()

  if compute_forgetting_metric:
    # for all previous (excl latest) tasks, find the maximum drop to current acc, and average
    if len(getattr(config, "%s_accs" % prefix)) >= 3:  # at least 1 previous (non pre training) eval
      assert (last_classes is not None)
      all_per_label_acc_dict = getattr(config, "%s_per_label_accs" % prefix)
      all_per_label_acc = list(all_per_label_acc_dict.values())
      assert ((all_per_label_acc[0] == all_per_label_acc_dict[0]).all())  # sanity
      all_per_label_acc = all_per_label_acc[1:]  # remove first, pre-training

      getattr(config, "%s_forgetting" % prefix)[t] = compute_forgetting(config, t,
                                                                        all_per_label_acc,
                                                                        last_classes, seen_classes)


def compute_forgetting(config, t, all_per_label_acc, last_classes, seen_classes):
  # (a + b)/2 - (a' - b')/2 = (a - a' + b - b')/2
  last_classes = last_classes.cpu().numpy()
  seen_classes = seen_classes.cpu().numpy()

  all_per_label_acc = np.array(all_per_label_acc)
  assert (t % config.eval_freq == 0)
  num_post_0_evals_so_far = int(t / config.eval_freq)

  num_out = int(np.prod(config.task_out_dims))
  assert (all_per_label_acc.shape == (num_post_0_evals_so_far, num_out))

  max_accs = all_per_label_acc[:(num_post_0_evals_so_far - 1), :].max(axis=0)
  assert (max_accs.shape == (num_out,))

  diffs = (max_accs - all_per_label_acc[num_post_0_evals_so_far - 1, :])
  diffs[last_classes] = 0.
  diffs = diffs[seen_classes]

  return diffs.sum() / (seen_classes.shape[0] - last_classes.shape[0])
