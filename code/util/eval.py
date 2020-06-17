from collections import OrderedDict, defaultdict

import numpy as np
import torch

from code.util.general import get_device


# accs_data: average over all data (old metric)
# per_label_accs: acc per class
# per_task_accs: acc per task
# accs: average over all seen tasks (after last task, is same as Chaudry def.)
# forgetting: average over all seen tasks

def evaluate_basic(config, tasks_model, data_loader, t, is_val, last_classes=None,
                   seen_classes=None,
                   compute_forgetting_metric=True, tag=""):
  if is_val:
    prefix = "val"
  else:
    prefix = "test"

  prefix = "%s%s" % (tag, prefix)

  tasks_model.eval()

  acc_data = 0.
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

    acc_data += (preds_flat == y).sum().item()
    counts += y.shape[0]

    for c in range(num_out):
      pos = (y == c)
      per_label_acc[c] += (pos * (preds_flat == c)).sum().item()
      per_label_counts[c] += pos.sum().item()

  # acc over all data
  acc_data /= counts

  # acc per class
  per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
  per_label_acc /= per_label_counts

  # acc per seen task and avg
  acc = None
  if hasattr(config, "%s_accs_data" % prefix):  # not pre training
    last_classes = last_classes.cpu().numpy()
    seen_classes = seen_classes.cpu().numpy()

    per_task_acc = defaultdict(list)
    for c in seen_classes:  # seen tasks only
      per_task_acc[config.class_dict_tasks[c]].append(per_label_acc[c])

    #seen_tasks = [task_i for task_i in per_task_acc if len(per_task_acc[task_i]) > 0]
    acc = 0.
    for task_i in per_task_acc:
      assert (len(per_task_acc[task_i]) == config.classes_per_task)
      per_task_acc[task_i] = np.array(per_task_acc[task_i]).mean()
      acc += per_task_acc[task_i]
    acc /= len(per_task_acc)

    print(per_task_acc)
    assert(False)

  if not hasattr(config, "%s_accs" % prefix):
    setattr(config, "%s_accs_data" % prefix, OrderedDict())
    setattr(config, "%s_per_label_accs" % prefix, OrderedDict())

    setattr(config, "%s_per_task_accs" % prefix, OrderedDict())
    setattr(config, "%s_accs" % prefix, OrderedDict())

    setattr(config, "%s_forgetting" % prefix, OrderedDict())

  getattr(config, "%s_accs_data" % prefix)[t] = acc_data
  getattr(config, "%s_per_label_accs" % prefix)[t] = per_label_acc
  if acc is not None:
    getattr(config, "%s_per_task_accs" % prefix)[t] = per_task_acc
    getattr(config, "%s_accs" % prefix)[t] = acc

  # for all previous (excl latest) tasks, find the maximum drop to curr acc
  if compute_forgetting_metric:
    if len(getattr(config, "%s_accs_data" % prefix)) >= 3:  # at least 1 previous (non pre training) eval
      assert (last_classes is not None)
      getattr(config, "%s_forgetting" % prefix)[t] = compute_forgetting(config, t,
                                                                        getattr(config,
                                                                                "%s_per_task_accs" % prefix),
                                                                        last_classes)


def compute_forgetting(config, t, per_task_accs, last_classes):
  # per_task_acc is not equal length per timestep so can't array

  assert (t % config.eval_freq == 0)

  # find task that just finished
  last_task_i = None
  for c in last_classes:
    task_i = config.class_dict_tasks[c]
    if last_task_i is None:
      last_task_i = task_i
    else:
      assert (last_task_i == task_i)

  forgetting_per_task = {}
  for task_i in range(last_task_i):  # excl last (tasks are numbered chronologically)
    best_acc = None
    for past_t in per_task_accs:
      if past_t == 0: continue
      if past_t == t: continue

      if best_acc is None or per_task_accs[past_t][task_i] > best_acc:
        best_acc = per_task_accs[past_t][task_i]
    assert (best_acc is not None)

    forgetting_per_task[task_i] = best_acc - per_task_accs[t][task_i]

  assert (len(forgetting_per_task) == last_task_i)
  return np.array(list(forgetting_per_task.values())).mean()
