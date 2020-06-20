import argparse
import os
import sys
from datetime import datetime

import torch.optim as optim

from code.models import *
from code.util.eval import evaluate_basic
from code.util.general import *
from code.util.load import *

# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------

orig_config = argparse.ArgumentParser(allow_abbrev=False)

orig_config.add_argument("--model_ind_start", type=int, required=True)

orig_config.add_argument("--num_runs", type=int, required=True)

orig_config.add_argument("--out_root", type=str, required=True)

# Data and model

orig_config.add_argument("--data", type=str, default="cifar100")

orig_config.add_argument("--data_path", type=str, required=True)

orig_config.add_argument("--stationary", default=False, action="store_true")

orig_config.add_argument("--classes_per_task", type=int, required=True)

orig_config.add_argument("--max_t", type=int, required=True)

orig_config.add_argument("--tasks_train_batch_sz", type=int, default=10)

orig_config.add_argument("--tasks_eval_batch_sz", type=int, default=10)

orig_config.add_argument("--num_iterations", type=int, default=1)

orig_config.add_argument("--task_model_type", type=str, required=True)

orig_config.add_argument("--lr", type=float, default=0.1)

# Admin

orig_config.add_argument("--cuda", default=False, action="store_true")

orig_config.add_argument("--eval_freq", type=int, required=True)

orig_config.add_argument("--store_results_freq", type=int, required=True)

orig_config.add_argument("--store_model_freq", type=int, required=True)

orig_config.add_argument("--specific_torch_seed", default=False, action="store_true")

orig_config.add_argument("--torch_seed", type=int)

orig_config = orig_config.parse_args()


def main(config):
  # ------------------------------------------------------------------------------------------------
  # Setup
  # ------------------------------------------------------------------------------------------------

  reproc_settings(config)

  config.out_dir = osp.join(config.out_root, str(config.model_ind))

  tasks_model, trainloader, testloader, valloader = get_model_and_data(config)

  if not osp.exists(config.out_dir):
    os.makedirs(config.out_dir)

  next_t = 0
  last_classes = None
  seen_classes = None

  optimizer = optim.SGD(tasks_model.parameters(), lr=config.lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)  # basic

  t = next_t

  while t <= config.max_t:
    for xs, ys in trainloader:
      present_classes = ys.unique().to(get_device(config.cuda))

      # --------------------------------------------------------------------------------------------
      # Eval
      # --------------------------------------------------------------------------------------------

      if (t - next_t) < 1000 or t % 100 == 0:
        print("m %d t %d %s, fst targets %s" % (
        config.model_ind, t, datetime.now(), str(list(present_classes.cpu().numpy()))))
        sys.stdout.flush()

      save_dict = {"tasks_model": tasks_model, "t": t, "last_classes": last_classes,
                   "seen_classes": seen_classes}

      last_step = t == (config.max_t)
      if (t % config.eval_freq == 0) or (t % config.batches_per_epoch == 0) or last_step or (
        t == 0):
        evaluate_basic(config, tasks_model, valloader, t, is_val=True, last_classes=last_classes,
                       seen_classes=seen_classes, compute_forgetting_metric=(not config.stationary))
        evaluate_basic(config, tasks_model, testloader, t, is_val=False, last_classes=last_classes,
                       seen_classes=seen_classes, compute_forgetting_metric=(not config.stationary))

      if (t % config.store_model_freq == 0) or last_step:
        torch.save(save_dict, osp.join(config.out_dir, "latest_models.pytorch"))

      if (t % config.store_results_freq == 0) or last_step:
        render_graphs(config)
        store(config)

      if last_step:
        return

      # --------------------------------------------------------------------------------------------
      # Train
      # --------------------------------------------------------------------------------------------

      tasks_model.train()

      optimizer.zero_grad()

      xs = xs.to(get_device(config.cuda))
      ys = ys.to(get_device(config.cuda))

      preds = tasks_model(xs)
      loss_orig = torch.nn.functional.cross_entropy(preds, ys, reduction="mean")

      loss_orig.backward()

      optimizer.step()

      record_and_check(config, "loss_orig", loss_orig.item(), t)

      t += 1
      if seen_classes is None:
        seen_classes = present_classes
      else:
        seen_classes = torch.cat((seen_classes, present_classes)).unique()
      last_classes = present_classes


if __name__ == "__main__":
  ms = range(orig_config.model_ind_start, orig_config.model_ind_start + orig_config.num_runs)
  for m in ms:
    print("Doing m %d" % m)
    c = deepcopy(orig_config)
    c.model_ind = m
    main(c)
    print("Done m %d" % m)
