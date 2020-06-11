import argparse
import os
from copy import deepcopy
from datetime import datetime

import torch.optim as optim

from code.util.eval import evaluate_basic
from code.util.general import *
from code.util.load import *
from code.util.losses import *

# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------

orig_config = argparse.ArgumentParser(allow_abbrev=False)

orig_config.add_argument("--model_ind_start", type=int, required=True)

orig_config.add_argument("--num_runs", type=int, required=True)

orig_config.add_argument("--out_root", type=str, default="/scratch/shared/nfs1/xuji/iam")

# Data and model

orig_config.add_argument("--data", type=str, default="cifar100")

orig_config.add_argument("--data_path", type=str,
                         default="/scratch/shared/nfs1/xuji/datasets/CIFAR")

orig_config.add_argument("--stationary", default=False, action="store_true")

orig_config.add_argument("--classes_per_task", type=int, required=True)

orig_config.add_argument("--max_t", type=int, required=True)

orig_config.add_argument("--tasks_train_batch_sz", type=int, default=10)

orig_config.add_argument("--tasks_eval_batch_sz", type=int, default=10)

orig_config.add_argument("--num_iterations", type=int, default=1)

orig_config.add_argument("--task_model_type", type=str, required=True)

# Distill options

orig_config.add_argument("--lr", type=float, default=0.1)

orig_config.add_argument("--recall_from_t", type=int, required=True)

orig_config.add_argument("--long_window", default=False, action="store_true")

orig_config.add_argument("--long_window_range", type=int, nargs="+", default=[1, 1])  # inclusive

orig_config.add_argument("--use_fixed_window", default=False, action="store_true")

orig_config.add_argument("--fixed_window", type=int, default=1)

orig_config.add_argument("--aux_distill_weight", type=float, default=1.0)

# Admin

orig_config.add_argument("--cuda", default=False, action="store_true")

orig_config.add_argument("--eval_freq", type=int, default=76)

orig_config.add_argument("--store_results_freq", type=int, default=76)

orig_config.add_argument("--store_model_freq", type=int, default=380)

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

  if config.long_window:
    old_tasks_model = None
    config.next_update_old_model_t_history = []
    config.next_update_old_model_t = 0
    assert (len(config.long_window_range) == 2)

  optimizer = optim.SGD(tasks_model.parameters(), lr=config.lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

  refine_theta_metrics = ["final_loss_aux", "loss_aux_distill"]

  t = next_t

  while t <= config.max_t:
    for xs, ys in trainloader:
      present_classes = ys.unique().to(get_device(config.cuda))

      # --------------------------------------------------------------------------------------------
      # Eval
      # --------------------------------------------------------------------------------------------

      if (t - next_t) < 1000 or t % 100 == 0:
        print("m %d t: %d %s, targets %s" % (
        config.model_ind, t, datetime.now(), str(list(present_classes.cpu().numpy()))))
        sys.stdout.flush()

      save_dict = {"tasks_model": tasks_model, "t": t, "last_classes": last_classes,
                   "seen_classes": seen_classes}
      if config.long_window:  # else no need to save because it's always the one just before
        # current update
        save_dict["old_tasks_model"] = old_tasks_model

      last_step = t == (config.max_t)
      if (t % config.eval_freq == 0) or (t % config.batches_per_epoch == 0) or last_step or (
        t == 0):
        evaluate_basic(config, tasks_model, valloader, t, is_val=True,
                       last_classes=last_classes, seen_classes=seen_classes)
        evaluate_basic(config, tasks_model, testloader, t, is_val=False,
                       last_classes=last_classes, seen_classes=seen_classes)

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

      xs = xs.to(get_device(config.cuda))
      ys = ys.to(get_device(config.cuda))

      curr_classes = ys.unique()
      assert (curr_classes.max() < config.task_out_dims[0])

      optimizer.zero_grad()

      # set old_tasks_model if needed
      if not config.long_window or (config.long_window and t == config.next_update_old_model_t):
        old_tasks_model = deepcopy(tasks_model)

        if config.long_window:
          config.next_update_old_model_t_history.append(config.next_update_old_model_t)
          if config.use_fixed_window:
            window_offset = config.fixed_window
          else:
            window_offset = np.random.randint(config.long_window_range[0],
                                              high=(config.long_window_range[1] + 1))

          config.next_update_old_model_t = t + window_offset

      # --------------------------------------------------------------------------------------------
      # Train on real data
      # --------------------------------------------------------------------------------------------

      preds = tasks_model(xs)
      loss_orig = F.cross_entropy(preds, ys, reduction="mean")
      record_and_check(config, "loss_orig", loss_orig.item(), t)  # added!
      loss_orig.backward()
      optimizer.step()  # updates tasks_model, which is now \theta'

      # --------------------------------------------------------------------------------------------
      # Distill
      # --------------------------------------------------------------------------------------------
      if t >= config.recall_from_t:
        optimizer.zero_grad()
        metrics = dict([(metric, 0.) for metric in refine_theta_metrics])

        with torch.no_grad():
          distill_xs_targets = old_tasks_model(xs)

        loss_aux_distill = F.kl_div(F.log_softmax(tasks_model(xs), dim=1),
                                    F.softmax(distill_xs_targets, dim=1), reduction="batchmean")
        final_loss_aux = config.aux_distill_weight * loss_aux_distill

        metrics["loss_aux_distill"] += loss_aux_distill.item()
        metrics["final_loss_aux"] += final_loss_aux.item()

        final_loss_aux.backward()
        optimizer.step()  # updates tasks_model

        for metric in refine_theta_metrics:
          record_and_check(config, metric, metrics[metric], t)

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
