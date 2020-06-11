import argparse
from copy import deepcopy
from datetime import datetime
import os

import torch.optim as optim

from code.models import *
from code.util.data import *
from code.util.eval import evaluate_basic
from code.util.general import *
from code.util.load import *
from code.util.losses import *
from code.util.render import render_aux_x

# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------

orig_config = argparse.ArgumentParser(allow_abbrev=False)

orig_config.add_argument("--model_ind_start", type=int, required=True)

orig_config.add_argument("--num_runs", type=int, required=True)

orig_config.add_argument("--out_root", type=str, required=True)

# Data and model

orig_config.add_argument("--data", type=str, required=True)

orig_config.add_argument("--data_path", type=str, required=True)

orig_config.add_argument("--stationary", default=False, action="store_true")

orig_config.add_argument("--classes_per_task", type=int, required=True)

orig_config.add_argument("--max_t", type=int, required=True)

orig_config.add_argument("--tasks_train_batch_sz", type=int, default=10)

orig_config.add_argument("--tasks_eval_batch_sz", type=int, default=10)

orig_config.add_argument("--num_iterations", type=int, default=1)

orig_config.add_argument("--task_model_type", type=str, required=True)

# ARM

orig_config.add_argument("--recall_from_t", type=int, required=True)

orig_config.add_argument("--M", type=int, default=100)

orig_config.add_argument("--lr", type=float, default=0.1)  # eta_0

orig_config.add_argument("--refine_sample_lr", type=float, default=0.1)  # eta_1

orig_config.add_argument("--refine_sample_steps", type=int, default=10)  # S

orig_config.add_argument("--aux_weight", type=float, default=1.0)  # lambda_0 (loss on recall)

orig_config.add_argument("--aux_distill", default=False, action="store_true")

orig_config.add_argument("--aux_distill_weight", type=float, default=1.0)  # lambda_0 (loss on real)

orig_config.add_argument("--divergence_loss_weight", type=float, default=1.0)  # optional weight D

orig_config.add_argument("--notlocal_weight", type=float, default=1.0)  # lambda_1

orig_config.add_argument("--notlocal_new_weight", type=float, default=1.0)  # lambda_2

orig_config.add_argument("--diversity_weight", type=float, default=1.0)  # lambda_3

orig_config.add_argument("--sharpen_class", default=False, action="store_true")

orig_config.add_argument("--sharpen_class_weight", type=float, default=1.0)  # lambda_4

orig_config.add_argument("--TV", default=False, action="store_true")

orig_config.add_argument("--TV_weight", type=float, default=1.0)  # lambda_5

orig_config.add_argument("--L2", default=False, action="store_true")

orig_config.add_argument("--L2_weight", type=float, default=1.0)  # lambda_6

orig_config.add_argument("--long_window", default=False, action="store_true")

orig_config.add_argument("--long_window_range", type=int, nargs="+", default=[1, 1])

orig_config.add_argument("--use_fixed_window", default=False, action="store_true")

orig_config.add_argument("--fixed_window", type=int, default=1)  # when to update old theta

# Usually not used, included for testing:

orig_config.add_argument("--refine_theta_steps", type=int, default=1)

orig_config.add_argument("--refine_sample_from_scratch", default=False, action="store_true")

orig_config.add_argument("--hard_targets", default=False, action="store_true")

orig_config.add_argument("--aux_x_random", default=False, action="store_true")

orig_config.add_argument("--use_crossent_as_D", default=False, action="store_true")

orig_config.add_argument("--opt_batch_stats", default=False, action="store_true")

orig_config.add_argument("--opt_batch_stats_weight", type=float, default=1.0)

# Admin

orig_config.add_argument("--cuda", default=False, action="store_true")

orig_config.add_argument("--eval_freq", type=int, required=True)

orig_config.add_argument("--store_results_freq", type=int, required=True)

orig_config.add_argument("--store_model_freq", type=int, required=True)

orig_config.add_argument("--specific_torch_seed", default=False, action="store_true")

orig_config.add_argument("--torch_seed", type=int)

orig_config.add_argument("--render_aux_x", default=False, action="store_true")

orig_config.add_argument("--render_aux_x_freq", type=int, default=10)

orig_config.add_argument("--render_aux_x_num", type=int, default=3)

orig_config.add_argument("--render_separate", default=False, action="store_true")

orig_config.add_argument("--count_params_only", default=False, action="store_true")

orig_config = orig_config.parse_args()


def main(config):
  # ------------------------------------------------------------------------------------------------
  # Setup
  # ------------------------------------------------------------------------------------------------

  reproc_settings(config)

  config.out_dir = osp.join(config.out_root, str(config.model_ind))

  tasks_model, trainloader, testloader, valloader = get_model_and_data(config)

  if config.count_params_only:
    sz = 0
    for n, p in tasks_model.named_parameters():
      curr_sz = np.prod(tuple(p.data.shape))
      print((n, p.data.shape, curr_sz))
      sz += curr_sz
    print("Num params: %d" % sz)
    exit(0)

  if not osp.exists(config.out_dir):
    os.makedirs(config.out_dir)

  next_t = 0
  last_classes = None  # for info only
  seen_classes = None

  if config.long_window:
    old_tasks_model = None
    config.next_update_old_model_t_history = []
    config.next_update_old_model_t = 0  # Old model set in first timestep
    assert (len(config.long_window_range) == 2)

  optimizer = optim.SGD(tasks_model.parameters(), lr=config.lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

  refine_sample_metrics = ["divergence_loss", "notlocal_loss", "notlocal_new_loss",
                           "diversity_loss", "loss_refine"]
  refine_theta_metrics = ["not_present_class", "loss_aux", "final_loss_aux"]

  if config.sharpen_class: refine_sample_metrics.append("sharpen_class_loss")
  if config.TV: refine_sample_metrics.append("TV_loss")
  if config.L2: refine_sample_metrics.append("L2_loss")
  if config.opt_batch_stats: refine_sample_metrics.append("opt_batch_stats_loss")
  if config.aux_distill: refine_theta_metrics.append("loss_aux_distill")

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

      optimizer.zero_grad()

      # Update old_tasks_model if needed
      if not config.long_window or (config.long_window and t == config.next_update_old_model_t):
        old_tasks_model = deepcopy(tasks_model)
        if config.opt_batch_stats: old_tasks_model.compute_batch_stats_loss()

        if config.long_window:
          config.next_update_old_model_t_history.append(config.next_update_old_model_t)
          if config.use_fixed_window:
            window_offset = config.fixed_window
          else:
            window_offset = np.random.randint(config.long_window_range[0], high=(
            config.long_window_range[1] + 1))  # randint is excl

          config.next_update_old_model_t = t + window_offset

      # --------------------------------------------------------------------------------------------
      # Train on real data
      # --------------------------------------------------------------------------------------------

      preds = tasks_model(xs)
      loss_orig = F.cross_entropy(preds, ys, reduction="mean")
      record_and_check(config, "loss_orig", loss_orig.item(), t)
      loss_orig.backward()
      optimizer.step()

      # --------------------------------------------------------------------------------------------
      # Generate recall
      # --------------------------------------------------------------------------------------------

      if t >= config.recall_from_t:
        optimizer.zero_grad()

        metrics = dict([(metric, 0.) for metric in refine_sample_metrics + refine_theta_metrics])

        for r in range(config.refine_theta_steps):  # each step updates tasks_model
          if config.refine_sample_from_scratch or r == 0:  # fresh aux_x
            if not config.aux_x_random:
              chosen_aux_inds = np.random.choice(xs.shape[0], config.M,
                                                 replace=(config.M > xs.shape[0]))
              aux_x = xs[chosen_aux_inds]
            else:
              aux_x = torch.rand((config.M,) + xs.shape[1:]).to(get_device(config.cuda))

          aux_x_orig = aux_x.clone()
          aux_x.requires_grad_(True)

          for s in range(config.refine_sample_steps):
            aux_preds_old = old_tasks_model(aux_x)
            aux_preds_new = tasks_model(aux_x)

            if config.opt_batch_stats:
              aux_preds_old, opt_batch_stats_loss = aux_preds_old

            if config.use_crossent_as_D:
              divergence_loss = - crossent_logits(preds=aux_preds_new, targets=aux_preds_old)
            else:
              divergence_loss = neg_symmetric_KL(aux_preds_old, aux_preds_new)

            notlocal_loss = notlocal(aux_preds_old, present_classes)
            notlocal_new_loss = notlocal(aux_preds_new, present_classes)

            diversity_loss = diversity(aux_preds_old)

            loss_refine = config.divergence_loss_weight * divergence_loss + \
                          config.diversity_weight * diversity_loss + \
                          config.notlocal_weight * notlocal_loss + config.notlocal_new_weight * \
                                                                   notlocal_new_loss

            if config.sharpen_class:
              sharpen_class_loss = sharpen_class(aux_preds_old)
              loss_refine += config.sharpen_class_weight * sharpen_class_loss

            if config.TV:
              TV_loss = TV(aux_x)
              loss_refine += config.TV_weight * TV_loss

            if config.L2:
              L2_loss = L2(aux_x)
              loss_refine += config.L2_weight * L2_loss

            if config.opt_batch_stats:
              loss_refine += config.opt_batch_stats_weight * opt_batch_stats_loss

            for metric in refine_sample_metrics:
              metrics[metric] += locals()[metric].item()

            aux_x_grads = \
            torch.autograd.grad(loss_refine, aux_x, only_inputs=True, retain_graph=False)[0]
            aux_x = (aux_x - config.refine_sample_lr * aux_x_grads).detach().requires_grad_(True)

          # Get final predictions on recalled data from old model
          aux_x.requires_grad_(False)
          with torch.no_grad():
            aux_y = old_tasks_model(aux_x)
            distill_xs_targets = old_tasks_model(xs)

            if config.opt_batch_stats:
              aux_y, _ = aux_y
              distill_xs_targets, _ = distill_xs_targets

          if config.render_aux_x and t % config.render_aux_x_freq == 0:
            render_aux_x(config, t, r, aux_x_orig, aux_x, aux_y, present_classes)

          # count number whose top class isn't in current classes
          aux_y_hard = aux_y.argmax(dim=1)

          if not hasattr(config, "aux_y_hard"):
            config.aux_y_hard = OrderedDict()
          if t not in config.aux_y_hard:
            config.aux_y_hard[t] = aux_y_hard.cpu()
          else:
            config.aux_y_hard[t] = torch.cat((config.aux_y_hard[t], aux_y_hard.cpu())).unique()

          if not hasattr(config, "aux_y_probs"):
            config.aux_y_probs = OrderedDict()
          aux_y_probs = F.softmax(aux_y, dim=1).mean(dim=0)
          if t not in config.aux_y_probs:
            config.aux_y_probs[t] = aux_y_probs.cpu() * (1. / config.refine_theta_steps)  # n c
          else:
            config.aux_y_probs[t] += (aux_y_probs.cpu() * (1. / config.refine_theta_steps))

          is_present_class = 0
          for c in present_classes:
            is_present_class += (aux_y_hard == c).sum().item()
          metrics["not_present_class"] += aux_y_hard.shape[0] - is_present_class

          # ----------------------------------------------------------------------------------------
          # Train on recalled data
          # ----------------------------------------------------------------------------------------

          preds = tasks_model(aux_x)
          if not config.hard_targets:
            loss_aux = crossent_logits(preds=preds, targets=aux_y)
          else:
            loss_aux = F.cross_entropy(preds, aux_y_hard, reduction="mean")

          final_loss_aux = loss_aux * config.aux_weight

          if config.aux_distill:
            loss_aux_distill = F.kl_div(F.log_softmax(tasks_model(xs), dim=1),
                                        F.softmax(distill_xs_targets, dim=1), reduction="batchmean")
            final_loss_aux += config.aux_distill_weight * loss_aux_distill
            metrics["loss_aux_distill"] += loss_aux_distill.item()

          metrics["loss_aux"] += loss_aux.item()
          metrics["final_loss_aux"] += final_loss_aux.item()

          final_loss_aux.backward()
          optimizer.step()

        for metric in refine_sample_metrics:
          metrics[metric] /= float(config.refine_sample_steps * config.refine_theta_steps)
          record_and_check(config, metric, metrics[metric], t)

        for metric in refine_theta_metrics:
          metrics[metric] /= float(config.refine_theta_steps)
          record_and_check(config, metric, metrics[metric], t)

      if t < config.recall_from_t and config.render_aux_x and t % config.render_aux_x_freq == 0:
        render_aux_x(config, t, 0, xs, None, None, present_classes)

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
