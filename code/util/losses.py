import sys

import torch
import torch.nn.functional as F


def TV(x):
  if len(x.shape) == 2:  # mnist
    side = int(x.shape[1] ** 0.5)
    assert (side == 28)  # sanity
    x = x.view(-1, 1, side, side)

  batch_sz = x.shape[0]
  h_x = x.shape[2]
  w_x = x.shape[3]
  count_h = batch_sample_size(
    x[:, :, 1:, :])  # num points in 1 image inc channels except 1 row missing
  count_w = batch_sample_size(x[:, :, :, 1:])
  h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
  w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
  return 2 * (h_tv / count_h + w_tv / count_w) / batch_sz


def batch_sample_size(x):
  return x.shape[1] * x.shape[2] * x.shape[3]


def L2(x):
  if len(x.shape) == 2:  # mnist
    side = int(x.shape[1] ** 0.5)
    assert (side == 28)  # sanity
    x = x.view(-1, 1, side, side)

  sz = batch_sample_size(x)
  x = x.view(x.shape[0], -1)
  res = torch.norm(x, p=2, dim=1).sum()  # this is not size normalised!
  return res / (x.shape[0] * sz)


def sharpen_class(aux_preds):
  # minimise cross entropy with its own argmax
  hard_targets = aux_preds.argmax(dim=1).detach()
  return F.cross_entropy(aux_preds, hard_targets, reduction="mean")


def img_distance(aux_x, xs):
  aux_x_exp = aux_x.repeat(xs.shape[0], 1, 1, 1)
  xs_exp = xs.repeat_interleave(aux_x.shape[0], dim=0)
  assert (aux_x_exp.shape[1:] == xs_exp.shape[1:])
  assert (aux_x.shape[1:] == aux_x_exp.shape[1:])

  assert (aux_x_exp.shape[0] == aux_x.shape[0] * xs.shape[0])
  assert (xs_exp.shape[0] == aux_x.shape[0] * xs.shape[0])

  # maximise the distance
  return -F.mse_loss(aux_x_exp, xs_exp, reduction="mean")


def diversity_raw(preds):
  # maximise the entropy, use avg of logs
  assert (len(preds.shape) == 2)
  cross_batch_preds = preds.mean(dim=0, keepdim=True)
  entrop = crossent_logits(cross_batch_preds, cross_batch_preds)
  return -entrop


def diversity(preds, EPS=sys.float_info.epsilon):
  # maximise the entropy, use avg after softmax!!
  assert (len(preds.shape) == 2)

  preds = F.softmax(preds, dim=1)
  cross_batch_preds = preds.mean(dim=0)
  cross_batch_preds[(cross_batch_preds < EPS).detach()] = EPS
  loss = (cross_batch_preds * torch.log(cross_batch_preds)).sum()  # minimise negative entropy
  return loss


def notlocal(preds, present_classes):
  # by minimizing this, maximise distance with present classes

  num_samples, _ = preds.shape
  assert (len(present_classes.shape) == 1)

  cum_losses = preds.new_zeros(num_samples)
  for c in present_classes:
    c_targets = preds.new_full((num_samples,), c, dtype=torch.long)
    y_loss = - F.cross_entropy(preds, c_targets, reduction="none")
    cum_losses += y_loss

  loss = cum_losses.mean() / present_classes.shape[0]
  assert (loss.requires_grad)

  return loss


def crossent_logits(preds, targets):
  # targets are not yet softmaxed probabilities
  assert (len(targets.shape) == 2)
  targets = F.softmax(targets, dim=1)

  num_samples, num_outputs = preds.shape
  cum_losses = preds.new_zeros(num_samples)

  for c in range(num_outputs):
    # grads should still bp back through pred and target if nec
    target_temp = preds.new_full((num_samples,), c, dtype=torch.long)  # filled with c
    y_loss = F.cross_entropy(preds, target_temp, reduction="none")
    cum_losses += targets[:, c] * y_loss  # weight each one by prob c given by actual target

  loss = cum_losses.mean()
  assert (loss.requires_grad)
  return loss


def deep_inversion_classes_loss(classes_to_refine, aux_preds_old):
  # try to get characteristic images for these classes
  assert (classes_to_refine.shape == (aux_preds_old.shape[0],))
  # aux preds old not softmaxed
  return F.cross_entropy(aux_preds_old, classes_to_refine, reduction="mean")


def neg_symmetric_KL(aux_preds_old, aux_preds_new):
  assert (len(aux_preds_old.shape) == 2)
  assert (len(aux_preds_new.shape) == 2)
  assert (aux_preds_new.shape == aux_preds_old.shape)

  # inputs logsoftmax, targets softmax
  avg_preds = 0.5 * (F.softmax(aux_preds_old, dim=1) + F.softmax(aux_preds_new, dim=1))

  return 1 - 0.5 * (
  F.kl_div(F.log_softmax(aux_preds_old, dim=1), avg_preds, reduction="batchmean") +
  F.kl_div(F.log_softmax(aux_preds_new, dim=1), avg_preds, reduction="batchmean"))


def avoid_unseen_classes(aux_preds_old, seen_classes):
  # max KL div with each of seen classes
  num_samples, num_outputs = aux_preds_old.shape

  total = None
  for c in seen_classes:
    target_temp = aux_preds_old.new_full((num_samples,), c, dtype=torch.long)  # filled with c
    y_loss = F.cross_entropy(aux_preds_old, target_temp, reduction="mean")

    if total is None:
      total = y_loss
    else:
      total += y_loss

  return - total / seen_classes.shape[0]  # avg and negate, because encouraging separation
