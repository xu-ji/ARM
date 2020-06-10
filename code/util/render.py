import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb


def render_aux_x(config, t, r, aux_x_orig, aux_x, aux_y, present_classes):
  if len(aux_x_orig.shape) == 2:
    assert (config.data == "mnist5k")
    side = int(aux_x_orig.shape[1] ** 0.5)
    aux_x_orig = aux_x_orig.view(-1, side, side)
    if not aux_x is None:
      aux_x = aux_x.view(-1, side, side)

    render_out_dir = os.path.join(config.out_dir, "render")
    if not os.path.exists(render_out_dir):
      os.makedirs(render_out_dir)

  if not aux_x is None:
    # render a random selection of render_aux_x_num samples

    if not config.render_separate:
      fig0, axarr0 = plt.subplots(2, max(config.render_aux_x_num, 2), sharex=False,
                                  figsize=(config.render_aux_x_num * 4, 2 * 4))

    num_aux_x = aux_x.shape[0]
    assert (num_aux_x == config.M)
    assert (config.render_aux_x_num <= num_aux_x)
    selected = np.random.choice(num_aux_x, config.render_aux_x_num, replace=False)

    s_aux_x = aux_x[selected]
    s_aux_y = aux_y[selected]
    s_aux_x_orig = aux_x_orig[selected]
    if config.render_aux_x_num == 1:
      s_aux_x = s_aux_x.unsqueeze(dim=0)
      s_aux_x_orig = s_aux_x_orig.unsqueeze(dim=0)
      s_aux_y = s_aux_y.unsqueeze(dim=0)

    s_aux_y = torch.nn.functional.softmax(s_aux_y, dim=1)  # pre softmax

    diff_sum = (s_aux_x_orig - s_aux_x).abs().sum().item() / config.render_aux_x_num

    for i in range(config.render_aux_x_num):
      img_orig = s_aux_x_orig[i]
      img = s_aux_x[i]

      if len(s_aux_x_orig.shape) == 4:
        img_orig = img_orig.permute(1, 2, 0)
        img = img.permute(1, 2, 0)  # h, w, 3

      orig_min, orig_max = img.min(), img.max()
      img = img - orig_min
      img = img / orig_max

      top_class = s_aux_y[i].argmax()

      if not config.render_separate:
        axarr0[0, i].imshow(img.cpu().numpy())

        vals = (
        top_class.item(), s_aux_y[i, top_class].item(), str(list(present_classes.cpu().numpy())),
        orig_min.item(), orig_max.item())

        axarr0[0, i].set_title("top c %d: val %f, pres %s, img min %f max %f" % vals, fontsize=6)
        axarr0[1, i].imshow(img_orig.cpu().numpy())
      else:
        fig_curr_orig, ax_curr_orig = plt.subplots(1, figsize=(4, 4))
        ax_curr_orig.imshow(img_orig.cpu().numpy())
        # ax_curr_orig.set_axis_off() # barebones

        ax_curr_orig.axis('off')
        fig_curr_orig.patch.set_visible(False)
        ax_curr_orig.patch.set_visible(False)

        fig_curr_orig.savefig(
          os.path.join(render_out_dir, "m_%d_t_%d_%r_i_%d_orig.png" % (config.model_ind, t, r, i)),
          bbox_inches=0)

        fig_curr_aux, ax_curr_aux = plt.subplots(1, figsize=(4, 4))
        ax_curr_aux.imshow(img.cpu().numpy())
        # ax_curr_aux.set_axis_off() # barebones

        ax_curr_aux.axis('off')
        fig_curr_aux.patch.set_visible(False)
        ax_curr_aux.patch.set_visible(False)

        fig_curr_aux.savefig(os.path.join(render_out_dir, "m_%d_t_%d_%r_i_%d_aux_%d_%.3f.png" %
                                          (config.model_ind, t, r, i, top_class.item(),
                                           s_aux_y[i, top_class].item())), bbox_inches=0)

        plt.close("all")

    if not config.render_separate:
      fig0.suptitle("Model %d t %d r %d, diffs %f" % (config.model_ind, t, r, diff_sum), fontsize=8)
      fig0.savefig(
        os.path.join(render_out_dir, "render_m_%d_t_%d_r_%d.png" % (config.model_ind, t, r)))
      plt.close("all")
  else:
    # just render original batch
    for i in range(config.render_aux_x_num):
      img_orig = aux_x_orig[i]
      if len(img_orig.shape) == 3:
        img_orig = img_orig.permute(1, 2, 0)  # channels last

      fig_curr_orig, ax_curr_orig = plt.subplots(1, figsize=(4, 4))
      ax_curr_orig.imshow(img_orig.cpu().numpy())
      # ax_curr_orig.set_axis_off() # barebones

      ax_curr_orig.axis('off')
      fig_curr_orig.patch.set_visible(False)
      ax_curr_orig.patch.set_visible(False)

      fig_curr_orig.savefig(
        os.path.join(render_out_dir, "m_%d_t_%d_%r_i_%d_orig.png" % (config.model_ind, t, r, i)),
        bbox_inches=0)


def get_colours(n):
  hues = np.linspace(0.0, 1.0, n + 1)[0:-1]  # ignore last one
  all_colours = [np.array(hsv_to_rgb(hue, 0.75, 0.75)) for hue in hues]
  return all_colours
