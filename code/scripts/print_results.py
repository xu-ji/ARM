import argparse
import os
import pickle
import torch
import numpy as np

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--root", type=str, required=True)
args.add_argument("--start", type=int, required=True)
args.add_argument("--load_model", default=False, action="store_true")
args.add_argument("--num_runs", type=int, default=5)
args = args.parse_args()

def treat_underscores(x):
  res = []
  for c in x:
    if c == "_":
      res.append("\\_")
    else:
      res.append(c)

  return "".join(res)


def print_results(args):
  ms_avg = {"val": {"acc": [], "forgetting": []},
            "test": {"acc": [], "forgetting": []}}

  for m in range(args.start, args.start + args.num_runs):
    out_dir = os.path.join(args.root, str(m))
    config_p = os.path.join(out_dir, "config.pickle")

    config = None
    tries = 0
    while tries < 1000:
      try:
        with open(config_p, "rb") as config_f:
          config = pickle.load(config_f)
        break
      except:
        tries += 1

    if config is None:
      continue

    if args.load_model:
      torch.load(os.path.join(config.out_dir, "latest_models.pytorch"))

    actual_t = config.max_t
    for prefix in ["val", "test"]:
      if not config.stationary:
        accs_dict = getattr(config, "%s_accs" % prefix)

        ms_avg[prefix]["acc"].append(accs_dict[actual_t])

        forgetting_dict = getattr(config, "%s_forgetting" % prefix)
        if actual_t in forgetting_dict:
          ms_avg[prefix]["forgetting"].append(forgetting_dict[actual_t])

        print("model %d, %s: acc %.4f, forgetting %.4f" % (
        config.model_ind, prefix, accs_dict[actual_t], forgetting_dict[actual_t]))
      else:
        accs_dict = getattr(config, "%s_accs_data" % prefix)
        ms_avg[prefix]["acc"].append(accs_dict[actual_t])
        print("model %d, %s: acc %.4f" % (config.model_ind, prefix, accs_dict[actual_t]))

    print("---")

  for prefix in ["val", "test"]:
    for metric in ["acc", "forgetting"]:
      if len(ms_avg[prefix][metric]) == 0:
        ms_avg[prefix][metric] = (-1, -1)
      else:
        avg = np.array(ms_avg[prefix][metric]).mean()
        std = np.array(ms_avg[prefix][metric]).std()
        ms_avg[prefix][metric] = (avg, std)

    print("average %s: acc %.4f +- %.4f, forgetting %.4f +- %.4f" % (
    prefix, ms_avg[prefix]["acc"][0], ms_avg[prefix]["acc"][1],
    ms_avg[prefix]["forgetting"][0], ms_avg[prefix]["forgetting"][1]))


if __name__ == "__main__":
  print_results(args)
