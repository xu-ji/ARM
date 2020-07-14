import torch
import os
import argparse
import pickle
import numpy as np

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--root", type=str, required=True)
args.add_argument("--num_runs", type=int, default=5)
args = args.parse_args()

experiments = [
  ("ARM MNIST", 6579),
  ("ARM Cifar10", 3717),
  ("ARM MiniImageNet", 4821),

  ("ADI MNIST", 6232),
  ("ADI Cifar10", 2262),
  ("ADI MiniImageNet", 6042),

  ("Distill MNIST", 6102),
  ("Distill Cifar10", 4082),
  ("Distill MiniImageNet", 6092),

  ("Naive MNIST", 4967),
  ("Naive Cifar10", 2522),
  ("Naive MiniImageNet", 4557),

  ("Naive stationary MNIST", 4947),
  ("Naive stationary Cifar10", 6462),
  ("Naive stationary MiniImageNet", 4527),

  # table 4
  ("Distill Cifar10, unit lag", 6452),
  ("ADI Cifar10, unit lag", 6467),
  ("ADI Cifar10, no distill", 2327),
  ("ARM Cifar10, unit lag", 6602),
  ("ARM Cifar10, no distill", 2717),

  # table 7
  ("$\lambda_1 = 0, \lambda_2 = 0$", 3982),
  ("$\lambda_3 = 0$", 6562),
  ("$\lambda_4 = 0$", 3977),
  ("$\lambda_5 = 0$", 3967),
  ("$\lambda_6 = 0$", 3972),
  ("$M = 150$ (+50)", 6624),
  ("$M = 50$ (-50)", 5922),
  ("$S = 20$ (doubled)", 3957),
  ("$S = 5$ (halved)", 3962),
  ("Cross-entropy as D", 3617),
  ("Random noise init", 4077),
  ("Recall 2x per t", 6502),
  ("Recall 4x per t", 6507),
]

num_runs = 5

print("LaTeX table:")
print("\\begin{table}[h]")
print("\\centering")
print("\\fontsize{7}{7}\\selectfont")
print("\\begin{tabular}{l c c c c}")
print("\\toprule")
print("& \\multicolumn{2}{c}{Val} & \\multicolumn{2}{c}{Test} \\\\")
print("& Accuracy & Forgetting & Accuracy & Forgetting \\\\")
print("\\midrule")
for name, m_start in experiments:
  ms_avg = {"val": {"acc": [], "forgetting": []},
            "test": {"acc": [], "forgetting": []}}

  counts = 0
  for m in range(m_start, m_start + args.num_runs):
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

    actual_t = config.max_t

    if not actual_t in config.test_accs:
      continue

    for prefix in ["val", "test"]:
      if not config.stationary:
        accs_dict = getattr(config, "%s_accs" % prefix)

        ms_avg[prefix]["acc"].append(accs_dict[actual_t])

        forgetting_dict = getattr(config, "%s_forgetting" % prefix)
        if actual_t in forgetting_dict:
          ms_avg[prefix]["forgetting"].append(forgetting_dict[actual_t])
      else:
        accs_dict = getattr(config, "%s_accs_data" % prefix)
        ms_avg[prefix]["acc"].append(accs_dict[actual_t])

    counts += 1

  for prefix in ["val", "test"]:
    for metric in ["acc", "forgetting"]:
      if len(ms_avg[prefix][metric]) == 0:
        ms_avg[prefix][metric] = (-1, -1)
      else:
        avg = np.array(ms_avg[prefix][metric]).mean()
        std = np.array(ms_avg[prefix][metric]).std()
        ms_avg[prefix][metric] = (avg, std)

  print("%s (%d) & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\\\" %
        (name, counts,

         ms_avg["val"]["acc"][0], ms_avg["val"]["acc"][1],
         ms_avg["val"]["forgetting"][0], ms_avg["val"]["forgetting"][1],

         ms_avg["test"]["acc"][0], ms_avg["test"]["acc"][1],
         ms_avg["test"]["forgetting"][0], ms_avg["test"]["forgetting"][1],
         ))

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
