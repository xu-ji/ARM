# Automatic Recall Machines

This repository contains the code for <a href="https://arxiv.org/abs/2006.12323">Automatic Recall Machines: Internal Replay, Continual Learning and the Brain</a>.

<img src="https://github.com/xu-ji/ARM/blob/master/summary.png" alt="ARM" height=250>

As well as ARM, we include implementations of Adaptive DeepInversion and LwF-style distillation.

# Dependencies

Our environment used:
- python 3.6.8
- pytorch 1.4.0
- torchvision 0.5.0
- numpy 1.18.4

# Run the code

Commands for all our results on CIFAR10, MiniImageNet and MNIST are given in `commands.txt`. 
For example, to run recall on CIFAR10:
```
python -m code.scripts.ARM --model_ind_start 3717 --num_runs 5 --data cifar10 --lr 0.01 --task_model_type resnet18 --classes_per_task 2 --recall_from_t 950 --num_iterations 1 --M 100 --refine_sample_steps 10 --refine_sample_lr 10.0 --divergence_loss_weight 1.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 950 --sharpen_class --sharpen_class_weight 0.1 --notlocal_weight 1.0 --notlocal_new_weight 0.1 --diversity_weight 16.0 --aux_distill --aux_distill_weight 1.0 --max_t 4750 --store_model_freq 4750 --store_results_freq 950 --eval_freq 950 --cuda --out_root /scratch/ARM --data_path /scratch/CIFAR
```
Print results:
```
python -m code.scripts.print_results --root /scratch/shared/nfs1/xuji/ARM --start 3717

average val: acc 0.2586 +- 0.0145, forgetting 0.1046 +- 0.0330 
average test: acc 0.2687 +- 0.0107, forgetting 0.0959 +- 0.0371
```
