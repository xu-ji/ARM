# ARM

# ------------------------------------------
Dependencies:
# ------------------------------------------

Our environment used:

python 3.6.8
pytorch 1.4.0
torchvision 0.5.0
numpy 1.18.4

# ------------------------------------------
Commands:
# ------------------------------------------

ARM

  MNIST
  export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.ARM  --model_ind_start 5752 --num_runs 5 --data mnist5k --lr 0.05 --task_model_type mlp  --classes_per_task 2 --recall_from_t 100 --num_iterations 1 --M 10 --refine_sample_steps 10 --refine_sample_lr 25.0 --divergence_loss_weight 1.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 100 --sharpen_class --sharpen_class_weight 0.1 --notlocal_weight 1.0 --notlocal_new_weight 0.1 --diversity_weight 16.0 --aux_distill --aux_distill_weight 1.0 --max_t 500 --store_model_freq 500 --store_results_freq 100 --eval_freq 100 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/ > out/m5752s.out &

  CIFAR10
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.ARM  --model_ind_start 3717 --num_runs 5 --data cifar10 --lr 0.01 --task_model_type resnet18 --classes_per_task 2 --recall_from_t 950 --num_iterations 1 --M 100 --refine_sample_steps 10 --refine_sample_lr 10.0 --divergence_loss_weight 1.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 950 --sharpen_class --sharpen_class_weight 0.1 --notlocal_weight 1.0 --notlocal_new_weight 0.1 --diversity_weight 16.0 --aux_distill --aux_distill_weight 1.0 --max_t 4750 --store_model_freq 4750 --store_results_freq 950 --eval_freq 950 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/CIFAR > out/m3717s.out &

  MiniImageNet
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.ARM --model_ind_start 4821 --num_runs 5 --data miniimagenet --lr 0.01 --task_model_type resnet18 --classes_per_task 5 --recall_from_t 684 --num_iterations 3 --M 100 --refine_sample_steps 10 --refine_sample_lr 10.0 --divergence_loss_weight 1.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 684 --sharpen_class --sharpen_class_weight 1.0 --notlocal_weight 1.0 --notlocal_new_weight 0.1 --diversity_weight 1.0 --aux_distill --aux_distill_weight 2.0 --max_t 13680 --store_model_freq 13680 --store_results_freq 684 --eval_freq 684 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/MINIIMAGENET/ > out/m4821.out &

ADI
  MNIST
  export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.ADI  --model_ind_start 6232 --num_runs 5 --data mnist5k --lr 0.05 --task_model_type mlp  --classes_per_task 2 --recall_from_t 100 --num_iterations 1 --M 10 --refine_theta_steps 1 --refine_sample_steps 10 --refine_sample_lr 25.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 100 --classes_loss_weight 1.0 --choose_past_classes --adaptive --adaptive_weight 1.0 --aux_distill_weight 0.5 --max_t 500 --store_model_freq 500 --store_results_freq 100 --eval_freq 100 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/ > out/m6232s.out &

  CIFAR10
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.ADI  --model_ind_start 2262 --num_runs 5 --data cifar10 --lr 0.01 --task_model_type resnet18_batch_stats --classes_per_task 2 --recall_from_t 950 --num_iterations 1 --M 100 --refine_theta_steps 1 --refine_sample_steps 10 --refine_sample_lr 10.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 950 --classes_loss_weight 1.0 --opt_batch_stats --opt_batch_stats_weight 0.1 --choose_past_classes --adaptive --adaptive_weight 1.0 --max_t 4750 --store_model_freq 4750 --store_results_freq 950 --eval_freq 950 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/CIFAR > out/m2262s.out &

  MiniImageNet
  export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.ADI --model_ind_start 6042 --num_runs 5 --data miniimagenet --lr 0.01 --task_model_type resnet18_batch_stats --classes_per_task 5 --recall_from_t 684 --num_iterations 3 --M 100 --refine_theta_steps 1 --refine_sample_steps 10 --refine_sample_lr 10.0 --L2 --L2_weight 1.0 --TV --TV_weight 1.0 --long_window --use_fixed_window --fixed_window 684 --classes_loss_weight 1.0 --opt_batch_stats --opt_batch_stats_weight 0.1 --choose_past_classes --adaptive --adaptive_weight 1.0 --aux_distill_weight 2.0 --max_t 13680 --store_model_freq 13680 --store_results_freq 684 --eval_freq 684 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/MINIIMAGENET/ > out/m6042s.out &

Distill
  MNIST
  export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.distill --model_ind_start 6102 --num_runs 5 --data mnist5k --lr 0.05 --task_model_type mlp --classes_per_task 2 --recall_from_t 100 --num_iterations 1 --long_window --use_fixed_window --fixed_window 100 --aux_distill_weight 1.0 --max_t 500 --store_model_freq 500 --store_results_freq 100 --eval_freq 100 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/ > out/m6102s.out &

  CIFAR10
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.distill  --model_ind_start 4082 --num_runs 5 --data cifar10 --lr 0.01 --task_model_type resnet18 --classes_per_task 2 --recall_from_t 950 --num_iterations 1 --long_window --use_fixed_window --fixed_window 950 --aux_distill_weight 1.0 --max_t 4750 --store_model_freq 4750 --store_results_freq 950 --eval_freq 950 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/CIFAR > out/m4082s.out &

  MiniImageNet
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.distill  --model_ind_start 6092 --num_runs 5 --data miniimagenet --lr 0.01 --task_model_type resnet18 --classes_per_task 5 --recall_from_t 684 --num_iterations 3 --long_window --use_fixed_window --fixed_window 684 --aux_distill_weight 2.0 --max_t 13680 --store_model_freq 13680 --store_results_freq 684 --eval_freq 684 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/MINIIMAGENET/ > out/m6092s.out &


Naive
  MNIST
  export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.naive --model_ind_start 4967 --num_runs 5 --data mnist5k  --lr 0.05 --task_model_type mlp --classes_per_task 2 --num_iterations 1 --max_t 500 --store_model_freq 500 --store_results_freq 100 --eval_freq 100 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/ > out/m4967s.out &

  CIFAR10
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.naive --model_ind_start 2522 --num_runs 5 --data cifar10 --lr 0.1 --task_model_type resnet18 --classes_per_task 2 --num_iterations 1 --max_t 4750 --store_model_freq 4750 --store_results_freq 950 --eval_freq 950 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/CIFAR > out/m2522s.out &

  MiniImageNet
  export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.naive --model_ind_start 4557 --num_runs 5 --data miniimagenet --classes_per_task 5 --lr 0.1 --task_model_type resnet18 --num_iterations 3 --max_t 13680 --store_model_freq 13680 --store_results_freq 684 --eval_freq 684 --cuda --out_root /scratch/shared/nfs1/yourfolder --data_path /scratch/shared/nfs1/yourfolder/datasets/MINIIMAGENET/ > out/m4557s.out &


# ------------------------------------------
To print results from run models:
# ------------------------------------------

E.g. ARM CIFAR10
  python -m code.scripts.print_results --root /scratch/shared/nfs1/yourfolder --start 3717
  average val: acc 0.2586 +- 0.0145, forgetting 0.1108 +- 0.0339
  average test: acc 0.2687 +- 0.0107, forgetting 0.1041 +- 0.0367
  