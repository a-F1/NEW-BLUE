# BLUE
```bash
conda activate tofu
cd TOFU

# ft
model=phi
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 finetune.py --config-name=finetune.yaml split=full batch_size=1 gradient_accumulation_steps=4 model_family=llama2-7b

# npo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 evaluate_util.py model_path=results/locuslab/tofu_ft_llama2-7b/8GPU_grad_diff_1e-05_forget10_epoch10_batch1_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 evaluate_util.py model_path=results/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/8GPU_npo_grad_diff_1e-05_forget01_epoch10_batch4_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1

# grad_diff
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=grad_diff

# idk
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=idk

# vector
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 vector_forget.py --config-name=vector_forget.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=18765 vector_forget.py --config-name=vector_forget.yaml

# bash
bash /egr/research-optml/chongyu/NEW-BLUE/TOFU/commands/run0.sb&

# debug
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=grad_diff save_dir=try
```