# import torch
# from transformers import AutoModelForCausalLM, AutoConfig

# model_path = "/egr/research-optml/chongyu/NEW-BLUE/TOFU/results/locuslab/tofu_ft_llama2-7b/8GPU_npo_grad_diff_1e-05_forget10_epoch10_batch1_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1"
# config = AutoConfig.from_pretrained(model_path)

# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
