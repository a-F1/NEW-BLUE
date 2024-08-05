# import torch
# from transformers import AutoModelForCausalLM, AutoConfig

# model_path = "/egr/research-optml/chongyu/NEW-BLUE/TOFU/results/locuslab/tofu_ft_llama2-7b/8GPU_npo_grad_diff_1e-05_forget10_epoch10_batch1_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1"
# config = AutoConfig.from_pretrained(model_path)

# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# # 1. 加载模型和分词器
# model_name = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 6. 保存模型和分词器
# model.save_pretrained("/egr/research-optml/chongyu/NEW-BLUE/TOFU")
# tokenizer.save_pretrained("/egr/research-optml/chongyu/NEW-BLUE/TOFU")

model_path = "/egr/research-optml/chongyu/NEW-BLUE/TOFU/"
model = AutoModelForCausalLM.from_pretrained(model_path)
print(model)
