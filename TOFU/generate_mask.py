import os
import sys
from copy import deepcopy
from time import time

sys.path.append("src")
import datasets
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer import is_datasets_available


class GenerateMask(Trainer):
    def __init__(self, score_type, ratios, mask_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_type = score_type
        # self.ratios = ratios

        self.ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.mask_dir = mask_dir
        print(f"self.args.optim in GenerateMask: {self.args.optim}")
        print(f"self.optimizer in GenerateMask: {self.optimizer}")

    def score2mask(self, scores, ratio, return_rank=False):
        sorted_dict_positions = {}
        hard_dict = {}

        threshold_idx = int(len(scores) * ratio)
        positions = torch.argsort(scores)
        ranks = torch.argsort(positions)
        if return_rank:
            return ranks
        start_index = 0
        for key, tensor in self.model.named_parameters():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_idx] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        return hard_dict

    def get_mask(self):
        if self.score_type == "gradient":
            self.gradient()
        elif self.score_type == "retain_gradient":
            self.gradient(dataset="retain")
        elif self.score_type == "forget_gradient":
            self.gradient(dataset="forget")
        else:
            raise ValueError(f"score_type {self.score_type} not supported")
        
        positions = torch.argsort(self.scores)
        ranks = torch.argsort(positions)
        for ratio in self.ratios:
            if os.path.exists(os.path.join(self.mask_dir, f"with_{ratio}.pt")):
                continue
            sorted_dict_positions = {}
            hard_dict = {}

            threshold_idx = int(len(self.scores) * ratio)
            start_index = 0
            for key, tensor in self.model.named_parameters():
                num_elements = tensor.numel()
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_idx] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements
            for key in hard_dict.keys():
                hard_dict[key] = hard_dict[key].type(torch.bool)
            torch.save(hard_dict, os.path.join(self.mask_dir, f"with_{ratio}.pt"))

    def gradient(self, dataset="forget"):
        gradients = {}

        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        # model = self._wrap_model(self.model)
        # # model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        # if model is not self.model:
        #     self.model_wrapped = model

        # print(f"model: {model}")
        # print(f"self.model: {self.model}")
        self.model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {dataset} gradient"):
            inputs = self._prepare_inputs(inputs)

            # for npo
            input_ids, labels, attention_mask = inputs[0] if dataset == "forget" else inputs[1]
            outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)  # attention_mask indicates which tokens to attend to
            loss = outputs.loss

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in self.model.named_parameters():
                    if key not in gradients:
                        gradients[key] = tensor.grad.detach().clone()
                    else:
                        gradients[key] += tensor.grad.detach().clone()

            self.model.zero_grad()

        with torch.no_grad():
            for key, tensor in self.model.named_parameters():
                gradients[key] = -torch.abs(gradients[key])

            self.scores = torch.cat(
                [grad.flatten().cpu() for grad in gradients.values()]
            )