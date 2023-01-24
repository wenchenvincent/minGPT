"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN


class Trainer:
    ## Hook to record activation tensors in the forward pass
    def get_activation(self, name):
        def forward_hook(module, input, output):
            self.input_activations[name] = input[0].detach()
            self.output_activations[name] = output.detach()
            self.weights[name] = module.weight.detach()
            self.biases[name] = module.bias.detach()
        return forward_hook

    ## Hook to record gradient tensors in the backward pass
    def get_gradient(self,name):
        def backward_hook(module, grad_input, grad_output):
            ## grad_output and grad_input are tuples
            ## grad_input are gradients wrt inputs of the layer
            ## grad_output are gradients wrt outputs of the layer
            self.output_gradients[name] = grad_output[0].detach()
            self.input_gradients[name] = grad_input[0].detach()
            self.weight_gradients[name] = module.weight.grad.detach()
            self.bias_gradients[name] = module.bias.grad.detach()
        return backward_hook

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        self.input_activations = {}
        self.output_activations = {}
        self.weights = {}
        self.biases = {}
        self.output_gradients = {}
        self.input_gradients = {}
        self.weight_gradients = {}
        self.bias_gradients = {}

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        model.transformer.h[0].mlp.c_fc.register_forward_hook(self.get_activation('transformer.mlp.c_fc'))
        model.transformer.h[0].mlp.c_fc.register_full_backward_hook(self.get_gradient('transformer.mlp.c_fc'))
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()


            torch.save(self.input_activations['transformer.mlp.c_fc'], 'input.pt')
            torch.save(self.weights['transformer.mlp.c_fc'], 'weight.pt')
            torch.save(self.biases['transformer.mlp.c_fc'], 'bias.pt')
            torch.save(self.output_activations['transformer.mlp.c_fc'], 'output.pt')

            torch.save(self.output_gradients['transformer.mlp.c_fc'], 'output_grad.pt')
            torch.save(self.input_gradients['transformer.mlp.c_fc'], 'input_grad.pt')
            torch.save(self.weight_gradients['transformer.mlp.c_fc'], 'weight_grad.pt')
            torch.save(self.bias_gradients['transformer.mlp.c_fc'], 'bias_grad.pt')
            sys.exit(0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
