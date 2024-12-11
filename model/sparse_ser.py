# Copyright 2020-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision.transforms.functional as ttf

from utils.buffer import Buffer
from backbone.sparse_resnet18_global import SparseResNet


def tf_tensor(xs, transforms):
    device = xs.device

    xs = torch.cat([transforms(x).unsqueeze_(0)
                    for x in xs.cpu()], dim=0)

    return xs.to(device=device)


num_classes_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'tinyimg': 200,
    'gcil-cifar100': 100
}


class SparseSER:
    def __init__(self, net: SparseResNet, args):
        super(SparseSER, self).__init__()
        self.net = net
        self.net_old = None
        self.optim = None

        self.args = args
        self.buffer = Buffer(args.buffer_size, args.device)

        # Initialize dropout params
        if 'mnist' in self.args.dataset:
            model = getattr(self, "net")
            print('Initializing Dropout')
            model._keep_probs['layer_1'] = torch.ones(100)
            model._keep_probs['layer_2'] = torch.ones(100)
            model._keep_probs['layer_1_classwise'] = torch.zeros(10, 100)
            model._keep_probs['layer_2_classwise'] = torch.zeros(10, 100)

        else:
            model = getattr(self, "net")
            print('Initializing Dropout')
            model._keep_probs['layer_1'] = torch.zeros(model.nf * 1)
            model._keep_probs['layer_2'] = torch.zeros(model.nf * 2)
            model._keep_probs['layer_3'] = torch.zeros(model.nf * 4)
            model._keep_probs['layer_4'] = torch.zeros(model.nf * 8)
            model._keep_probs['layer_1'][:min(
                int(self.args.kw[0] * self.args.init_active_factor[0] * model.nf), model.nf)] = 1
            model._keep_probs['layer_2'][:min(int(
                self.args.kw[1] * self.args.init_active_factor[1] * model.nf * 2), model.nf * 2)] = 1
            model._keep_probs['layer_3'][:min(int(
                self.args.kw[2] * self.args.init_active_factor[2] * model.nf * 4), model.nf * 4)] = 1
            model._keep_probs['layer_4'][:min(int(
                self.args.kw[3] * self.args.init_active_factor[3] * model.nf * 8), model.nf * 8)] = 1
            model._keep_probs['layer_1_classwise'] = torch.zeros(
                num_classes_dict[args.dataset], model.nf * 1)
            model._keep_probs['layer_2_classwise'] = torch.zeros(
                num_classes_dict[args.dataset], model.nf * 2)
            model._keep_probs['layer_3_classwise'] = torch.zeros(
                num_classes_dict[args.dataset], model.nf * 4)
            model._keep_probs['layer_4_classwise'] = torch.zeros(
                num_classes_dict[args.dataset], model.nf * 8)

    def end_task(self):
        # freeze old model parameters
        self.net_old = deepcopy(self.net)
        self.net_old.eval()
        # Calculate the Dropout keep Probabilities
        model = getattr(self, "net")
        for layer_idx in range(1, len(model._layers) + 1):
            # Heterogeneous Dropout
            activation_counts = model._activation_counts[f'layer_{layer_idx}']
            max_act = torch.max(activation_counts)
            model._keep_probs[f'layer_{layer_idx}'] = torch.exp(
                -activation_counts * self.args.dropout_alpha[layer_idx - 1] / max_act)
            # Classwise Dropout
            activation_counts = model._activation_counts[f'layer_{layer_idx}_classwise']
            max_act = torch.max(activation_counts, dim=1)[0]
            model._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(
                -activation_counts * self.args.classwise_dropout_alpha[layer_idx - 1] / (max_act[:, None] + 1e-16))

    def end_epoch(self, epoch) -> None:
        if epoch > self.args.dropout_warmup:
            model = getattr(self, "net")
            for layer_idx in range(1, len(model._layers) + 1):
                activation_counts = model._activation_counts[f'layer_{layer_idx}_classwise']
                max_act = torch.max(activation_counts, dim=1)[0]
                model._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(
                    -activation_counts * self.args.classwise_dropout_alpha[layer_idx - 1] / (max_act[:, None] + 1e-16))

    def observe(self, inputs, labels):

        self.optim.zero_grad()

        inputs_aug = tf_tensor(inputs, self.args.transform)
        outputs = self.net(inputs_aug, y=labels)
        loss = F.cross_entropy(outputs, labels)

        if self.net_old is not None:
            if self.args.setting == 'domain_il':
                augment = None
            else:
                augment = self.args.transform

            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                inputs.size(0), transform=augment)
            buf_outputs = self.net(buf_inputs, y=buf_labels)
            loss += F.cross_entropy(buf_outputs, buf_labels)

            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            outputs_old = self.net_old(inputs_aug, y=labels)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

        loss.backward()
        self.optim.step()

        if self.args.setting == 'domain_il':
            self.buffer.add_data(examples=inputs_aug,
                                 labels=labels, logits=outputs.data)
        else:
            self.buffer.add_data(
                examples=inputs, labels=labels, logits=outputs.data)

        return loss
