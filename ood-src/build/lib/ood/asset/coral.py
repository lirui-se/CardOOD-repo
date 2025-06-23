# "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
# https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py
import datetime
import random

import torch
import numpy as np
from asset.base import BaseTrainer, recursive_to_device
from typing import List

class DeepCoralTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer = None, scheduler = None, model_wrapper=None):
        super(DeepCoralTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)

    def compute_covariance(self, input_data):
        n = input_data.size(0) # batch size
        id_row = torch.ones(n).view(1, n).to(self.args.device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
        return c

    def coral(self, src_rep, trg_rep):
        d = src_rep.size(1)
        src_c = self.compute_covariance(src_rep)
        trg_c = self.compute_covariance(trg_rep)
        loss = torch.sum(torch.mul((src_c - trg_c), (src_c - trg_c)))
        return loss / (4 * d * d)

    def loader_sample_rate(self, train_datasets: List):
        num_instances = [len(train_dataset) for train_dataset in train_datasets]
        p = np.array(num_instances, dtype=np.float32)
        return p / np.sum(p)


    def train(self, train_datasets: List):
        train_loaders = [ self.prepare_train_loader(train_dataset) for train_dataset in train_datasets]
        p = self.loader_sample_rate(train_datasets)
        min_n_batches = min([len(train_loader) for train_loader in train_loaders])
        print("loader sample rate:", p)
        start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            total_loss, total_mse, total_coral = 0.0, 0.0, 0.0
            self.model.train()
            for _ in range(int(min_n_batches * len(train_loaders) * (len(train_loaders) - 1) / 2)):
                # sampled_loaders = random.sample(train_loaders, k = 2)
                # import pdb; pdb.set_trace()
                loader_idx = np.random.choice(list(range(len(train_datasets))), size = 2, replace = False, p = p)
                src_loader, trg_loader = train_loaders[loader_idx[0]], train_loaders[loader_idx[1]]
                src_x, src_y = next(iter(src_loader))
                trg_x, trg_y = next(iter(trg_loader))
                src_x, src_y = recursive_to_device((src_x, src_y), self.args.device)
                trg_x, trg_y = recursive_to_device((trg_x, trg_y), self.args.device)
                self.optimizer.zero_grad()
                src_rep, src_output = self.model(src_x)
                trg_rep, trg_output = self.model(trg_x)
                mse_loss = self.criterion(src_output, src_y) + self.criterion(trg_output, trg_y)
                coral_loss = self.coral(src_rep, trg_rep)
                loss = mse_loss + self.args.lambda_coral * coral_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_coral += coral_loss.item()
            print("{}-th Epoch: CORAL Training Loss={:.4f}, MSE Loss={:.4f}, CORAL Loss={:.4f}"
                  .format(epoch, total_loss, total_mse, total_coral))
            #if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
            #    self.scheduler.step()
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('Deep Coral Training in %s seconds.' % duration)


    def test(self, test_dataset, return_out=False):
        test_loader = self.prepare_test_loader(test_dataset)
        outputs, labels = list(), list()
        self.model.eval()
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                rep, output = self.model(x)
                outputs.append(output)
                labels.append(y)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        if self.model_wrapper is not None:
            outputs = self.model_wrapper.featurizer.unnormalize_with_log(outputs, None)
            labels = self.model_wrapper.featurizer.unnormalize_with_log(labels, None)
        # import pdb; pdb.set_trace()
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors

