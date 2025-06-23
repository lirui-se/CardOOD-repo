# Code base: https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
import torch
import torch.optim as optim
import torch.autograd as autograd
import datetime
from typing import List
from asset.base import BaseTrainer, recursive_to_device


class IRMTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer= None, scheduler = None, model_wrapper=None):
        super(IRMTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)


    def penalty(self, output, y):
        scale = torch.tensor(1.0).to(self.args.device).requires_grad_()
        loss = self.criterion(output * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def train(self, train_datasets: List):
        train_loaders = [ self.prepare_train_loader(train_dataset) for train_dataset in train_datasets]
        min_n_batches = min([len(train_loader) for train_loader in train_loaders])
        start = datetime.datetime.now()
        for step in range(self.args.epochs * min_n_batches):
            all_mse, all_penalty = list(), list()
            self.model.train()
            for train_loader in train_loaders:
                x, y = next(iter(train_loader))
                x, y = recursive_to_device((x, y), self.args.device)
                output = self.model(x)
                #print(self.criterion(output, y).shape, self.penalty(output, y).shape)
                all_mse.append(self.criterion(output, y).unsqueeze(-1))
                all_penalty.append(self.penalty(output, y).unsqueeze(-1))
            mse = torch.cat(all_mse, dim=0).mean()
            penalty = torch.cat(all_penalty, dim=0).mean()
            loss = mse.clone()
            penalty_weight = (self.args.penalty_weight
                              if step >= self.args.penalty_anneal_iters else 1.0)
            loss += penalty * penalty_weight
            if penalty_weight > 1.0:
                # rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if step % 100 == 0:
                print("{}-th Step: IRM Training Loss={:.4f}".format(step, loss.item()))
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('ERM Training in %s seconds.' % duration)
