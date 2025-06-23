#
# https://github.com/fungtion/DANN_py3/blob/master/model.py
import torch
import math
import datetime
from asset.base import BaseTrainer, recursive_to_device
from torch.nn.modules.module import Module
from torch.autograd import Function
import torch.nn as nn
import numpy as np

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANNModel(Module):
    def __init__(self, feat_dim: int, n_groups: int):
        super(DANNModel, self).__init__()
        self.regressor = nn.Linear(in_features=feat_dim, out_features= 1)
        self.classifier = nn.Sequential(nn.Linear(feat_dim, n_groups), nn.LogSoftmax(dim=1))

    def forward(self, rep, alpha):
        reverse_feature = ReverseLayerF.apply(rep, alpha)
        # rep.shape = [1024, 130] 1024 => batchsize 
        output = self.regressor(rep)
        group_output = self.classifier(reverse_feature)
        #print(output.shape, group_output.shape)
        return output, group_output

class DANNTrainer(BaseTrainer):
    def __init__(self, args, model, dann_model: DANNModel, optimizer= None, scheduler= None, model_wrapper=None):
        super(DANNTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)
        self.group_criterion = torch.nn.NLLLoss()
        self.dann_model = dann_model
        self.dann_model.to(self.args.device)


    def train(self, train_dataset):
        train_loader = self.prepare_train_loader(train_dataset)
        start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss, total_mse, total_group = 0.0, 0.0, 0.0
            for batch_idx, (x, y, g) in enumerate(train_loader):
                p = float(batch_idx + epoch * len(train_loader)) / (self.args.epochs * len(train_loader))
                alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1
                x, y, g = recursive_to_device((x, y, g), self.args.device)
                self.optimizer.zero_grad()
                # import pdb; pdb.set_trace()
                rep, _ = self.model(x)
                # rep.shape = [1024, 130], alpha = float
                output, group_output = self.dann_model(rep, alpha)

                mse_loss = self.criterion(output, y)
                group_loss = self.group_criterion(group_output, g.squeeze(dim=-1))
                loss = mse_loss + self.args.lambda_dann * group_loss
                # loss = self.args.lambda_dann * group_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_group += group_loss.item()
            print("{}-th Epoch: Train Loss={:.4f}, MSE Loss={:.4f}, Group Loss={:.4f}".format(epoch, total_loss,
                                                                                              total_mse, total_group))
            if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
                self.scheduler.step()
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('DANN Training in %s seconds.' % duration)

    def test(self, test_dataset, dump=False, return_out = False):
        test_loader = self.prepare_test_loader(test_dataset)
        outputs, labels = list(), list()
        self.model.eval()
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                rep, _ = self.model(x)
                output = self.dann_model.regressor(rep)
                outputs.append(output)
                labels.append(y)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        if self.model_wrapper is not None:
            outputs = self.model_wrapper.featurizer.unnormalize_with_log(outputs, None)
            labels = self.model_wrapper.featurizer.unnormalize_with_log(labels, None)
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        if dump:
            print("output",type(outputs), outputs[0], outputs[0].shape)
            print("labels", type(labels), labels[0], labels[0].shape)
            print("errors", type(errors), errors[0], errors[0].shape)
            for o, l, e in zip(outputs, labels, errors):
                print(np.e ** o.item(), np.e ** l.item(), np.e ** e.item())
        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors

    def test_train_dataset(self, train_dataset, return_out=False):
        test_loader = self.prepare_test_loader(train_dataset)
        outputs, labels = list(), list()
        self.model.eval()
        with torch.no_grad():
            for (x, y, g) in test_loader:
                x, y, g = recursive_to_device((x, y, g), self.args.device)
                rep, _ = self.model(x)
                output = self.dann_model.regressor(rep)
                outputs.append(output)
                labels.append(y)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors

