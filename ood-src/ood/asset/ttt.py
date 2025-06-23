#Code base: https://github.com/yueatsprograms/ttt_cifar_release/blob/master/utils/test_helpers.py

import torch
import torch.optim as optim
import datetime
from torch.nn.modules.module import Module
from asset.base import BaseTrainer, recursive_to_device
from typing import List
import copy
import numpy as np
class OrderCriterion(Module):
    # Order embedding loss for self-supervised learning
    def __init__(self):
        super(OrderCriterion, self).__init__()

    def forward(self, rep, neg_rep):
        # rep: [N, d], neg_rep [N, num_neg, d]
        # import pdb; pdb.set_trace()
        rep = rep[:, None, :].expand_as(neg_rep) # [batch_size, num_negs, feat_dim]
        loss = torch.sum(torch.max(torch.zeros_like(neg_rep), neg_rep - rep)**2)
        return loss



class TestTimeTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer= None, scheduler=None, model_wrapper=None):
        super(TestTimeTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)
        self.mse_criterion = self.criterion
        self.order_criterion = OrderCriterion()

    def train(self, train_dataset, dump=False, dump_card_distri=False):
        train_loader = self.prepare_train_loader(train_dataset)
        # import pdb; pdb.set_trace()
        start = datetime.datetime.now()
        labels = list()
        if dump:
            dump_ever = False
        for epoch in range(self.args.epochs):
            total_loss, total_mse, total_order = 0.0, 0.0, 0.0
            self.model.train()
            if dump and not dump_ever:
                xs = list()
            for (x, y, x_neg) in train_loader:
                x, y, x_neg = recursive_to_device((x, y, x_neg), self.args.device)
                # for regular ttt:
                # x = ( [32, 6, 6], [32, 22, 20], [32, 15, 15] )
                # x_neg = ( [32, 6, 10, 6], [32, 22, 10, 20], [32, 15, 10, 15] )
                # 
                # for CEB_MCSN:
                # x = { "table": [1024, 12, 18], "join": [1024, 13, 47], ... }
                # x_neg = { "table": [ [1024, 12, 18], [1024, 12, 18], ... ], ... }
                # 
                # after processing:
                # x = { "table": [1024, 12, 18], "join": [1024, 13, 47], ... }
                # x_neg = { "table": [1024, 12, 3, 18], "join": [1024, 13, 3, 47], ... }
                if dump_card_distri:
                    labels.append(y)
                self.optimizer.zero_grad()
                # import pdb; pdb.set_trace()
                # eee = self.model(x)
                rep, output = self.model(x)
                neg_rep, _ = self.model(x_neg)
                mse_loss = self.mse_criterion(output, y)
                order_loss = self.order_criterion(rep, neg_rep)
                #print(x)
                #print(x_neg)
                #print(rep)
                #print(neg_rep)
                #print(order_loss)
                loss = mse_loss + order_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_order += order_loss.item()
                if dump and not dump_ever:
                    xs.append(x)
            print("{}-th Epoch: Train Loss={:.4f}, MSE Loss={:.4f}, Order Loss={:.4f}".format(epoch, total_loss, total_mse, total_order))
            if dump:
                dump_ever = True
            if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
                self.scheduler.step()
        end = datetime.datetime.now()
        if dump_card_distri and self.model_wrapper is not None:
            labels = torch.cat(labels, dim=0)
            labels = self.model_wrapper.featurizer.unnormalize_with_log(labels, None)
            labels = np.power(labels, np.e)
            counts, bin_edges = np.histogram(labels, bins=30, density=True)
            print(" ==== histogram ==== ")
            print("counts: ", counts)
            print("bin_edges: ", bin_edges)
        duration = (end - start).total_seconds()
        print('TTT Training in %s seconds.' % duration)
        if dump:
            xs = list(zip(*xs))
            for xs_list in xs:
                for e in xs_list:
                    print(e.shape)


    def top_10_frequent_numbers(self, arr):
        d = {}
        for a in arr:
            if a not in d.keys():
                d[a] =0
            d[a] += 1
        t = []
        for k in d.keys():
            t.append( (k, d[k]) )
        t.sort(reverse=True,key=lambda x:x[1])
        return t[0:50]


    def test(self, test_dataset, return_out = False, dump=False, dump_card_distri=False, filename="", debug_file="", debug_table_num=-1, query_infos=[], query_str = []):
        test_loader = self.prepare_test_loader(test_dataset, padding_func_type="TypeA")
        # import pdb; pdb.set_trace()
        outputs, labels = list(), list()
        self.model.eval()
        xs = list()
        # print(" ==== for test ==== ")
        with torch.no_grad():
            for (x, y, x_neg) in test_loader:
                # x = tuple([1000, 5, 6], [1000, 18, 20], [1000, 15, 15])
                # x = tuple([1000, 5, 3, 6], [1000, 18, 3, 20], [1000, 15, 3, 15])
                x, y, x_neg = recursive_to_device((x, y, x_neg), self.args.device)
                rep, output = self.model(x)
                outputs.append(output)
                # print(x["table"][0], x["join"][0], x["pred"][0])
                # o = output.cpu().detach().numpy()
                # print("predict: ", o)
                labels.append(y)
                if dump:
                    xs.append(x)
            # xs = list(zip(*xs))
            # for xs_list in xs:
            #     for e in xs_list:
            #         print(e.shape)
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
            # for o, l, e in zip(outputs, labels, errors):
            #     print(np.e ** o.item(), np.e ** l.item(), np.e ** e.item())
        if dump_card_distri:
            labels = np.exp(labels)
            
            # 指定 bin 边界
            bin_edges = np.linspace(1, 100000, 41).tolist() + [np.inf]
            
            # 计算直方图
            counts, _ = np.histogram(labels, bins=bin_edges, density=False)
            
            print(" ==== histogram ==== ")
            print("counts: ", counts)
            print("bin_edges: ", bin_edges)
            result = self.top_10_frequent_numbers(labels)
            print("top 50: ", result)

        # import pdb; pdb.set_trace()

        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors

    def test_adapt(self, test_dataset):
        test_loader = self.prepare_test_loader(test_dataset)
        outputs, labels = list(), list()
        for (x, y, x_neg) in test_loader:
            x, y, x_neg = recursive_to_device((x, y, x_neg), self.args.device)
            # test time train for adaption
            model = copy.deepcopy(self.model).to(self.args.device)
            model.train()
            optimizer =  optim.Adam(model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.weight_decay)
            optimizer.zero_grad()
            rep, _ = model(x)
            neg_rep, _ = model(x_neg)
            loss = self.order_criterion(rep, neg_rep)
            loss.backward()
            optimizer.step()
            # test
            model.eval()
            with torch.no_grad():
                _, output = model(x)
                outputs.append(output)
                labels.append(y)
            del model
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        return errors


    def test_adapt_group(self, test_datasets: List):
        test_loaders = [self.prepare_test_loader(test_dataset) for test_dataset in test_datasets]
        outputs, labels = list(), list()
        for test_loader in test_loaders:
            for (x, y, x_neg) in test_loader:
                x, y, x_neg = recursive_to_device((x, y, x_neg), self.args.device)
                # test time train for adaption
                model = copy.deepcopy(self.model).to(self.args.device)
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.weight_decay)
                optimizer.zero_grad()
                rep, _ = model(x)
                neg_rep, _ = model(x_neg)
                loss = self.order_criterion(rep, neg_rep)
                loss.backward()
                optimizer.step()
                # test
                model.eval()
                with torch.no_grad():
                    _, output = model(x)
                    outputs.append(output)
                    labels.append(y)
                del model
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        return errors
