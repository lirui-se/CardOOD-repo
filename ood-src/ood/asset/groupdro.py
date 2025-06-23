# https://github.com/kohpangwei/group_DRO/blob/master/loss.py

import torch
import math
import datetime
from encoder.transform import QueryDataset, JoinQueryDataset
from db.table import Table
from db.schema import DBSchema
from typing import List
from asset.base import BaseTrainer, recursive_to_device

class GroupQueryDataset(QueryDataset):
    def __init__(self, table: Table, queries, cards, query_infos, encoder_type: str = 'dnn'):
        super(GroupQueryDataset, self).__init__(table, queries, cards, query_infos, encoder_type)
        self.n_groups = 0
        self.group_map =dict()
        for query_info in self.query_infos:
            if query_info.num_predicates not in self.group_map.keys():
                self.group_map[query_info.num_predicates] = self.n_groups
                self.n_groups += 1

    def group_count(self):
        group_counts = torch.zeros(size=(self.n_groups,), dtype=torch.float32)
        for query_info in self.query_infos:
            group_counts[self.group_map[query_info.num_predicates]] += 1
        return group_counts

    def __getitem__(self, item):
        pred_list = self.queries[item]
        card = self.cards[item]
        query_info = self.query_infos[item]
        y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(pred_list)
        group_idx = self.group_map[query_info.num_predicates]
        group_idx = torch.LongTensor([group_idx])
        return x, y, group_idx


class GroupJoinQueryDataset(JoinQueryDataset):
    def __init__(self, schema: DBSchema, queries: List, cards: List, query_infos: List, encoder_type: str = 'dnn'):
        super(GroupJoinQueryDataset, self).__init__(schema, queries, cards, query_infos, encoder_type)
        self.n_groups = 0
        self.group_map = dict()
        for query_info in self.query_infos:
            if query_info.num_joins not in self.group_map.keys():
                self.group_map[query_info.num_joins] = self.n_groups
                self.n_groups += 1

    def group_count(self):
        group_counts = torch.zeros(size=(self.n_groups,), dtype=torch.float32)
        for query_info in self.query_infos:
            group_counts[self.group_map[query_info.num_joins]] += 1
        return group_counts

    def __getitem__(self, item):
        table_ids, all_pred_list, join_infos = self.queries[item]
        card = self.cards[item]
        query_info = self.query_infos[item]
        y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(table_ids, all_pred_list, join_infos)
        group_idx = self.group_map[query_info.num_joins]
        group_idx = torch.LongTensor([group_idx])
        return x, y, group_idx

class LossComputer(object):
    def __init__(self, criterion, dataset, gamma=0.1,
                 adj=None, step_size=0.01, normalize_loss = True):
        self.criterion = criterion
        self.gamma = gamma
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_count().cuda()
        self.group_frac = self.group_counts / self.group_counts.sum()
        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.batch_count = 0.

    def _update_stats(self, actual_loss, group_loss, group_count, weights = None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count * ((weights > 0).float())
        self.update_batch_counts += ((group_count * weights) > 0).float()
        self.batch_count += 1

        # avg per-sample
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss

    def _update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def loss(self, output, y, group_idx = None):
        # compute per-sample and per-group losses
        group_idx = group_idx.squeeze(dim = -1)
        per_sample_losses = self.criterion(output, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # update historical losses
        self._update_exp_avg_loss(group_loss, group_count)
        actual_loss, weights = self.compute_robust_loss(group_loss, group_count)

        # update status
        self._update_stats(actual_loss, group_loss, group_count, weights)
        return actual_loss


    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / adjusted_loss.sum()
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        #print('adv_probs',self.adv_probs)
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute the observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count


class GroupDROTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer = None, scheduler = None, model_wrapper=None):
        super(GroupDROTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)
        # overwrite loss function
        self.criterion = torch.nn.MSELoss(reduction='none')

    def train(self, train_dataset):
        loss_computer = LossComputer(self.criterion, train_dataset, self.args.gamma, step_size=self.args.step_size)
        train_loader = self.prepare_train_loader(train_dataset)
        start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (x, y, g) in enumerate(train_loader):
                x, y, g = recursive_to_device((x, y, g), self.args.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_main = loss_computer.loss(output, y, g)
                #print(loss_main)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                total_loss += loss_main.item()
                if (batch_idx + 1) % self.args.log_every == 0:
                    loss_computer.reset_stats()
            if loss_computer.batch_count > 0:
                loss_computer.reset_stats()
            print("{}-th Epoch: Train MSE Loss={:.4f}".format(epoch, total_loss))
            if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
                self.scheduler.step()
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('GroupDRO Training in %s seconds.' % duration)

    def test_train_dataset(self, test_dataset, return_out=False):
        test_loader = self.prepare_test_loader(test_dataset)
        outputs, labels = list(), list()
        self.model.eval()
        with torch.no_grad():
            for (x, y, g) in test_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                output = self.model(x)
                outputs.append(output)
                labels.append(y)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        errors = outputs - labels
        errors = errors.cpu().detach().numpy()
        #print('labels:', torch.isnan(labels).any())
        #print('outputs:', torch.isnan(outputs).any())
        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors
