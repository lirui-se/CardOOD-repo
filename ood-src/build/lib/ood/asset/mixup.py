#https://github.com/huaxiuyao/C-Mixup/blob/main/src/algorithm.py
from asset.base import BaseTrainer, recursive_to_device
import numpy as np
import torch
import datetime
from sklearn.neighbors import KernelDensity

class MixUpTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer = None, scheduler = None, model_wrapper=None):
        super(MixUpTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)
    def get_card(self, train_dataset):
        if self.model_wrapper is not None:
            cards = train_dataset.Y
        else:
            cards = torch.tensor(train_dataset.cards, dtype=torch.float32).unsqueeze(dim = -1)
        return cards

    def mixup_rate_sample(self, train_dataset):
        n = len(train_dataset)
        cards = self.get_card(train_dataset)
        mix_idx = list()
        for i in range(n):
            card_i = cards[i]
            card_i = card_i.view(-1, 1)

            kd = KernelDensity(kernel=self.args.kde_type, bandwidth=self.args.kde_bandwidth).fit(card_i)
            each_rate = np.exp(kd.score_samples(cards))
            #print('each rate:', each_rate.shape)
            each_rate /= np.sum(each_rate)
            #print('each rate:', each_rate.shape)
            mix_idx.append(each_rate)
        mix_idx = np.array(mix_idx)
        return mix_idx

    def train(self, train_dataset):
        mix_idx_sample_rate = self.mixup_rate_sample(train_dataset)
        print(mix_idx_sample_rate.shape)
        train_loader = self.prepare_train_loader(train_dataset)
        collate_fn = train_loader.collate_fn
        iteration = len(train_dataset) // self.args.batch_size
        start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            shuffle_idx = np.random.permutation(np.arange(len(train_dataset)))

            total_loss = 0.0
            self.model.train()
            for idx in range(iteration):
                lambd = np.random.beta(self.args.mix_alpha, self.args.mix_alpha)
                idx_1 = shuffle_idx[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
                idx_2 = np.array([np.random.choice(np.arange(len(train_dataset)), p=mix_idx_sample_rate[sel_idx]) for sel_idx in idx_1])
                #print(len(idx_1), len(idx_2))
                """
                x_1, y_1 = collate_fn([train_dataset[i] for i in idx_1])
                x_2, y_2 = collate_fn([train_dataset[i] for i in idx_2])
                x_1, y_1 = x_1.to(self.args.device), y_1.to(self.args.device)
                x_2, y_2 = x_2.to(self.args.device), y_2.to(self.args.device)
                print(x_1.shape, x_2.shape)
                mixup_x = x_1 * lambd + x_2 * (1 - lambd)
                mixup_y = y_1 * lambd + y_2 * (1 - lambd)
                """
                x, y = collate_fn([train_dataset[i] for i in idx_1] + [train_dataset[i] for i in idx_2])
                x, y = recursive_to_device((x, y), self.args.device)
                mixup_x, mixup_y = self._recursive_mixup((x, y), self.args.batch_size, lambd)

                self.optimizer.zero_grad()
                output = self.model(mixup_x)
                loss = self.criterion(output, mixup_y)
                #print(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print("{}-th Epoch: Train MSE Loss={:.4f}".format(epoch, total_loss))
            if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
                self.scheduler.step()
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('Mixup Training in %s seconds.' % duration)

    def _recursive_mixup(self, x, batch_size, lambd):
        if isinstance(x, torch.Tensor):
            return x[:batch_size] * lambd + x[batch_size:] * (1 - lambd)
        if isinstance(x, dict):
            return {k: self._recursive_mixup(v, batch_size, lambd) for k, v in x.items()}
        if isinstance(x, list):
            return [self._recursive_mixup(v, batch_size, lambd) for v in x]
        if isinstance(x, tuple):
            return (self._recursive_mixup(v, batch_size, lambd) for v in x)
        else:
            raise NotImplementedError('Unsupported input type {} to device'.format(type(x)))



