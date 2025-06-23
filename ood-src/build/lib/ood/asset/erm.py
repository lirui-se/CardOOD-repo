import torch
import torch.optim as optim
import datetime
import pdb
from asset.base import BaseTrainer, recursive_to_device

class ERMTrainer(BaseTrainer):
    def __init__(self, args, model, optimizer =None, scheduler = None, model_wrapper=None):
        super(ERMTrainer, self).__init__(args, model, optimizer, scheduler, model_wrapper)

    def train(self, train_dataset):
        train_loader = self.prepare_train_loader(train_dataset)
        start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            self.model.train()
            for (x, y) in train_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                # pdb.set_trace()
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print("{}-th Epoch: Train MSE Loss={:.4f}".format(epoch, total_loss))
            if self.scheduler and (epoch + 1) % self.args.decay_patience == 0:
                self.scheduler.step()
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('ERM Training in %s seconds.' % duration)
