import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from encoder.collator import PaddingCollate, PaddingCollateTypeA, PaddingCollateTypeB
import numpy as np


import abc


DEBUG_MODE=True

def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return (recursive_to_device(v, device) for v in x)
    else:
        raise NotImplementedError('Unsupported input type {} to device'.format(type(x)))


class BaseTrainer(object):
    def __init__(self, args, model, optimizer = None, scheduler = None, model_wrapper=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(),
                                    lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = scheduler if scheduler else optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.decay_factor)
        self.model_wrapper = model_wrapper
        self.criterion = torch.nn.MSELoss()
        self.model.to(self.args.device)

    def prepare_train_loader(self, train_dataset) -> DataLoader:
        if self.args.model_type == 'MSCN':
            print("For debug: Training using function ", PaddingCollate(), " as func.")
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=PaddingCollate())
        elif self.args.model_type == "CEB_MSCN":
            print("For debug: Training using function ", self.model_wrapper.collate_fn, " as func.")
            train_loader = self.model_wrapper.get_trainloader(train_dataset)
        else:
            print("For debug: Training does not use function ", PaddingCollate(), " as func.")
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        return train_loader

    def prepare_test_loader(self, test_dataset) -> DataLoader:
        assert False, "Should not enter this function."
        if self.args.model_type == 'MSCN':
            print("For debug: Using function ", PaddingCollate(), " as func.")
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=PaddingCollate())
        elif self.args.model_type == 'CEB_MSCN':
            print("For debug: Using function ", self.model_wrapper.collate_fn, " as func.")
            test_loader = self.model_wrapper.get_testloader(test_dataset)
        else:
            print("For debug: Not Using function ", PaddingCollate(), " as func.")
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        return test_loader

    def prepare_test_loader(self, test_dataset, padding_func_type=None, batch_size=1000) -> DataLoader:
        if self.args.model_type == 'MSCN':
            if padding_func_type is None:
                if DEBUG_MODE:
                    print("For debug: Using function ", PaddingCollate(), " as func.")
                test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=PaddingCollate())
            elif padding_func_type == 'TypeA':
                if DEBUG_MODE:
                    print("For debug: Using function A", PaddingCollate(), " as func.")
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PaddingCollateTypeA())
            elif padding_func_type == 'TypeB':
                if DEBUG_MODE:
                    print("For debug: Using function B", PaddingCollate(), " as func.")
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PaddingCollateTypeB())

            else:
                assert False, "Unsupported padding_func type! " + padding_func_type
        elif self.args.model_type == 'CEB_MSCN':
            print("For debug: Using function ", self.model_wrapper.collate_fn, " as func.")
            test_loader = self.model_wrapper.get_testloader(test_dataset)
        else:
            if DEBUG_MODE:
                print("For debug: Not Using function ", PaddingCollate(), " as func.")
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        return test_loader


    @abc.abstractmethod
    def train(self, train_dataset):
        return


    def test(self, test_dataset, dump=False, return_out =False):
        test_loader = self.prepare_test_loader(test_dataset)
        outputs, labels = list(), list()
        self.model.eval()
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                output = self.model(x)
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
        if dump:
            print("output",type(outputs), outputs[0], outputs[0].shape)
            print("labels", type(labels), labels[0], labels[0].shape)
            print("errors", type(errors), errors[0], errors[0].shape)
            for o, l, e in zip(outputs, labels, errors):
                print(np.e ** o.item(), np.e ** l.item(), np.e ** e.item())
        #print('labels:', torch.isnan(labels).any())
        #print('outputs:', torch.isnan(outputs).any())
        if return_out:
            outputs = outputs.cpu().detach().numpy()
            return errors, outputs
        return errors
