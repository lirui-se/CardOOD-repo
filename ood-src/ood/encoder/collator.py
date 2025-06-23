import torch
from typing import List
from torch.utils.data._utils.collate import default_collate
DEBUG_MODE=False

def pretty_print(data_list, depth):
    if isinstance(data_list, list):
        print((" " * depth * 4) + "list: len = ", len(data_list))
        pretty_print(data_list[0], depth + 1)
    elif isinstance(data_list, tuple):
        print((" " * depth * 4) + "tuple: ele num = ", len(data_list))
        for e in data_list:
            pretty_print(e, depth + 1)
    elif isinstance(data_list, torch.Tensor):
        print((" " * depth * 4), data_list.shape)

class PaddingCollateOrigin(object):
    def __init__(self):
        super().__init__()

    def _pad_one(self, x, n, value =0.0):
        assert x.size(0) <= n, "actual size: {}, pad size: {}".format(x.size(0), n)
        if x.size(0) == n:
            return x
        if DEBUG_MODE:
            pass
            # print("actual size: {}, pad size: {}".format(x.size(0), n))
        pad_size = [n -  x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size, fill_value = value).to(x)
        return torch.cat([x, pad], dim= 0)

    def _pad_data_list(self, x_list):
        if isinstance(x_list[0], torch.Tensor):
            max_length = max(x.size(0) for x in x_list)
            x_list_padded = [self._pad_one(x, max_length) for x in x_list]
        elif isinstance(x_list[0], tuple): # x_list is a list of tuple
            x_data_list = list(zip(*x_list))
            x_list_padded = [self._pad_data_list(x_list) for x_list in x_data_list]
            x_list_padded = list(zip(*x_list_padded))
        else:
            raise NotImplementedError("Can not support x_list type: {}, x_list[0] type: {}".format(type(x_list), type(x_list[0])))
        return x_list_padded

    def __call__(self, data_list: List): # [(t1, t2), ]
        if DEBUG_MODE:
            print("Padding Type Base checkpoint 1. dump data_list.")
            pretty_print(data_list, 0)
        data_list = list(zip(*data_list))
        data_list_padded = [self._pad_data_list(x_list) for x_list in data_list]
        data_list_padded = list(zip(*data_list_padded))
        return default_collate(data_list_padded)


class PaddingCollate(object):
    def __init__(self):
        super().__init__()

    def _pad_one(self, x, n, value =0.0):
        assert x.size(0) <= n, "actual size: {}, pad size: {}".format(x.size(0), n)
        if x.size(0) == n:
            return x
        if DEBUG_MODE:
            pass
            # print("actual size: {}, pad size: {}".format(x.size(0), n))
        pad_size = [n -  x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size, fill_value = value).to(x)
        return torch.cat([x, pad], dim= 0)

    def _pad_data_list(self, x_list, xl=None):
        if isinstance(x_list[0], torch.Tensor):
            if xl is not None:
                max_length = xl
            else:
                max_length = max(x.size(0) for x in x_list)
            x_list_padded = [self._pad_one(x, max_length) for x in x_list]
            if DEBUG_MODE:
                for x in x_list:
                    print(x.shape)
                    print(x)
        elif isinstance(x_list[0], tuple): # x_list is a list of tuple
            assert len(x_list[0]) == 3
            assert isinstance(x_list[0][0], torch.Tensor)
            x_data_list = list(zip(*x_list))
            x_len = (6, 22, 15)
            x_list_padded = []
            for xl, x_list in zip(x_len, x_data_list):
                x_list_padded.append(self._pad_data_list(x_list, xl))
            # x_list_padded = [self._pad_data_list(x_list) for x_list in x_data_list]
            x_list_padded = list(zip(*x_list_padded))
        else:
            raise NotImplementedError("Can not support x_list type: {}, x_list[0] type: {}".format(type(x_list), type(x_list[0])))
        return x_list_padded

    def __call__(self, data_list: List): # [(t1, t2), ]
        if DEBUG_MODE:
            print("Padding Type Base checkpoint 1. dump data_list.")
            pretty_print(data_list, 0)
        data_list = list(zip(*data_list))
        data_list_padded = [self._pad_data_list(x_list) for x_list in data_list]
        data_list_padded = list(zip(*data_list_padded))
        return default_collate(data_list_padded)

class PaddingCollateTypeA(object):
    def __init__(self):
        super().__init__()

    def _pad_one(self, x, n, value =0.0):
        assert x.size(0) <= n, "actual size: {}, pad size: {}".format(x.size(0), n)
        if x.size(0) == n:
            return x
        pad_size = [n -  x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size, fill_value = value).to(x)
        return torch.cat([x, pad], dim= 0)

    def _pad_data_list(self, x_list, size_list=None):
        if isinstance(x_list[0], torch.Tensor):
            if size_list is None:
                max_length = max(x.size(0) for x in x_list)
                x_list_padded = [self._pad_one(x, max_length) for x in x_list]
            else:
                x_list_padded = [self._pad_one(x, size) for x, size in zip(x_list, size_list)]
        elif isinstance(x_list[0], tuple): # x_list is a list of tuple
            assert len(x_list[0]) == 3
            assert isinstance(x_list[0][0], torch.Tensor)
            x_list_padded = []
            x_len_dict = { 1: (1, 6, 15), 2: (2, 10, 15), 3: (3, 14, 15), 4: (4, 16, 15) , 5: (5, 18, 15) }
            x_len_list = [ x_len_dict[ e[0].shape[0] ] for e in x_list ]
            x_data_list = list(zip(*x_list))
            x_len_list = list(zip(*x_len_list))
            for xl, x_list in zip(x_len_list, x_data_list):
                x_list_padded.append(self._pad_data_list(x_list, xl))
            # x_list_padded = [self._pad_data_list(x_list) for x_list in x_data_list]
            x_list_padded = list(zip(*x_list_padded))
        else:
            raise NotImplementedError("Can not support x_list type: {}, x_list[0] type: {}".format(type(x_list), type(x_list[0])))
        return x_list_padded

    def __call__(self, data_list: List): # [(t1, t2), ]
        if DEBUG_MODE:
            print("Padding Type A checkpoint 1. dump data_list.")
            # pretty_print(data_list, 0)
            if len(data_list) < 200:
                print("data_list: len = ", len(data_list))
                for e in data_list:
                    print("(", e[0][0].shape, e[0][1].shape, e[0][2].shape, ")", e[1].shape, "(", e[2][0].shape, e[2][1].shape, e[2][2].shape , ")")
        data_list = list(zip(*data_list))
        data_list_padded = [self._pad_data_list(x_list) for x_list in data_list]
        data_list_padded = list(zip(*data_list_padded))
        if DEBUG_MODE:
            print("Padding Type A checkpoint 2. dump data_list.")
            # pretty_print(data_list, 0)
            if len(data_list_padded) < 200:
                print("data_list: len = ", len(data_list_padded))
                for e in data_list_padded:
                    print("(", e[0][0].shape, e[0][1].shape, e[0][2].shape, ")", e[1].shape, "(", e[2][0].shape, e[2][1].shape, e[2][2].shape , ")")

        return default_collate(data_list_padded)


class PaddingCollateTypeB(object):
    def __init__(self):
        super().__init__()

    def _pad_one(self, x, n, value =0.0):
        assert x.size(0) <= n, "actual size: {}, pad size: {}".format(x.size(0), n)
        if x.size(0) == n:
            return x
        pad_size = [n -  x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size, fill_value = value).to(x)
        return torch.cat([x, pad], dim= 0)

    def _pad_data_list(self, x_list, xl=None):
        if isinstance(x_list[0], torch.Tensor):
            if xl is None:
                max_length = max(x.size(0) for x in x_list)
            else:
                max_length = xl
            x_list_padded = [self._pad_one(x, max_length) for x in x_list]
        elif isinstance(x_list[0], tuple): # x_list is a list of tuple
            assert len(x_list[0]) == 3
            assert isinstance(x_list[0][0], torch.Tensor)
            x_data_list = list(zip(*x_list))
            x_list_padded = []
            x_len = (5, 18, 15)
            for xl, x_list in zip(x_len, x_data_list):
                x_list_padded.append(self._pad_data_list(x_list, xl))
            # x_list_padded = [self._pad_data_list(x_list) for x_list in x_data_list]
            x_list_padded = list(zip(*x_list_padded))
        else:
            raise NotImplementedError("Can not support x_list type: {}, x_list[0] type: {}".format(type(x_list), type(x_list[0])))
        return x_list_padded

    def __call__(self, data_list: List): # [(t1, t2), ]
        if DEBUG_MODE:
            print("Padding Type B checkpoint 1. dump data_list.")
            # pretty_print(data_list, 0)
            if len(data_list) < 200:
                print("data_list: len = ", len(data_list))
                for e in data_list:
                    print("(", e[0][0].shape, e[0][1].shape, e[0][2].shape, ")", e[1].shape, "(", e[2][0].shape, e[2][1].shape, e[2][2].shape , ")")
        data_list = list(zip(*data_list))
        data_list_padded = [self._pad_data_list(x_list) for x_list in data_list]
        data_list_padded = list(zip(*data_list_padded))
        if DEBUG_MODE:
            print("Padding Type B checkpoint 2. dump data_list.")
            # pretty_print(data_list, 0)
            if len(data_list_padded) < 200:
                print("data_list: len = ", len(data_list_padded))
                for e in data_list_padded:
                    print("(", e[0][0].shape, e[0][1].shape, e[0][2].shape, ")", e[1].shape, "(", e[2][0].shape, e[2][1].shape, e[2][2].shape , ")")

        return default_collate(data_list_padded)
