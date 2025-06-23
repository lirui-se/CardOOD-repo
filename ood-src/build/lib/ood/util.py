import numpy as np
import torch
import random
import math
from typing import List

def seed_all(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    print("set all seed!")


def train_test_val_split(all_queries, all_cards, train_frac=0.6, test_frac=0.2, all_query_infos= None):
    # default split train/test/val: 6/2/2
    assert 0 < train_frac + test_frac <= 1.0, "Error in the train/test split fraction"
    num_instances = len(all_queries)
    num_train, num_test = int(train_frac * num_instances), int(test_frac * num_instances)
    num_val = num_instances - num_train - num_test

    tmp = list(zip(all_queries, all_cards, all_query_infos))
    random.shuffle(tmp)
    all_queries, all_cards, all_query_infos = zip(*tmp)
    train_queries, train_cards = all_queries[:num_train], all_cards[:num_train]
    test_queries, test_cards = all_queries[num_train: num_train + num_test], all_cards[num_train: num_train + num_test]
    val_queries = all_queries[num_train + num_test :] if num_val > 0 else None
    val_cards = all_cards[num_train + num_test :] if num_val > 0 else None
    train_query_infos = all_query_infos[:num_train] if all_query_infos else None
    test_query_infos = all_query_infos[num_train : num_train + num_test] if all_query_infos else None
    val_query_infos = all_query_infos[num_train + num_test:] if all_query_infos and num_val > 0 else None
    return train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos, val_queries, val_cards, val_query_infos


def uneven_train_test_split(all_queries, all_cards, all_query_infos, skew_split_keys: str, train_frac = 0.6, skew_ratio =0.5):
    # split train/test data by train_frac, then unevenly split the train data by attributes in skew_split_keys
    pred_stat = PredictionStatistics()
    partition_query_indices = pred_stat.get_partitioned_indices(all_query_infos, part_keys=skew_split_keys)
    num_partition_set = len(partition_query_indices)

    train_queries, train_cards, train_query_infos = list(), list(), list()
    test_queries, test_cards, test_query_infos = list(), list(), list()
    # split the train/test data by train_frac
    for key in sorted(partition_query_indices.keys()):
        random.shuffle(partition_query_indices[key])
        num_train = int(len(partition_query_indices[key]) * train_frac)

        partition_test_queries = [all_queries[idx] for idx in partition_query_indices[key][num_train:]]
        partition_test_cards = [all_cards[idx] for idx in partition_query_indices[key][num_train:]]
        partition_test_query_infos = [all_query_infos[idx] for idx in partition_query_indices[key][num_train:]]
        test_queries += partition_test_queries
        test_cards += partition_test_cards
        test_query_infos += partition_test_query_infos
    # make the train data uneven distribution
    for i, key in enumerate(sorted(partition_query_indices.keys())):
        if i < int(num_partition_set / 2):
            select_ratio = skew_ratio
        elif num_partition_set % 2 == 1 and i == int(num_partition_set / 2):
            select_ratio = 0.5
        else:
            select_ratio = 1.0 - skew_ratio
        num_train = int(len(partition_query_indices[key]) * select_ratio)
        print(key, num_train, select_ratio)
        partition_train_queries = [all_queries[idx] for idx in partition_query_indices[key][:num_train]]
        partition_train_cards = [all_cards[idx] for idx in partition_query_indices[key][:num_train]]
        partition_train_query_infos = [all_query_infos[idx] for idx in partition_query_indices[key][:num_train]]
        train_queries += partition_train_queries
        train_cards += partition_train_cards
        train_query_infos += partition_train_query_infos
    return train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos

# def query_pattern_train_test_split(all_queries, all_cards, all_query_infos, pattern_indices, skew_split_keys='query_pattern', train_frac = 0.6, skew_ratio=1):
#     pred_stat = PredictionStatistics()
#     partition_query_indices = pred_stat.get_partitioned_indices_by_query_pattern(all_query_infos, pattern_indices=pattern_indices, part_keys=skew_split_keys)
#     num_partition_set = len(partition_query_indices)
#     train_queries, train_cards, train_query_infos = list(), list(), list()
#     test_queries, test_cards, test_query_infos = list(), list(), list()
#     # Only one key. 
#     keys = [ k for k  in partition_query_indices.keys() ]
#     random.shuffle(keys)
#     num_train = int(len(keys) * train_frac)
#     train_keys = keys[:num_train]
#     test_keys = keys[num_train:]
#     # for key in train_keys:
#     #     train_queries += [ all_queries[idx] for idx in partition_query_indices[key] ]
#     #     train_cards += [ all_cards[idx] for idx in partition_query_indices[key]]
#     #     train_query_info += [ all_query_infos[idx] for idx in partition_query_indices[key] ]
#     train_queries, train_cards, train_query_infos = list(), list(), list()
#     test_queries, test_cards, test_query_infos = list(), list(), list()
#     for key in test_keys:
#         test_queries += [ all_queries[idx] for idx in partition_query_indices[key] ]
#         test_cards += [ all_cards[idx] for idx in partition_query_indices[key] ]
#         test_query_infos += [ all_query_infos[idx] for idx in partition_query_indices[key] ]
#     for i, key in enumerate( train_keys ):
#         if i < int(num_partition_set / 2):
#             select_ratio = 1.
#         elif num_partition_set % 2 == 1 and i == int(num_partition_set / 2):
#             select_ratio = 1.
#         else:
#             select_ratio = 1.
#         num_train = int(len(partition_query_indices[key]) * select_ratio)
#         partition_train_queries = [ all_queries[idx] for idx in partition_query_indices[key][:num_train]]
#         partition_train_cards = [ all_cards[idx] for idx in partition_query_indices[key][:num_train]]
#         partition_train_query_infos = [ all_query_infos[idx] for idx in partition_query_indices[key][:num_train]]
#         train_queries += partition_train_queries
#         train_cards += partition_train_cards
#         train_query_infos += partition_train_query_infos
#     return train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos


""" The following random split the queries by query  pattern ID. """
# def query_pattern_train_test_split(all_queries, all_cards, all_query_infos, pattern_indices, skew_split_keys='query_pattern', train_frac = 0.6, skew_ratio=1):
#     pred_stat = PredictionStatistics()
#     partition_query_indices = pred_stat.get_partitioned_indices_by_query_pattern(all_query_infos, pattern_indices=pattern_indices, part_keys=skew_split_keys)
#     num_partition_set = len(partition_query_indices)
#     train_queries, train_cards, train_query_infos = list(), list(), list()
#     test_queries, test_cards, test_query_infos = list(), list(), list()
#     # Only one key. 
#     keys = [ k for k  in partition_query_indices.keys() ]
#     random.shuffle(keys)
#     num_train = int(len(keys) * train_frac)
#     train_keys = keys[:num_train]
#     test_keys = keys[num_train:]
#     # for key in train_keys:
#     #     train_queries += [ all_queries[idx] for idx in partition_query_indices[key] ]
#     #     train_cards += [ all_cards[idx] for idx in partition_query_indices[key]]
#     #     train_query_info += [ all_query_infos[idx] for idx in partition_query_indices[key] ]
#     train_queries, train_cards, train_query_infos = list(), list(), list()
#     test_queries, test_cards, test_query_infos = list(), list(), list()
#     for key in test_keys:
#         test_queries += [ all_queries[idx] for idx in partition_query_indices[key] ]
#         test_cards += [ all_cards[idx] for idx in partition_query_indices[key] ]
#         test_query_infos += [ all_query_infos[idx] for idx in partition_query_indices[key] ]
#     for i, key in enumerate( train_keys ):
#         if i < int(num_partition_set / 2):
#             select_ratio = 1.
#         elif num_partition_set % 2 == 1 and i == int(num_partition_set / 2):
#             select_ratio = 1.
#         else:
#             select_ratio = 1.
#         num_train = int(len(partition_query_indices[key]) * select_ratio)
#         partition_train_queries = [ all_queries[idx] for idx in partition_query_indices[key][:num_train]]
#         partition_train_cards = [ all_cards[idx] for idx in partition_query_indices[key][:num_train]]
#         partition_train_query_infos = [ all_query_infos[idx] for idx in partition_query_indices[key][:num_train]]
#         train_queries += partition_train_queries
#         train_cards += partition_train_cards
#         train_query_infos += partition_train_query_infos
#     return train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos


def query_pattern_train_test_split(all_queries, all_cards, all_query_infos, all_queries_str=None, skew_split_keys='template_no', train_frac = 0.6, skew_ratio=1):
    pred_stat = PredictionStatistics()
    # When getting partitioned indices, the function will automatically add 'num_table' into skew_split_keys, to support splitting by number of tables.  
    print(len(all_queries), len(all_cards), len(all_query_infos))
    partition_query_indices = pred_stat.get_partitioned_indices_by_query_pattern(all_query_infos, part_keys=skew_split_keys)
    train_queries_str, train_queries, train_cards, train_query_infos = list(), list(), list(), list()
    test_queries_str, test_queries, test_cards, test_query_infos = list(), list(), list(), list()


    if skew_split_keys == 'template_no':
        table_keys = partition_query_indices['num_table'].keys()
        table_keys = [ k for k in sorted(table_keys) ]
        for k in table_keys:
            random.shuffle(partition_query_indices['num_table'][k])
    
        num_partition_set = len(partition_query_indices['num_table'].keys())
        num_train = int(num_partition_set * train_frac)

        train_keys = []
        test_keys = []

        # Make sure that the queries are partitioned by table number. 
        for k in table_keys:
            #  if len(train_keys) < num_train:
            if int(k) < 4:
                train_keys.append(k)
            else:
                test_keys.append(k)

        for key in test_keys:
            test_queries += [ all_queries[idx] for idx in partition_query_indices['num_table'][key] ]
            test_cards += [ all_cards[idx] for idx in partition_query_indices['num_table'][key] ]
            test_query_infos += [ all_query_infos[idx] for idx in partition_query_indices['num_table'][key] ]
            test_queries_str += [ all_queries_str[idx] for idx in partition_query_indices['num_table'][key] ]
        for i, key in enumerate( train_keys ):
            if i < int(num_partition_set / 2):
                select_ratio = 1.
            elif num_partition_set % 2 == 1 and i == int(num_partition_set / 2):
                select_ratio = 1.
            else:
                select_ratio = 1.
            num_train = int(len(partition_query_indices['num_table'][key]) * select_ratio)
            partition_train_queries = [ all_queries[idx] for idx in partition_query_indices['num_table'][key][:num_train]]
            partition_train_cards = [ all_cards[idx] for idx in partition_query_indices['num_table'][key][:num_train]]
            partition_train_query_infos = [ all_query_infos[idx] for idx in partition_query_indices['num_table'][key][:num_train]]
            partition_train_queries_str = [ all_queries_str[idx] for idx in partition_query_indices['num_table'][key][:num_train]]
            train_queries += partition_train_queries
            train_cards += partition_train_cards
            train_query_infos += partition_train_query_infos
            train_queries_str += partition_train_queries_str
    else:
        # split by template_no (which is essentially the file no.)
        # print(partition_query_indices)
        partition_keys = partition_query_indices[skew_split_keys].keys()
        # No need to sort. 
        train_queries = len(all_queries) * train_frac
        train_keys = []
        test_keys = []
        train_num = 0
        for k in partition_keys:
            if train_num < train_queries: 
                train_keys.append(k)
            else: 
                test_keys.append(k)
            train_num += len(partition_query_indices[skew_split_keys][k])
        for key in test_keys:
            test_queries += [ all_queries[idx] for idx in partition_query_indices[skew_split_keys][key] ]
            test_cards += [ all_cards[idx] for idx in partition_query_indices[skew_split_keys][key] ]
            test_query_infos += [ all_query_infos[idx] for idx in partition_query_indices[skew_split_keys][key] ]
            test_queries_str += [ all_queries_str[idx] for idx in partition_query_indices[skew_split_keys][key] ]
        for i, key in enumerate(train_keys):
            if i < int(len(partition_keys) / 2):
                select_ratio = 1.
            elif len(partition_keys) % 2 == 1 and i == int(len(partition_keys) / 2):
                select_ratio = 1.
            else:
                select_ratio = 1.
            num_train = int(len(partition_query_indices[skew_split_keys][key]) * select_ratio)
            train_queries += [ all_queries[idx] for idx in partition_query_indices[skew_split_keys][key][:num_train] ]
            train_cards += [ all_cards[idx] for idx in partition_query_indices[skew_split_keys][key][:num_train] ]
            train_query_infos += [ all_query_infos[idx] for idx in partition_query_indices[skew_split_keys][key][:num_train]]
            train_queries_str += [ all_queries_str[idx] for idx in partition_query_indices[skew_split_keys][key][:num_train]]

    print("Train and test:")
    print(len(train_queries_str), len(train_queries), len(train_cards), len(train_query_infos), len(test_queries_str), len(test_queries), len(test_cards), len(test_query_infos))
    return train_queries_str, train_queries, train_cards, train_query_infos, test_queries_str, test_queries, test_cards, test_query_infos



def uneven_train_test_split_v2(all_queries, all_cards, all_query_infos, skew_split_keys: str, train_frac = 0.9, skew_ratio =0.5):
    test_frac = 1.0 - train_frac
    train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos, val_queries, val_cards, val_query_infos = \
        train_test_val_split(all_queries, all_cards, train_frac, test_frac, all_query_infos)
    pred_stat = PredictionStatistics()
    partition_query_indices_train = pred_stat.get_partitioned_indices(train_query_infos, part_keys=skew_split_keys)
    skew_train_queries, skew_train_cards, skew_train_query_infos = list(), list(), list()
    num_partition_set = len(partition_query_indices_train)
    # make the train data uneven distribution
    for i, key in enumerate(sorted(partition_query_indices_train.keys())):
        if i < int(num_partition_set / 2):
            select_ratio = skew_ratio
        elif num_partition_set % 2 == 1 and i == int(num_partition_set / 2):
            select_ratio = 0.5
        else:
            select_ratio = 1.0 - skew_ratio
        num_train = int(len(partition_query_indices_train[key]) * select_ratio)
        print(key, num_train, select_ratio)
        partition_train_queries = [all_queries[idx] for idx in partition_query_indices_train[key][:num_train]]
        partition_train_cards = [all_cards[idx] for idx in partition_query_indices_train[key][:num_train]]
        partition_train_query_infos = [all_query_infos[idx] for idx in partition_query_indices_train[key][:num_train]]
        skew_train_queries += partition_train_queries
        skew_train_cards += partition_train_cards
        skew_train_query_infos += partition_train_query_infos
    return skew_train_queries, skew_train_cards, skew_train_query_infos, test_queries, test_cards, test_query_infos
    

class PredictionStatistics(object):
    def __init__(self):
        self.keys = [ 'num_table', 'num_joins', 'num_predicates', 'template_no', 'distri', 'table_comb' ]

    def get_prediction_details(self, errors, base, query_infos = None, partition_keys=''):
        print("The Overall Prediction Results:")
        print(partition_keys)
        self.get_prediction_statistics(errors, base)
        if query_infos is None or not partition_keys:
            return
        partition_keys = partition_keys.strip().split(',')
        partition_keys = [key.strip() for key in partition_keys]
        for key in partition_keys:
            assert key in self.keys, "Unsupported partition key!"

        partition_errors = {}
        for error, query_info in zip(errors.tolist(), query_infos):
            query_attrs = tuple( getattr(query_info, key) for key in partition_keys)
            if query_attrs not in partition_errors.keys():
                partition_errors[query_attrs] = []
            partition_errors[query_attrs].append(error)

        # if partition_keys[0] == 'num_table' or partition_keys[0] == 'num_predicates':
        #     print("called.")
        #     simple_complex_errors = {}
        #     keys = [ k for k in sorted(partition_errors.keys()) ]
        #     middle = math.ceil( (keys[0][0] + keys[-1][0] ) / 2 )
        #     for error, query_info in zip(errors.tolist(), query_infos):
        #         query_attrs = getattr(query_info, partition_keys[0])
        #         key = 'simple' if query_attrs <= middle else 'complex'
        #         # print(query_attrs, key, middle, "_".join( [ str(k) for k in partition_errors.keys() ] ))
        #         if key not in simple_complex_errors.keys():
        #             simple_complex_errors[key] = []
        #         simple_complex_errors[key].append(error)
        #     for simple_or_complex in simple_complex_errors.keys():
        #         info_str = ["{}={}".format(key, attr) for key, attr in zip(['simple_complex'], [simple_or_complex])]
        #         info_str = 'Query attributes:' + ','.join(info_str)
        #         print(info_str)
        #         print('# Queries = {}'.format(len(simple_complex_errors[simple_or_complex])))
        #         error = np.array(simple_complex_errors[simple_or_complex])
        #         self.get_prediction_statistics(error, base)


        # # shrink the result display size
        # tmp_partition_errors = {}
        # if len(partition_errors) > 6:
        #     tmp_partition_errors_list = [(query_attrs, partition_errors[query_attrs])
        #                                  for query_attrs in sorted(partition_errors.keys())]
        #     for i, (query_attrs, errors) in enumerate(tmp_partition_errors_list):
        #         if i % 2 == 0 and i < len(tmp_partition_errors_list) - 1:
        #             continue
        #         elif i % 2 == 1:
        #             errors += tmp_partition_errors_list[i - 1][1]
        #             tmp_partition_errors[query_attrs] = errors
        #         else:
        #             tmp_partition_errors[query_attrs] = errors
        #     partition_errors = tmp_partition_errors

        for query_attrs in sorted(partition_errors.keys()):
            info_str = ["{}={}".format(key, attr) for key, attr in zip(partition_keys, list(query_attrs))]
            info_str = 'Query attributes:' + ','.join(info_str)
            print(info_str)
            print('# Queries = {}'.format(len(partition_errors[query_attrs])))
            error = np.array(partition_errors[query_attrs])
            self.get_prediction_statistics(error, base)

    def get_prediction_details_by_query_pattern(self, errors, base, query_infos = None, partition_keys = 'template_no'):
        print("The Overall Prediction Result:")
        self.get_prediction_statistics(errors, base)
        if query_infos is None or not partition_keys:
            return 
        
        partition_errors = {}
        for error, query_info in zip(errors.tolist(), query_infos):
            query_attrs = tuple( getattr(query_info, partition_key))
            partition_erros[query_attrs].append(error)
        tmp_partition_errors = {}   
        if len(partition_errors) > 6:
            tmp_partition_errors_list = [(query_attrs, partition_errors[query_attrs])
                                         for query_attrs in sorted(partition_errors.keys())]
            for i, (query_attrs, errors) in enumerate(tmp_partition_errors_list):
                if i % 2 == 0 and i < len(tmp_partition_errors_list) - 1:
                    continue
                elif i % 2 == 1:
                    errors += tmp_partition_errors_list[i - 1][1]
                    tmp_partition_errors[query_attrs] = errors
                else:
                    tmp_partition_errors[query_attrs] = errors
            partition_errors = tmp_partition_errors


        for query_attrs in sorted(partition_errors.keys()):
            info_str = ["{}={}".format(key, attr) for key, attr in zip(partition_keys, list(query_attrs))]
            info_str = 'Query attributes:' + ','.join(info_str)
            print(info_str)
            print('# Queries = {}'.format(len(partition_errors[query_attrs])))
            error = np.array(partition_errors[query_attrs])
            self.get_prediction_statistics(error, base)

    def get_prediction_statistics(self, errors, base):
        # import pdb; pdb.set_trace()
        try:
            errors = np.power(base, errors) # transform back from log scale
        except BaseException as exp:
            print("base = ", base)
            print("type(errors) = ", type(errors))
            print("errors = ", errors)
            raise exp
        # inds = np.where(errors < 1.0)
        # errors[inds] = 1 / errors[inds] 
        print("-" * 80)
        # print(errors)
        print("-" * 80)
        lower, upper = np.quantile(errors, 0.25), np.quantile(errors, 0.75)
        print("<" * 80)
        print("Predict Result Profile of {} Queries:".format(len(errors)))
        print("Min/Max: {:.15f} / {:.15f}".format(np.min(errors), np.max(errors)))
        print("Mean: {:.8f}".format(np.mean(errors)))
        print("Median: {:.8f}".format(np.median(errors)))
        print("50%/50% Quantiles: {:.8f} / {:.8f}".format(np.quantile(errors, 0.5), np.quantile(errors, 0.5)))
        print("25%/75% Quantiles: {:.8f} / {:.8f}".format(lower, upper))
        print("5%/95% Quantiles: {:.8f} / {:.8f}".format(np.quantile(errors, 0.05), np.quantile(errors, 0.95)))
        print("1%/99% Quantiles: {:.8f} / {:.8f}".format(np.quantile(errors, 0.01), np.quantile(errors, 0.99)))
        #plot_str = "lower whisker={:.10f}, \nlower quartile={:.10f}, \nmedian={:.10f}, \nupper quartile={:.10f}, \nupper whisker={:.10f},"\
        #	.format(onp.min(errors), lower, onp.median(errors), upper, onp.max(errors))
        #print(plot_str)
        print(">" * 80)
        error_median = abs(upper - lower)
        return error_median

    def get_permutation_index(self, query_infos, perm_keys=''):
        # return a numpy array as the permutation index based on the perm_keys
        num_instances = len(query_infos)
        if not perm_keys:
            return np.array(list(range(num_instances)))

        partition_query_indices = self.get_partitioned_indices(query_infos, part_keys= perm_keys)

        permutation = []
        for query_attrs in sorted(partition_query_indices.keys()):
            for idx in partition_query_indices[query_attrs]:
                permutation.append(idx)
        return np.array(permutation)

    def get_permutation_data(self, X, query_infos, perm_keys):
        num_instances = len(X) if isinstance(X, list) else X.shape[0]
        assert num_instances == len(query_infos), "Data size inconsistent with query info!"
        permutation = self.get_permutation_index(query_infos, perm_keys)
        if isinstance(X, list):
            X = [X[idx] for idx in permutation.tolist()]
        else:
            X = X[permutation]
        return X

    def get_partitioned_data(self, X, query_infos, part_keys):
        num_instances = len(X) if isinstance(X, list) else X.shape[0]
        assert num_instances == len(query_infos), "Data size inconsistent with query info!"
        partition_query_indices = self.get_partitioned_indices(query_infos, part_keys)

        partitioned_X = []
        for query_attrs in sorted(partition_query_indices.keys()):
            x = [X[idx] for idx in partition_query_indices[query_attrs]]
            if not isinstance(X, list):
                x = np.asarray(x)
            partitioned_X.append(x)
        return partitioned_X

    def get_partitioned_indices(self, query_infos, part_keys):
        part_keys = part_keys.strip().split(',')
        part_keys = [ key.strip() for key in part_keys]
        for key in part_keys:
            assert key in self.keys, "Unsupported partition key!"
        partition_query_indices = dict()
        for i, query_info in enumerate(query_infos):
            query_attrs = tuple(getattr(query_info, key) for key in part_keys)
            if query_attrs not in partition_query_indices.keys():
                partition_query_indices[query_attrs] = list()
            partition_query_indices[query_attrs].append(i)
        return partition_query_indices

    def get_partitioned_indices_by_table_comb(self, query_infos, part_keys):
        # assert part_keys == 'query_pattern', 'Not query pattern!'
        part_keys = part_keys.strip().split(',')
        part_keys = [ key.strip() for key in part_keys ]
        for key in part_keys:
            assert key in self.keys, "Unsupported partition key!"
        partition_query_indices = { key: {} for key in part_keys }
        for i, query_info in enumerate(query_infos):
            for key in part_keys:
                query_attr = getattr(query_info, key)
                if query_attr not in partition_query_indices[key]:
                    partition_query_indices[key][query_attr] = list()
                # print("i = ", i, "key = ", key, "query_attr", query_attr)
                partition_query_indices[key][query_attr].append(i)
        return partition_query_indices

    # Should also consider the table number ( as the partition key )
    def get_partitioned_indices_by_query_pattern(self, query_infos, part_keys):
        # assert part_keys == 'query_pattern', 'Not query pattern!'
        part_keys = part_keys + ',num_table'
        part_keys = part_keys.strip().split(',')
        part_keys = [ key.strip() for key in part_keys ]
        for key in part_keys:
            assert key in self.keys, "Unsupported partition key!"
        partition_query_indices = { key: {} for key in part_keys }
        for i, query_info in enumerate(query_infos):
            for key in part_keys:
                query_attr = getattr(query_info, key)
                if query_attr not in partition_query_indices[key]:
                    partition_query_indices[key][query_attr] = list()
                # print("i = ", i, "key = ", key, "query_attr", query_attr)
                partition_query_indices[key][query_attr].append(i)
        return partition_query_indices
