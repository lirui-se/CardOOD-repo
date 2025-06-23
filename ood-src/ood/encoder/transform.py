from db.table import Table, Address
from db.schema import DBSchema
from torch.utils.data.dataset import Dataset
from typing import List
from torch.utils.data import DataLoader
import torch
import math
import random

class DnnEncoder(object):
    def __init__(self, table: Table, chunk_size: int = 64):
        super().__init__()
        self.table = table
        self.num_cols = table.num_cols
        self.col_types = table.col_types
        self.chunk_size = chunk_size
        self.df = table.df
        self.feat_dim = 0
        self.all_col_address = list()
        for i in range(self.num_cols):
            col_name = self.df.columns[i]
            if self.col_types[i] == 'categorical':
                # categorical type
                num_cat = len(self.table.categorical_codes_dict[col_name])
                encode_dim = math.ceil(float(num_cat) / self.chunk_size)
                self.all_col_address.append(Address(start=self.feat_dim, end=self.feat_dim + encode_dim))
                self.feat_dim += encode_dim
            else:  # numerical type
                self.all_col_address.append(Address(start=self.feat_dim, end=self.feat_dim + 2))
                self.feat_dim += 2

    def _factorized_encoding(self, col_idx, cat_set):
        assert self.col_types[col_idx] == 'categorical', 'Only categorical attribute supports factorized encoding！'
        encode_address = self.all_col_address[col_idx]
        encode_dim = encode_address.end - encode_address.start
        encoding_str = ['0'] * (encode_dim * self.chunk_size)
        cat_set = [int(cat) for cat in cat_set]
        for cat in cat_set:
            encoding_str[cat] = '1'
        encoding_str = "".join(encoding_str)
        encoding_str = [encoding_str[i: i + self.chunk_size] for i in range(0, len(encoding_str), self.chunk_size)]
        factorized_encoding = [int(code, 2) for code in encoding_str]
        return factorized_encoding

    def __call__(self, pred_list: List):
        # predicate encoding used for DNN
        x = torch.zeros(size=(self.feat_dim,), dtype=torch.float32)
        for col_idx in range(self.num_cols):
            if self.col_types[col_idx] == 'numerical':
                x[self.all_col_address[col_idx].start + 1] = 1000
        for pred in pred_list:
            col_name = pred[0]
            col_idx = self.df.columns.get_loc(col_name)
            encode_address = self.all_col_address[col_idx]
            if self.col_types[col_idx] == 'categorical':
                factorized_encoding = self._factorized_encoding(col_idx, pred[1])
                idx = list(range(encode_address.start, encode_address.end))
                x[idx] = factorized_encoding
            else:
                upper, lower = pred[1], pred[2]
                upper = (upper - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
                lower = (lower - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
                x[encode_address.start] = upper
                x[encode_address.start + 1] = lower
        return x


class DnnJoinQueryEncoder(object):
    def __init__(self, schema: DBSchema, chunk_size: int = 64):
        super().__init__()
        self.schema = schema
        self.tables = schema.tables
        self.table_encoders = [ DnnEncoder(table, chunk_size) for table in self.tables]
        self.total_num_joins = len(schema.all_join_triples)
        self.join_ops_dict = {'=': 0} # only support Equi-join
        self.join_feat_dim = self.total_num_joins * len(self.join_ops_dict)
        self.feat_dim = self.join_feat_dim + sum([table_encoder.feat_dim for table_encoder in self.table_encoders])

    def _join_encoding(self, join_infos):
        x_join = torch.zeros(size=(self.join_feat_dim,), dtype=torch.float32)
        for join_info in join_infos:
            t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
            join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
            idx = self.schema.all_join_triples.index(join_triple)
            for c in op:
                x_join[idx * len(self.join_ops_dict) + self.join_ops_dict[c]] = 1
        return x_join

    def __call__(self, table_ids, all_pred_list, join_infos):
        x = list()
        for t_id in range(len(self.tables)):
            pred_list = all_pred_list[table_ids.index(t_id)] if t_id in table_ids else []
            x_pred = self.table_encoders[t_id](pred_list)
            x.append(x_pred)
        x.append(self._join_encoding(join_infos))
        x = torch.cat(x, dim= -1)
        return x


class OnehotEncoderKMeans(object):
    def __init__(self, table: Table):
        super().__init__()
        self.table = table
        self.num_cols = table.num_cols
        self.col_types = table.col_types
        self.df = table.df
        self.feat_dim = self.num_cols + 3

    def pred_encoding(self, pred_list: List):
        # predict encoding for MSCN, TreeLSTM
        cols_x = torch.zeros(size=(2 * self.num_cols, self.num_cols), dtype=torch.float32)
        ops_x = torch.zeros(size=(2 * self.num_cols, 3), dtype=torch.float32)
        for i, pred in enumerate(pred_list):
            col_name = pred[0]
            col_idx = self.df.columns.get_loc(col_name)
            if self.col_types[col_idx] == 'categorical':
                raise NotImplementedError('Can not support categorical attribute!')
            upper, lower = pred[1], pred[2]
            upper = (upper - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
            lower = (lower - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
            cols_x[2 * col_idx, col_idx] = 1
            ops_x[2 * col_idx, 0] = 1
            ops_x[2 * col_idx, 2] = upper

            cols_x[2 * col_idx + 1, col_idx] = 1
           
            ops_x[2 * col_idx + 1, 2] = lower
        return cols_x, ops_x

    def __call__(self, pred_list: List):
        cols_x, ops_x = self.pred_encoding(pred_list)
        x = torch.cat([cols_x, ops_x], dim=-1) # [2 * len(pred_list), num_cols + 3]
        return x


class OnehotEncoder(object):
    def __init__(self, table: Table):
        super().__init__()
        self.table = table
        self.num_cols = table.num_cols
        self.col_types = table.col_types
        self.df = table.df
        self.feat_dim = self.num_cols + 3

    def pred_encoding(self, pred_list: List):
        # predict encoding for MSCN, TreeLSTM
        cols_x = torch.zeros(size=(2 * len(pred_list), self.num_cols), dtype=torch.float32)
        ops_x = torch.zeros(size=(2 * len(pred_list), 3), dtype=torch.float32)
        for i, pred in enumerate(pred_list):
            col_name = pred[0]
            col_idx = self.df.columns.get_loc(col_name)
            if self.col_types[col_idx] == 'categorical':
                raise NotImplementedError('Can not support categorical attribute!')
            upper, lower = pred[1], pred[2]
            upper = (upper - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
            lower = (lower - self.table.all_col_ranges[col_idx][0]) / self.table.all_col_denominator[col_idx] * 1000
            cols_x[2 * i, col_idx] = 1
            ops_x[2 * i, 0] = 1
            ops_x[2 * i, 2] = upper

            cols_x[2 * i + 1, col_idx] = 1
           
            ops_x[2 * i + 1, 2] = lower
        return cols_x, ops_x

    def __call__(self, pred_list: List):
        cols_x, ops_x = self.pred_encoding(pred_list)
        x = torch.cat([cols_x, ops_x], dim=-1) # [2 * len(pred_list), num_cols + 3]
        return x

class OnehotJoinQueryEncoderKMeans(object):
    def __init__(self, schema: DBSchema, chunk_size: int = 64):
        super().__init__()
        self.schema = schema
        self.tables = schema.tables
        self.table_encoders = [OnehotEncoderKMeans(table) for table in self.tables]
        self.total_num_joins = len(schema.all_join_triples)
        self.join_ops_dict = {'=': 0}  # only support Equi-join
        self.join_feat_dim = self.total_num_joins
        self.pred_address_dim = 0
        self.all_pred_address = []  # [Address] : the address ([start, end)) of encoding in the feature of column i
        for table in self.tables:
            self.all_pred_address.append(Address(start=self.pred_address_dim, end=self.pred_address_dim + table.num_cols))
            self.pred_address_dim += table.num_cols
        self.pred_feat_dim = self.pred_address_dim + 3
        self.table_feat_dim = len(self.tables)

    def _join_encoding(self, join_infos):
        x_join = torch.zeros(size=(self.total_num_joins, self.join_feat_dim), dtype=torch.float32)
        for join_info in join_infos:
            t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
            join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
            idx = self.schema.all_join_triples.index(join_triple)
            x_join[idx, idx] = 1
        return x_join


    def _table_encoding(self, table_ids):
        x_table = torch.zeros(size=(self.table_feat_dim, self.table_feat_dim), dtype=torch.float32)
        for i, t_id in enumerate(table_ids):
            x_table[t_id, t_id] = 1
        return x_table


    def _pred_encoding(self, table_ids, all_pred_list):
        x_pred = list()
        indx = 0
        for t_id, pred_list in zip(table_ids, all_pred_list):
            if not pred_list:
                # x_pred.append(torch.zeros(size=(1, self.pred_feat_dim)))
                continue
            cols_x, ops_x = self.table_encoders[t_id].pred_encoding(pred_list)
            start, end = self.all_pred_address[t_id].start, self.all_pred_address[t_id].end
            one_table_pred_x = [torch.zeros(size=(cols_x.shape[0], start), ), cols_x,
                                torch.zeros(size=(cols_x.shape[0], self.pred_address_dim - end), ), ops_x]
            one_table_pred_x = torch.cat(one_table_pred_x, dim=-1)
            # if t_id > indx:
            #     for i in range(indx, t_id):
            #         x_pred.append(torch.zeros(size=(one_table_pred_x.shape[0], one_table_pred_x.shape[1])))
            #         indx += 1
            # assert t_id == indx
            x_pred.append(one_table_pred_x)
        x_pred = torch.cat(x_pred, dim=0)
        return x_pred


    def __call__(self, table_ids, all_pred_list, join_infos):
        x_table = self._table_encoding(table_ids)
        x_pred = self._pred_encoding(table_ids, all_pred_list)
        x_join = self._join_encoding(join_infos)
        x = (x_table, x_pred, x_join)
        return x




class OnehotJoinQueryEncoder(object):
    def __init__(self, schema: DBSchema, chunk_size: int = 64):
        super().__init__()
        self.schema = schema
        self.tables = schema.tables
        self.table_encoders = [OnehotEncoder(table) for table in self.tables]
        self.total_num_joins = len(schema.all_join_triples)
        self.join_ops_dict = {'=': 0}  # only support Equi-join
        self.join_feat_dim = self.total_num_joins
        self.pred_address_dim = 0
        self.all_pred_address = []  # [Address] : the address ([start, end)) of encoding in the feature of column i
        for table in self.tables:
            self.all_pred_address.append(Address(start=self.pred_address_dim, end=self.pred_address_dim + table.num_cols))
            self.pred_address_dim += table.num_cols
        self.pred_feat_dim = self.pred_address_dim + 3
        self.table_feat_dim = len(self.tables)

    def _join_encoding(self, join_infos):
        x_join = torch.zeros(size=(self.total_num_joins, self.join_feat_dim), dtype=torch.float32)
        for join_info in join_infos:
            t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
            join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
            idx = self.schema.all_join_triples.index(join_triple)
            x_join[idx, idx] = 1
        return x_join


    def _table_encoding(self, table_ids):
        x_table = torch.zeros(size=(len(table_ids), self.table_feat_dim), dtype=torch.float32)
        for i, t_id in enumerate(table_ids):
            x_table[i, t_id] = 1
        return x_table


    def _pred_encoding(self, table_ids, all_pred_list):
        x_pred = list()
        for t_id, pred_list in zip(table_ids, all_pred_list):
            if not pred_list:
                x_pred.append(torch.zeros(size=(1, self.pred_feat_dim)))
                continue
            cols_x, ops_x = self.table_encoders[t_id].pred_encoding(pred_list)
            start, end = self.all_pred_address[t_id].start, self.all_pred_address[t_id].end
            one_table_pred_x = [torch.zeros(size=(cols_x.shape[0], start), ), cols_x,
                                torch.zeros(size=(cols_x.shape[0], self.pred_address_dim - end), ), ops_x]
            one_table_pred_x = torch.cat(one_table_pred_x, dim=-1)
            x_pred.append(one_table_pred_x)
        x_pred = torch.cat(x_pred, dim=0)
        return x_pred


    def __call__(self, table_ids, all_pred_list, join_infos):
        x_table = self._table_encoding(table_ids)
        x_pred = self._pred_encoding(table_ids, all_pred_list)
        x_join = self._join_encoding(join_infos)
        x = (x_table, x_pred, x_join)
        return x



class QueryDataset(Dataset):
    def __init__(self, table: Table, queries, cards, query_infos: List = None, encoder_type: str = 'dnn'):
        self.encoder = DnnEncoder(table= table) if encoder_type in ['dnn', 'DNN'] else OnehotEncoder(table)
        self.table = table
        self.feat_dim = self.encoder.feat_dim
        self.queries = queries
        self.cards = cards
        self.query_infos = query_infos

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        pred_list = self.queries[item]
        card = self.cards[item]
        y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(pred_list)

        return x, y


class MaskedQueryDataset(QueryDataset):
    def __init__(self, table: Table, queries, cards, query_infos: List = None, p: float= 0.2,  encoder_type: str = 'dnn'):
        super(MaskedQueryDataset, self).__init__(table, queries, cards, query_infos, encoder_type)
        self.p = p

    def __getitem__(self, item):
        pred_list = self.queries[item]
        card = self.cards[item]
        y = torch.FloatTensor([math.log2(card)])
        masked_pred_list = list()
        for pred in pred_list:
            if random.random() <= self.p: # random dropout
                continue
            masked_pred_list.append(pred)
        x = self.encoder(masked_pred_list)
        return x, y


class TTTQueryDataset(QueryDataset):
    def __init__(self, table: Table, queries, cards, query_infos: List = None, num_negs: int= 5,  encoder_type: str = 'dnn'):
        super(TTTQueryDataset, self).__init__(table, queries, cards, query_infos, encoder_type)
        self.num_negs = num_negs

    def __getitem__(self, item):
        pred_list = self.queries[item]
        card = self.cards[item]
        y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(pred_list)
        x_neg = list()
        for _ in range(self.num_negs):
            sub_pred_list = self.table.subquery_sample(pred_list)
            x_neg.append(self.encoder(sub_pred_list).unsqueeze(dim = 0))
        x_neg = torch.cat(x_neg, dim=0)
        #print(x)
        #print(x_neg[0])
        if x_neg.dim() == 3:
            x_neg = x_neg.permute(1, 0, 2) # [num_pred, num_negs, feat_dim]
        return x, y, x_neg


class JoinQueryDataset(Dataset):
    def __init__(self, schema: DBSchema, queries:List, cards: List, query_infos: List = None, encoder_type: str = 'dnn'):
        self.encoder = DnnJoinQueryEncoder(schema= schema) if encoder_type in ['dnn', 'DNN'] else \
            OnehotJoinQueryEncoder(schema=schema)
        self.encoder_type = encoder_type
        if encoder_type in ['dnn', 'DNN']:
            self.feat_dim = self.encoder.feat_dim
            self.pred_feat_dim, self.table_feat_dim, self.join_feat_dim = None, None, None
        else:
            self.feat_dim = None
            self.pred_feat_dim, self.table_feat_dim, self.join_feat_dim = self.encoder.pred_feat_dim, self.encoder.table_feat_dim, self.encoder.join_feat_dim
        self.queries = queries
        self.cards = cards
        self.query_infos = query_infos

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        table_ids, all_pred_list, join_infos = self.queries[item]
        card = self.cards[item]
        y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(table_ids, all_pred_list, join_infos)
        return x, y

    

class MaskedJoinQueryDataset(JoinQueryDataset):
    def __init__(self, schema: DBSchema, queries: List, cards: List, query_infos: List = None, p: float= 0.2,encoder_type: str= 'dnn'):
        super(MaskedJoinQueryDataset, self).__init__(schema, queries, cards, query_infos, encoder_type)
        self.p = p

    def __getitem__(self, item):
        table_ids, all_pred_list, join_infos = self.queries[item]
        card = self.cards[item]
        y = torch.FloatTensor([math.log2(card)])
        masked_all_pred_list = list()
        for pred_list in all_pred_list:
            masked_pred_list = list()
            for pred in pred_list:
                if random.random() <= self.p:
                    continue
                masked_pred_list.append(pred)
            masked_all_pred_list.append(masked_pred_list)
        x = self.encoder(table_ids, masked_all_pred_list, join_infos)
        return x, y

class TTTJoinQueryDataset(JoinQueryDataset):
    def __init__(self, schema: DBSchema, queries: List, cards: List, query_infos: List = None, num_negs: int= 5,encoder_type: str= 'dnn'):
        super(TTTJoinQueryDataset, self).__init__(schema, queries, cards, query_infos, encoder_type)
        self.num_negs = num_negs
        self.tables = schema.tables

    def __getitem__(self, item):
        table_ids, all_pred_list, join_infos = self.queries[item]
        # print(self.queries[item])
        # print(table_ids, all_pred_list, join_infos)
        card = self.cards[item]
        try:
            y = torch.FloatTensor([math.log2(card)])
        except ValueError:
            print("ValueError: {}".format(card))
            y = torch.FloatTensor([math.log2(card)])
        x = self.encoder(table_ids, all_pred_list, join_infos)
        x_neg = list()
        table_neg, pred_neg, join_neg = list(), list(), list()
        for _ in range(self.num_negs):
            tmp_all_pred_list = list()
            for t_id, pred_list in zip(table_ids, all_pred_list):
                if len(pred_list) > 0:
                    sub_pred_list = self.tables[t_id].subquery_sample(pred_list)
                else:
                    sub_pred_list = []
                tmp_all_pred_list.append(sub_pred_list)
            if self.encoder_type in ['dnn', 'DNN']:
                x_neg.append(self.encoder(table_ids, tmp_all_pred_list, join_infos).unsqueeze(dim = 0))
            else:
                x_table_neg, x_pred_neg, x_join_neg = self.encoder(table_ids, tmp_all_pred_list, join_infos)
                table_neg.append(x_table_neg.unsqueeze(dim = 0))
                pred_neg.append(x_pred_neg.unsqueeze(dim = 0))
                join_neg.append(x_join_neg.unsqueeze(dim = 0))
        if self.encoder_type in ['dnn', 'DNN']:
            x_neg = torch.cat(x_neg, dim=0)
        else:
            x_table_neg = torch.cat(table_neg, dim=0).permute(1, 0, 2)
            x_pred_neg = torch.cat(pred_neg, dim=0).permute(1, 0, 2)
            x_join_neg = torch.cat(join_neg, dim=0).permute(1, 0, 2)
            x_neg = (x_table_neg, x_pred_neg, x_join_neg)
        # print("[INSIDE DATASET]: ", type(x), type(y), type(x_neg), x.shape if isinstance(x, torch.Tensor) else "x", y.shape if isinstance(y, torch.Tensor) else "y", x_neg.shape if isinstance(x_neg, torch.Tensor) else "x_neg" )
        # import pdb; pdb.set_trace()
        # x = ( [4, 6], [13, 20], [15, 15] ), y = [1]
        # x_neg = ( [4, 10, 6], [13, 10, 20], [15, 10, 15] )
        return x, y, x_neg
