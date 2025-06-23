from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.schema import load_schema, load_table
from db.parser import QueryParser, JoinQueryParser
from db.table import QueryInfo
from encoder.transform import QueryDataset, TTTQueryDataset, TTTJoinQueryDataset
from model.model import MLP, MSCN, MSCNJoin
from util import train_test_val_split, uneven_train_test_split, query_pattern_train_test_split, PredictionStatistics, seed_all
from asset.ttt import TestTimeTrainer
from asset.base import recursive_to_device
from asset.env.environment import TTTJoinQueryEnvironment
from scripts.erm_main import get_query_infos
from ceb import get_testds_from_sql, load_model_datasets, log_predicts_stat
import numpy as np
import pdb

def ttt_init_args(arg_list: list):
    parser = ArgumentParser("CardOOD", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--chunk_size", default=64, type=int, help="dimension of factorized encoding")
    parser.add_argument("--num_hid", default=512, type=int, help="number of hidden")
    parser.add_argument("--mlp_num_hid", default=256, type=int, help="number of hidden")
    parser.add_argument("--table_num_hid", default=64, type=int, help="number of hidden")
    parser.add_argument("--pred_num_hid", default=64, type=int, help="number of hidden")
    parser.add_argument("--join_num_hid", default=64, type=int, help="number of hidden")
    parser.add_argument("--table_num_out", default=64, type=int, help="number of hidden")
    parser.add_argument("--pred_num_out", default=64, type=int, help="number of hidden")
    parser.add_argument("--join_num_out", default=64, type=int, help="number of hidden")

    parser.add_argument("--model_type", type=str, default='DNN', help="DNN, MSCN")
    parser.add_argument("--model_name", type=str, default='ttt', help="ttt,coral,erm...")
    parser.add_argument('--skew_ratio', type=float, default=0.8,
                        help='fraction of small queries in the training set')
    parser.add_argument('--num_negs', type=int, default=5,
                        help='# negative samples for order loss compute')
    parser.add_argument("--adapt_lr", default=1e-2, type=float)

    # Training parameters
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument('--weight_decay', type=float, default=2e-4,
    					help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.85,
    					help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10,
    					help='num of epochs for one lr decay.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
    					help='Disables CUDA training.')



    # input dir
    parser.add_argument("--is_join_query", type=bool, default=False)
    parser.add_argument("--skew_split_keys", type=str, default="num_table")
    parser.add_argument("--table_name", type=str, default='forest')
    parser.add_argument("--table_query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')
    parser.add_argument("--table_data_path", type=str, default='/home/kfzhao/data/UCI')
    parser.add_argument("--schema_name", type=str, default='imdb_simple', help='tpcds')
    parser.add_argument("--schema_data_path", type=str, default='/home/kfzhao/data/rdb/imdb_clean')
    parser.add_argument("--schema_query_path", type=str,
                        default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_824_FP')


    parser.add_argument("--manual_split", type=bool, default=False)
    parser.add_argument("--train_sql_path", type=str, default=None)
    parser.add_argument("--test_sql_path", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--dump_prediction", type=str, default=None)
    parser.add_argument("--debug_log_file", type=str, default=None)
    parser.add_argument("--debug_table_num", type=int, default=None)
    parser.add_argument("--prediction_file", type=str, default=None)
    parser.add_argument("--parallel_load", type=str, default=None)

    parser.add_argument("--cli_mode", type=str, default="off")

    parser.add_argument("--pred_mode", type=str, default="off")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--pred_file", type=str, default=None)


    # ceb arguments
    parser.add_argument("--config", type=str, required=False, default="configs/config.yaml")
    parser.add_argument("--alg", type=str, required=False, default="mscn")
    parser.add_argument("--regen_featstats", type=int, required=False, default=0)
    parser.add_argument("--save_featstats", type=int, required=False, default=1)
    parser.add_argument("--use_saved_feats", type=int, required=False, default=1)

    # logging arguments
    parser.add_argument("--wandb_tags", type=str, required=False, default=None, help="additional tags for wandb logs")
    parser.add_argument("--result_dir", type=str, required=False, default="./results")
    parser.add_argument("--eval_fns", type=str, required=False, default="ppc,qerr")

    # log arguments
    parser.add_argument("--dump_prediction", type=int, required=False, default=0)

    # pdb.set_trace()

    if arg_list is None:
        args = parser.parse_args()
    else:
        print(arg_list)
        args = parser.parse_args(arg_list)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    if args.is_join_query:
        args.query_path, args.data_path = args.schema_query_path, args.schema_data_path
    else:
        args.query_path, args.data_path = args.table_query_path, args.table_data_path
    return args

def prepare_table_datasets(args):
    table = load_table(args.table_name, args.data_path)
    query_parser = QueryParser(table=table)
    all_queries, all_cards, all_query_infos = query_parser.load_queries(query_path=args.query_path)
    #train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos, val_queries, val_cards, val_query_infos = (
    #    train_test_val_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos))
    train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos = (
        uneven_train_test_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos,
                                skew_split_keys='num_predicates', skew_ratio=args.skew_ratio))
    if args.model_type == 'MSCN':
        train_dataset = TTTQueryDataset(table=table, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='onehot')
        test_dataset = TTTQueryDataset(table=table, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='onehot')
    else:
        train_dataset = TTTQueryDataset(table=table, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='dnn')
        test_dataset = TTTQueryDataset(table=table, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='dnn')
    return train_dataset, train_query_infos, test_dataset, test_query_infos


def prepare_schema_datasets(args):
    schema = load_schema(args.schema_name, args.data_path)
    query_parser = JoinQueryParser(schema=schema)
    all_queries, all_cards, all_query_infos = query_parser.load_queries(query_path=args.query_path)
    #train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos, val_queries, val_cards, val_query_infos = (
    #    train_test_val_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos))
    train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos = (
       uneven_train_test_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos,
                               skew_split_keys=args.skew_split_keys, skew_ratio=args.skew_ratio))
    if args.model_type == 'MSCN':
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='onehot')
        test_dataset = TTTJoinQueryDataset(schema=schema, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='onehot')
    else:
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='dnn')
        test_dataset = TTTJoinQueryDataset(schema=schema, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='dnn')
    return train_dataset, test_dataset, test_query_infos

def prepare_datasets_with_table_comb(args, schema=None):
    if schema is None:
        schema = load_schema(args.schema_name, args.data_path)
    query_parser = JoinQueryParser(schema=schema)
    all_queries_str, all_queries, all_cards, all_query_infos = query_parser.load_queries_by_query_pattern(query_path=args.query_path)
    pred_stat = PredictionStatistics()
    partition_query_indices = pred_stat.get_partitioned_indices_by_table_comb(all_query_infos, part_keys='table_comb')
    keys = partition_query_indices['table_comb'].keys()
    keys = [ k for k in sorted(keys) ]
    import random
    for k in keys:
        random.shuffle(partition_query_indices['table_comb'][k])
    queries_list, cards_list, query_infos_list = [], [], []
    for key in keys:
        queries_list += [ all_queries[idx] for idx in partition_query_indices['table_comb'][key] ]
        cards_list += [ all_cards[idx] for idx in partition_query_indices['table_comb'][key] ]
        query_infos_list += [ all_query_infos[idx] for idx in partition_query_indices['table_comb'][key]]
    for k in keys:
        print(k, len( partition_query_indices['table_comb'][k]) )
    if args.model_type == 'MSCN':
        table_comb_dataset = TTTJoinQueryDataset(schema=schema, queries=queries_list, cards=cards_list, num_negs= args.num_negs, encoder_type='onehot')
    else:
        table_comb_dataset = TTTJoinQueryDataset(schema=schema, queries=queries_list, cards=cards_list, num_negs= args.num_negs, encoder_type='dnn')
    return table_comb_dataset, query_infos_list

def prepare_query_pattern_datasets(args, schema=None):
    if schema is None:
        schema = load_schema(args.schema_name, args.data_path)
    query_parser = JoinQueryParser(schema=schema)
    all_queries_str, all_queries, all_cards, all_query_infos = query_parser.load_queries_by_query_pattern(query_path=args.query_path)
    if args.manual_split:
        assert args.train_sql_path is not None and os.path.exists(args.train_sql_path)
        assert args.test_sql_path is not None and os.path.exists(args.test_sql_path)
        train_queries_str, train_queries, train_cards, train_query_infos = query_parser.load_queries_by_query_pattern(query_path=args.train_sql_path)
        test_queries_str, test_queries, test_cards, test_query_infos = query_parser.load_queries_by_query_pattern(query_path=args.test_sql_path)
    else:
        train_queries_str, train_queries, train_cards, train_query_infos, test_queries_str, test_queries, test_cards, test_query_infos = (
            query_pattern_train_test_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos, all_queries_str=all_queries_str, skew_split_keys=args.skew_split_keys, train_frac=0.6, skew_ratio=args.skew_ratio) 
            )
    print(len(train_queries_str) ,len(train_queries), len(train_cards), len(train_query_infos), len(test_queries_str), len(test_queries), len(test_cards), len(test_query_infos))
    # Dump the train query split. 
    if not args.manual_split and args.train_sql_path is not None and os.path.exists(args.train_sql_path):
        # dump by split keys.
        assert ',' not in args.skew_split_keys
        if args.skew_split_keys == 'template_no':
            partition_key = 'num_table'
        else:
            partition_key = args.skew_split_keys
        partition_strs = {}
        for train_str, query_info in zip(train_queries_str, train_query_infos):
            attr = getattr(query_info, partition_key)
            if attr not in partition_strs.keys():
                partition_strs[attr] = []
            partition_strs[attr].append(train_str)
        for attr in sorted(partition_strs.keys()):
            f = open(args.train_sql_path + "/" + partition_key + "_" + str(attr) + ".txt", "w")
            # partition_strs[attr].sort()
            f.write("\n".join(partition_strs[attr]) + "\n")
            f.close()
    if not args.manual_split and args.test_sql_path is not None and os.path.exists(args.test_sql_path):
        # dump by split keys.
        assert ',' not in args.skew_split_keys
        if args.skew_split_keys == 'template_no':
            partition_key = 'num_table'
        else:
            partition_key = args.skew_split_keys
        partition_strs = {}
        for test_str, query_info in zip(test_queries_str, test_query_infos):
            attr = getattr(query_info, partition_key)
            if attr not in partition_strs.keys():
                partition_strs[attr] = []
            partition_strs[attr].append(test_str)
        for attr in sorted(partition_strs.keys()):
            f = open(args.test_sql_path + "/" + partition_key + "_" + str(attr) + ".txt", "w")
            # partition_strs[attr].sort()
            f.write("\n".join(partition_strs[attr]) + "\n")
            f.close()
    if args.model_type == 'MSCN':
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='onehot')
        test_dataset = TTTJoinQueryDataset(schema=schema, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='onehot')
    else:
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='dnn')
        test_dataset = TTTJoinQueryDataset(schema=schema, queries=test_queries, cards=test_cards, num_negs= args.num_negs, encoder_type='dnn')
    return train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str
    
    


def prepare_schema_group_datasets(args):
    schema = load_schema(args.schema_name, args.data_path)
    query_parser = JoinQueryParser(schema=schema)
    all_queries, all_cards, all_query_infos = query_parser.load_queries(query_path=args.query_path)
    #train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos, val_queries, val_cards, val_query_infos = (
    #    train_test_val_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos))
    train_queries, train_cards, train_query_infos, test_queries, test_cards, test_query_infos = (
       uneven_train_test_split(all_queries=all_queries, all_cards=all_cards, all_query_infos=all_query_infos, skew_split_keys=args.skew_split_keys, skew_ratio=args.skew_ratio))
    if args.model_type == 'MSCN':
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='onehot')
        test_env = TTTJoinQueryEnvironment(schema=schema, queries=test_queries, cards=test_cards, query_infos=test_query_infos,
                                           num_negs=args.num_negs, part_keys='num_table', encoder_type='onehot')
    else:
        train_dataset = TTTJoinQueryDataset(schema=schema, queries=train_queries, cards=train_cards, num_negs= args.num_negs, encoder_type='dnn')
        test_env = TTTJoinQueryEnvironment(schema=schema, queries=test_queries, cards=test_cards, query_infos=test_query_infos,
                                           num_negs= args.num_negs, part_keys='num_table', encoder_type='dnn')
    return train_dataset, test_env, test_query_infos


def prepare_ceb_datasets(args):
    # import pdb; pdb.set_trace()
    model, trainds, testds, debugds = load_model_datasets(args)
    train_infos = get_query_infos(trainds)
    test_infos = get_query_infos(testds)
    debug_infos = get_query_infos(debugds)
    train_strs = [ "" for i in range(len(train_infos)) ]
    test_strs = [ "" for i in range(len(test_infos)) ]
    debug_strs = [ "" for i in range(len(debug_infos)) ]
    return model, trainds, train_infos, train_strs, testds, test_infos, test_strs, debugds, debug_infos, debug_strs

def prepare_model_and_dataset(args, schema=None):
    print("ttt_main.py::prepare_model_and_dataset::", end="")
    seed_all()
    if args.is_join_query:
        if args.model_type == 'CEB_MSCN':
            # model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str = prepare_ceb_datasets(args, schema=schema)
            model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str, debug_dataset, debug_query_infos, debug_queries_str = prepare_ceb_datasets(args)
        else:
            train_dataset,train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str = prepare_query_pattern_datasets(args, schema=schema)
    else:
        train_dataset, train_query_infos, test_dataset, test_query_infos = prepare_table_datasets(args)
        train_queries_str = []
        test_queries_str = []

    if args.model_type == 'MSCN':
        if args.is_join_query:
            model = MSCNJoin(train_dataset.table_feat_dim, args.table_num_hid, args.table_num_out,
                             train_dataset.pred_feat_dim, args.pred_num_hid, args.pred_num_out,
                             train_dataset.join_feat_dim, args.join_num_hid, args.join_num_out, args.mlp_num_hid, return_rep=True)
        else:
            model = MSCN(train_dataset.feat_dim, args.pred_num_hid, args.pred_num_out, args.mlp_num_hid,
                         return_rep=True)
    elif args.model_type == 'CEB_MSCN':
        pass
    elif args.model_type == 'DNN':
        model = MLP(in_ch=train_dataset.feat_dim, hid_ch=args.num_hid, out_ch=1, return_rep=True)
    else:
        raise NotImplementedError('Unsupported model type {}!'.format(args.model_type))
    if args.model_type == 'CEB_MSCN':
        return model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str, debug_dataset, debug_query_infos, debug_queries_str
    else:
        return model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str

# def prepare_ceb_mdoel_and_dataset(args):
#     seed_all()
#     assert args.is_join_query:

class TTTModel:
    def __init__(self, arg_list: list, debug_mode=False, logfilename=""):
        self.args = ttt_init_args(arg_list)
        args = self.args
        self.pred_stat = PredictionStatistics()
        if args.model_type == 'CEB_MSCN':
            self.model, self.train_dataset, self.train_query_infos, self.train_queries_str, self.test_dataset, self.test_query_infos, self.test_queries_str, self.debug_dataset, self.debug_query_infos, self.debug_queries_str = prepare_model_and_dataset(self.args)
            self.base = np.e
            self.ttt_trainer = TestTimeTrainer(self.args, self.model.net, optimizer=self.model.optimizer, model_wrapper=self.model)
        else:
            self.schema = load_schema(self.args.schema_name, self.args.data_path)
            self.query_parser = JoinQueryParser(schema=self.schema)
            self.model, self.train_dataset, self.train_query_infos, self.train_queries_str, self.test_dataset, self.test_query_infos, self.test_queries_str = prepare_model_and_dataset(self.args, self.schema)
            self.base = 2.0
            # self.model.to(self.args.device)
            self.ttt_trainer = TestTimeTrainer(self.args, self.model)
        self.debug_mode = debug_mode
        self.logfilename = logfilename
        # self.base = 2
        if self.debug_mode:
            self.logfile = open(logfilename, "w")
        self.server_info = {}
    
    def set_server_info(self, key = None, value = None):
        if key is not None:
            self.server_info[key] = value
    
    def get_server_info(self, key = None):
        if key is not None:
            return self.server_info[key]

    def load_model(self):
        args = self.args
        para = '_'.join( [ 'ttt', str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        assert args.model_save_path is not None
        assert os.path.exists(args.model_save_path)
        name = args.model_save_path + "/" + para + ".model"
        print("Model save path:", name)
        if os.path.exists(name):
            self.ttt_trainer.model.load_state_dict(torch.load(name))
        else:
            self.ttt_trainer.train(self.train_dataset)
            self.save_model(name)
        errors = self.ttt_trainer.test(self.test_dataset, dump=True)
        pred_stat = PredictionStatistics()
        if args.is_join_query:
            if args.skew_split_keys == 'template_no' or args.skew_split_keys == 'distri':
                # pred_stat.get_prediction_details(errors, test_query_infos, partition_keys='template_no')
                pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys="num_table")
            else:
                pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys="num_table")
        else:
            pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys="num_predicates")

    def save_model(self, name):
        torch.save( self.ttt_trainer.model.state_dict(), name )


    def predict(self, query_lines: list):
        if self.args.model_type == 'CEB_MSCN':
            # only preserve the first query, cause the rest are all empty (for placeholder)
            test_dataset = get_testds_from_sql(self.args, self.model, self.server_info,  [ query_lines[0] ])
        else:
            all_queries, all_cards, all_query_infos = list(), list(), list()
            # print("All query lines:") 
            # print(query_lines)
            for l in query_lines:
                l = l + "@1000"
                # table_ids, all_pred_list, join_infos, card = self.query_parser.parse_line(l)
                table_names, table_ids, all_pred_list, join_infos, card = self.query_parser.parse_line_return_name(l)
                all_queries.append((table_ids, all_pred_list, join_infos))
                all_cards.append(card)
                table_pairs = set([(join_info.t1_id, join_info.t2_id) for join_info in join_infos])
                is_multi_key = True if len(table_pairs) < len(join_infos) else False
                num_predicates = sum([len(pred_list) for pred_list in all_pred_list])
                all_query_infos.append(QueryInfo(num_table=len(table_ids), num_joins=len(join_infos),
                                        num_predicates=num_predicates, is_equal_join=True,
                                        is_multi_key=is_multi_key, template_no=-1, distri=-1, table_comb=",".join(table_names)))
            args = self.args
            if self.args.model_type == 'MSCN':
                test_dataset = TTTJoinQueryDataset(schema=self.schema, queries=all_queries, cards=all_cards, num_negs=self.args.num_negs, encoder_type='onehot') 
                # print(test_dataset.shape)
            else:
                test_dataset = TTTJoinQueryDataset(schema=self.schema, queries=all_queries, cards=all_cards,num_negs=self.args.num_negs, encoder_type='dnn')
           
        # test_loader = self.ttt_trainer.prepare_test_loader(test_dataset, padding_func_type="TypeA", batch_size = 1)
        test_loader = self.ttt_trainer.prepare_test_loader(test_dataset)
        outputs = list()
        self.ttt_trainer.model.eval()
        # print('==== for test ====')
        with torch.no_grad():
            for (x, y, x_neg) in test_loader:
                x, y, x_neg = recursive_to_device((x, y, x_neg), self.args.device)
                rep, output = self.ttt_trainer.model(x)
                outputs.append(output)
                # print(x["table"][0].sum(dim=0), x["join"][0].sum(dim=1), x["pred"][0].sum(dim=1))
                # o = output.cpu().detach().numpy()
                # print("predict = ", o)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.cpu().detach().numpy()
        # print(outputs)
        # print(self.test_dataset.X[0]["table"].sum(dim=0), self.test_dataset.X[0]["join"].sum(dim=1), self.test_dataset.X[0]["pred"].sum(dim=1))
        # x  = recursive_to_device( self.test_dataset.X[0], self.args.device)
        # rep, o = self.ttt_trainer.model(x)
        # o = o.cpu().detach().numpy()
        # print("predict: = ", o)
        if self.args.model_type == 'CEB_MSCN':
            outputs = self.model.featurizer.unnormalize_with_log(outputs, None) / np.log(2)
        return outputs

def save_model(self, name):
    torch.save( self.ttt_trainer.model.state_dict(), name)

def main(args):
    # Step 1: prepare model, datasets, pred_stat, and trainer. 
    if args.model_type == 'CEB_MSCN':
        model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str, debug_dataset, debug_query_infos, debug_queries_str = prepare_model_and_dataset(args)
    else:
        model, train_dataset, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str = prepare_model_and_dataset(args)
    # table_comb_dataset, table_comb_query_infos = prepare_datasets_with_table_comb(args, schema=None)
    pred_stat = PredictionStatistics()
    if args.model_type == 'CEB_MSCN':
        # for ceb_MSCN,
        # model.net => MSCN(NN)
        # model.optimizer => optimizer
        # model.featurizer => featurizer 
        ttt_trainer = TestTimeTrainer(args, model.net, optimizer=model.optimizer, model_wrapper=model)
        # to unnormalize the y value. 
        base = np.e
    else:
        ttt_trainer = TestTimeTrainer(args, model)
        base = 2.0

    # Step 2: train. 
    if args.model_save_path is not None and os.path.exists(args.model_save_path): 
        para = '_'.join( [ 'ttt', str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        name = args.model_save_path + '/' + para + ".model"
        print("Model save path:", name)

        if os.path.exists(name):
            ttt_trainer.model.load_state_dict(torch.load(name))
        else:
            ttt_trainer.train(train_dataset)
            torch.save(ttt_trainer.model.state_dict(), name)
    else:
        ttt_trainer.train(train_dataset)

    # Step 3: test && output error statistics. 
    if args.model_type == "CEB_MSCN" and args.dump_prediction == 1:
        errors, test_out = ttt_trainer.test(test_dataset, return_out=True)
        train_errors, train_out = ttt_trainer.test(train_dataset, return_out=True)
        test_out = np.power(base, test_out)
        train_out = np.power(base, train_out)

        train_d = {}
        train_d["Y"] = train_dataset.Y
        train_d["origin"] = train_dataset.label
        train_d["Y_"] = np.power(base, model.featurizer.unnormalize_with_log(train_dataset.Y , None))
        train_d["prediction"] = train_out
        
        test_d = {}
        test_d["Y"] = test_dataset.Y
        test_d["origin"] = test_dataset.label
        test_d["Y_"] = np.power(base, model.featurizer.unnormalize_with_log(test_dataset.Y, None))
        test_d["prediction"] = test_out
        
        log_predicts_stat(args, train_d,  test_d)
        debug_errors = ttt_trainer.test(debug_dataset)
    else:
        # pdb.set_trace()
        train_errors = ttt_trainer.test(train_dataset,dump=False)
        errors = ttt_trainer.test(test_dataset,dump=False)
        # train_errors = ttt_trainer.test(train_dataset)
    # table_comb_errors = ttt_trainer.test(table_comb_dataset)
    # processing some datasets infos.
    # pdb.set_trace()
    # train_X, train_Y, train_X_neg, train_info = train_dataset.X, train_dataset.Y, train_dataset.X_neg, train_dataset.info
    # test_X, test_Y, test_X_neg, test_info = test_dataset.X, test_dataset.Y, test_dataset.X_neg, test_dataset.info

    # equal_num = 0
    # for test_x in test_X:
    #     for train_x in train_X:
    #         equal = True
    #         for k in train_x.keys():
    #             v1 = train_x[k]
    #             v2 = test_x[k]
    #             if not torch.equal(v1, v2):
    #                 equal = False
    #                 break
    #         if equal:
    #             equal_num += 1
    # print("equal_num: ", equal_num, "total_sample: ", len(test_X))

                


    #errors = ttt_trainer.test_adapt(test_dataset)
    #errors = ttt_trainer.test_adapt_group(test_env.datasets)
    if args.is_join_query:
        if args.skew_split_keys == 'template_no' or args.skew_split_keys == 'distri':
            print("Qerror for testing queries:")
            pred_stat.get_prediction_details(errors, base, test_query_infos, partition_keys="template_no")
            print("Qerror for debuging queries:")
            pred_stat.get_prediction_details(debug_errors, base, debug_query_infos, partition_keys="num_table")
            # print("Qerror for training queries:")
            # pred_stat.get_prediction_details(train_errors, base, train_query_infos, partition_keys="num_table")
            # print("Qerror for table combine queries:")
            # pred_stat.get_prediction_details(table_comb_errors, table_comb_query_infos, partition_keys="table_comb")
        else:
            print("Qerror for testing queries:")
            pred_stat.get_prediction_details(errors, base, test_query_infos, partition_keys="template_no")
            print("Qerror for debuging queries:")
            pred_stat.get_prediction_details(debug_errors, base, debug_query_infos, partition_keys="num_table")
            # print("Qerror for training queries:")
            # pred_stat.get_prediction_details(train_errors, base, train_query_infos, partition_keys="num_table")
            # print("Qerror for table combine queries:")
            # pred_stat.get_prediction_details(table_comb_errors, base, table_comb_query_infos, partition_keys="table_comb")
    else:
        print("Qerror for testing queries:")
        pred_stat.get_prediction_details(errors, base, test_query_infos, partition_keys="num_predicates")
        print("Qerror for debuging queries:")
        pred_stat.get_prediction_details(debug_errors, base, debug_query_infos, partition_keys="num_table")
        # print("Qerror for training queries:")
        # pred_stat.get_prediction_details(train_errors, base, train_query_infos, partition_keys="num_predicates")
        # print("Qerror for table combine queries:")
        # pred_stat.get_prediction_details(table_comb_errors, base, table_comb_query_infos, partition_keys="table_comb")


if __name__ == "__main__":
    args = ttt_init_args(None)
    print("ttt_main.py: received args", args)
    import datetime
    starttime = datetime.datetime.now()
    if args.cli_mode == 'on':
        ttt_model = TTTModel(None)
        ttt_model.load_model()
        while True:
            input_line = input("# ttt @ cli mode\n$ ")
            card = int(input_line.split("@")[-1])
            input_line = "@".join( input_line.split("@")[0:-1])
            ests = ttt_model.predict([input_line])
            try:
                if torch.is_tensor(ests[0]):
                    print("[LOG]: Est log scale: {:>12.2f}".format(ests[0].item()))
                    print("[LOG]: Est result {:>12.0f}, Card {:>12.0f}, Qerr {:>12.2f}".format(2**ests[0].item(), card, 2**ests[0].item() / card))
                elif type(ests[0]) is np.ndarray:
                    print("[LOG]: Est log scale: {:>12.2f}".format(ests[0].item()))
                    print("[LOG]: Est result {:>12.0f}, Card {:>12.0f}, Qerr {:>12.2f}".format(2**ests[0].item(), card, 2**ests[0].item() / card))
                else:
                    print("[LOG]: Est log scale: {:>12.2f}".format(ests[0]))
                    print("[LOG]: Est result {:>12.0f}, Card {:>12.0f}, Qerr {:>12.2f}".format(2**ests[0], card, 2**ests[0] / card))
            except OverflowError as err:
                    print("[ERROR]: Overflow.")
            print("")
    if args.pred_mode == "on":
        assert args.input_file is not None and args.pred_file is not None
        input_lines = open(args.input_file, "r").readlines()
        cards = [ float(l.strip("\n").split("@")[-1]) for l in input_lines ]
        input_lines = [ "@".join(l.strip("\n").split("@")[0:-1]) for l in input_lines ]
        print("lines number", len(input_lines))
        ttt_model = TTTModel(None)
        ttt_model.load_model()
        ests = ttt_model.predict(input_lines)
        ests = [ ests, ests.copy() ]
        fw = open(args.pred_file, "w")
        print(len(ests))
        for r, c in zip(ests[0], cards):
            try:
                if torch.is_tensor(r):
                    fw.write( str(2 ** r.item()) + "," + str(2 ** r.item() / c) + "\n")
                elif type(r) is np.ndarray:
                    fw.write( str(2 ** r.item()) + "," + str(2 ** r.item() / c) + "\n")
                else:
                    fw.write( str(2 ** r) + "," + str(2 ** r / c) + "\n")
            except OverflowError as err:
                fw.write("overflow,truecard=" + str(c) + "\n")
        fw.close()
    else:
        main(args)
    endtime = datetime.datetime.now()
    print("[TIME]: ", (endtime - starttime).seconds)
