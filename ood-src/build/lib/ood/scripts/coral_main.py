from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.schema import load_schema, load_table
from db.parser import QueryParser, JoinQueryParser
from db.table import QueryInfo
from encoder.transform import *
from model.model import MLP, MSCN, MSCNJoin
# from util import train_test_val_split, uneven_train_test_split, PredictionStatistics, seed_all
from ceb import get_testds_from_sql, load_model_datasets, log_predicts_stat
from util import train_test_val_split, uneven_train_test_split, query_pattern_train_test_split, PredictionStatistics, seed_all
from asset.coral import DeepCoralTrainer
from asset.base import recursive_to_device
from scripts.irm_main import prepare_table_datasets, prepare_schema_datasets, prepare_query_pattern_datasets, prepare_ceb_datasets, get_query_infos, get_query_infos_from_env

 
def init_args(arg_list: list):
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
    parser.add_argument("--model_name", type=str, default='coral', help="ttt,coral,erm...")
    parser.add_argument("--support_skew", type=str, default='False')
    parser.add_argument('--skew_ratio', type=float, default=None,
                        help='fraction of small queries in the training set')
    parser.add_argument('--num_negs', type=int, default=5,
                        help='Do not use it. It is provided for compatibility.')

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

    parser.add_argument('--lambda_coral', type=float, default=0.1,# 0.5 by default
                        help='weight of coral loss')

    # input dir
    parser.add_argument("--is_join_query", type=str, default='True')
    parser.add_argument("--skew_split_keys", type=str, default="num_table")
    parser.add_argument("--table_name", type=str, default='forest')
    parser.add_argument("--table_query_path", type=str,
                        default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')
    parser.add_argument("--table_data_path", type=str, default='/home/kfzhao/data/UCI')
    parser.add_argument("--schema_name", type=str, default='imdb_simple', help='tpcds')
    parser.add_argument("--schema_data_path", type=str, default='/home/kfzhao/data/rdb/imdb_clean')
    parser.add_argument("--schema_query_path", type=str,
                        default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_824_FP')
    parser.add_argument("--manual_split", type=str, default='False')
    parser.add_argument("--train_sql_path", type=str, default=None)
    parser.add_argument("--test_sql_path", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--parallel_load", type=str, default=None)

    # ceb arguments
    parser.add_argument("--config", type=str, required=False, default="configs/config.yaml")
    parser.add_argument("--alg", type=str, required=False, default="mscn")
    parser.add_argument("--regen_featstats", type=int, required=False, default=0)
    parser.add_argument("--save_featstats", type=int, required=False, default=0)
    parser.add_argument("--use_saved_feats", type=int, required=False, default=1)

    # debug arguments
    parser.add_argument("--debug_test", type=int, required=False, default=0)
    parser.add_argument("--debug_end", type=int, required=False, default=0)



    # logging arguments
    parser.add_argument("--wandb_tags", type=str, required=False, default=None, help="additional tags for wandb logs")
    parser.add_argument("--result_dir", type=str, required=False, default="./results")
    parser.add_argument("--eval_fns", type=str, required=False, default="ppc,qerr")

    # log arguments
    parser.add_argument("--dump_prediction", type=int, required=False, default=0)

    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    assert args.support_skew in [ 'True', 'true', 'False', 'false', 1, 0 ], "support_skew must be True or true or 1 or False or false or 0"
    assert args.manual_split in [ 'True', 'true', 'False', 'false', 1, 0 ], "manual_split must be True or true or 1 or False or false or 0"
    assert args.is_join_query in [ 'True', 'true', 'False', 'false', 1, 0 ], "is_join_query must be True or true or 1 or False or false or 0"
    args.support_skew = True if args.support_skew in [ 'True', 'true', 1 ] else False
    args.manual_split = True if args.manual_split in [ 'True', 'true', 1 ] else False
    args.is_join_query = True if args.is_join_query in [ 'True', 'true', 1 ] else False

    if args.is_join_query:
        args.query_path, args.data_path = args.schema_query_path, args.schema_data_path
        print("is join query.")
    else:
        args.query_path, args.data_path = args.table_query_path, args.table_data_path
        print("is not join query.")
    if args.skew_ratio is not None and not args.support_skew :
        print("[ERROR]: If you specify skew_ratio, you should turn support_skew on (--support_skew=True.")
        sys.exit(-1)
    return args

def prepare_model_and_dataset(args, schema=None):
    seed_all()
    if args.is_join_query:
        if args.model_type == "CEB_MSCN":
            model, train_env, train_query_infos, train_queries_str, test_dataset, test_query_infos, test_queries_str = prepare_ceb_datasets(args)
        else:
            if args.support_skew:
                train_env, train_query_infos, test_dataset, test_query_infos = prepare_schema_datasets(args)
            else:
                train_env, train_query_infos, test_dataset, test_query_infos = prepare_query_pattern_datasets(args, schema=schema)

    else:
        train_env, train_query_infos, test_dataset, test_query_infos = prepare_table_datasets(args)

    if args.model_type == 'MSCN':
        if args.is_join_query:
            print("table dim={}, pred dim={}, join_dim={}".format(test_dataset.table_feat_dim,
                                                                  test_dataset.pred_feat_dim, test_dataset.join_feat_dim))
            model = MSCNJoin(test_dataset.table_feat_dim, args.table_num_hid, args.table_num_out,
                         test_dataset.pred_feat_dim, args.pred_num_hid, args.pred_num_out,
                         test_dataset.join_feat_dim, args.join_num_hid, args.join_num_out, args.mlp_num_hid, return_rep=True)
        else:
            model = MSCN(test_dataset.feat_dim, args.pred_num_hid, args.pred_num_out, args.mlp_num_hid, return_rep=True)
    elif args.model_type == 'DNN':
        model = MLP(in_ch=test_dataset.feat_dim, hid_ch=args.num_hid, out_ch=1, return_rep=True)
    elif args.model_type == 'CEB_MSCN':
        pass
    else:
        raise NotImplementedError('Unsupported model type {}'.format(args.model_type))
    return model, train_env, train_query_infos, test_dataset, test_query_infos

class CORALModel:
    def __init__(self, arg_list: str):
        self.args = init_args(arg_list)
        args = self.args
        self.pred_stat = PredictionStatistics()
        if args.model_type == 'CEB_MSCN':
            self.model, self.train_env, self.train_query_infos, self.test_dataset, self.test_query_infos = prepare_model_and_dataset(self.args)
            self.coral_trainer = DeepCoralTrainer(args, self.model.net, optimizer = self.model.optimizer, model_wrapper = self.model)
            self.base = np.e
        else:
            self.schema = load_schema(self.args.schema_name, self.args.data_path)
            self.query_parser = JoinQueryParser(schema=self.schema)
            self.model, self.train_env, self.train_query_infos, self.test_dataset, self.test_query_infos = prepare_model_and_dataset(self.args, self.schema)
            self.pred_stat = PredictionStatistics()
            self.coral_trainer = DeepCoralTrainer(self.args, self.model)
            self.base = 2.0
        self.server_info = {}

    def set_server_info(self, key = None, value = None):
        if key is not None:
            self.server_info[key] = value
    
    def get_server_info(self, key = None):
        if key is not None:
            return self.server_info[key]

        
    def load_model(self):
        args = self.args
        if not args.support_skew:
            para = '_'.join( [ 'coral', str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        else:
            para = '_'.join( [ 'coral', str(args.skew_ratio), str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        assert args.model_save_path is not None
        assert os.path.exists(args.model_save_path)
        # name = "/home/lirui/codes/PG_CardOOD/CardOOD/ood/run/coral_models/" + para + ".model"
        name = args.model_save_path + "/" + para + ".model"
        print("Model save path:", name)
        # Currently do not support skew. Supporting skew requires rename the models.
        if os.path.exists(name):
            self.coral_trainer.model.load_state_dict(torch.load(name))
        else:
            self.coral_trainer.train(self.train_dataset)
            self.save_model(name)
        errors = self.coral_trainer.test(self.test_dataset)
        pred_stat = PredictionStatistics()
        if args.is_join_query:
            if args.skew_split_keys == 'template_no' or args.skew_split_keys == 'distri':
                pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys='num_table')
            else:
                pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys='num_table')
        else:
            pred_stat.get_prediction_details(errors, self.base, self.test_query_infos, partition_keys='num_predicates')

    def save_model(self, name):
        torch.save( self.coral_trainer.model.state_dict(), name)


    def predict(self, query_lines: list):
        if self.args.model_type == 'CEB_MSCN':
            test_dataset = get_testds_from_sql(self.args, self.model, self.server_info, [ query_lines[0] ])
        else:
            all_queries, all_cards, all_query_infos = list(), list(), list()
            for l in query_lines:
                # Append a dummy card to allow parser. 
                l = l + "@1000"
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
            if args.model_type == 'MSCN':
                test_dataset = JoinQueryDataset(schema=self.schema, queries=all_queries, cards=all_cards, encoder_type='onehot')
            else:
                test_dataset = JoinQueryDataset(schema=self.schema, queries=all_queries, cards=all_cards, encoder_type='dnn')

        test_loader = self.coral_trainer.prepare_test_loader(test_dataset)
        # test_loader = self.coral_trainer.prepare_test_loader(test_dataset, padding_func_type="TypeB")
        # test_loader = self.coral_trainer.prepare_test_loader(test_dataset, padding_func_type="TypeB")
        outputs, labels = list(), list()
        self.coral_trainer.model.eval()
        with torch.no_grad():
            for(x, y) in test_loader:
                x, y = recursive_to_device((x, y), self.args.device)
                rep, output = self.coral_trainer.model(x)
                outputs.append(output)
                labels.append(y)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.cpu().detach().numpy()
        if self.args.model_type == 'CEB_MSCN':
            outputs = self.model.featurizer.unnormalize_with_log(outputs, None) / np.log(2)
        return outputs
 

def main(args):
    model, train_env, train_query_infos, test_dataset, test_query_infos = prepare_model_and_dataset(args)
    pred_stat = PredictionStatistics()
    if args.model_type == 'CEB_MSCN':
        coral_trainer = DeepCoralTrainer(args, model.net, optimizer = model.optimizer, model_wrapper = model)
        base = np.e
    else:
        coral_trainer = DeepCoralTrainer(args, model)
        base = 2.0

    if args.model_save_path is not None and os.path.exists(args.model_save_path):
        if not args.support_skew: 
            para = '_'.join( [ 'coral', str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        else:
            para = '_'.join( [ 'coral', str(args.skew_ratio), str(args.skew_split_keys), str(args.model_type), str(args.num_negs), str(args.epochs), str('{:.0e}'.format(args.learning_rate)), str(args.batch_size) ] )
        name = args.model_save_path + '/' + para + ".model"
        print("Model save path:", name)
        if os.path.exists(name):
            coral_trainer.model.load_state_dict(torch.load(name))
        else:
            coral_trainer.train(train_env.datasets)
            torch.save(coral_trainer.model.state_dict(), name)
    else:
        coral_trainer.train(train_env.datasets)

    if args.model_type == "CEB_MSCN" and args.dump_prediction == 1:
        errors, test_out = coral_trainer.test(test_dataset, return_out=True)
        train_errors, train_out = coral_trainer.test(train_dataset, return_out=True)
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
        debug_errors = coral_trainer.test(debug_dataset)
    else:
        errors = coral_trainer.test(test_dataset)

    # for d in train_env.datasets:
    #     train_errors.append(coral_trainer.test(d))

    if args.is_join_query:
        print("Qerror for testing queries:")
        pred_stat.get_prediction_details(errors, base, test_query_infos, partition_keys="template_no")
        # print("Qerror for training queries:")
        # for de in train_errors:
        #     pred_stat.get_prediction_details(de, train_query_infos, partition_keys="template_no")
    else:
        print("Qerror for testing queries:")
        pred_stat.get_prediction_details(errors, base, test_query_infos, partition_keys="num_predicates")
        # print("Qerror for training queries:")
        # for de in train_errors:
        #     pred_stat.get_prediction_details(de, train_query_infos, partition_keys="num_predicates")
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    args = init_args(None)
    print(args)
    import datetime
    starttime = datetime.datetime.now()
    main(args)
    endtime = datetime.datetime.now()
    print("[TIME]: ", (endtime - starttime).seconds)
