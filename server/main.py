# from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import getopt
import socketserver
import json
import struct
import sys
import time
import os
import math
import re
import numpy as np
import torch
from itertools import combinations

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
TODO: Should modify the path to your own path
"""
PG_OPTIMIZER_INDEX = 0
SCHEMA_NAME = 'imdb'
MODEL_NAME = 'nngp'
#MODEL_NAME = 'MSCN'
DATA_PATH=''
QUERY_PATH=''
TRAIN_SQL_PATH=''
TEST_SQL_PATH=''
OUTPUT_PATH=''
MANUAL_SPLIT=''
LOG_PATH=''
CHUNK_SIZE=64
USE_AUX=False
Q_ERROR_THRESHOLD=0
COEF_VAR_THRESHOLD=0
TRAIN_SET_ID='2_4_0.9'
MODEL_NAME='erm'
SKEW_SPLIT_KEYS = 'distri'
IS_JOIN_QUERY = 'True'
MODEL_TYPE = 'DNN'
NUM_NEGS = '2'
EPOCHS = '140'
LEARNING_RATE = '1e-3'
BATCH_SIZE = '32'
MODEL_SAVE_PATH=""
PREDICT_FILE=None
CONFIG_FILE=None
# print(sys.argv)


# LOGs
# DEBUG_MODE=True
DEBUG_MODE=False
#DEBUG_MODE=True
# read query from file instead of reading from PG
READ_QUERY_FROM_FILE=False
query_file=None
# dump details in sql2nngpformat
DUMP_TO_NNGP=False
# dump subqueries in nngp format to txt file
# subqueries_file: all subqueries. total_file: only the largest query
DUMP_SUB=False
subqueries_file=None
total_file=None
subqueries_card_file=None
total_card_file=None
total_x_file=None

# chosen subqueries && valid subqueries
DUMP_CHOSEN, DUMP_VALID =False, False
chosen_file, valid_file=None, None

if DEBUG_MODE:
    ################ READ_QUERY_FROM_FILE ###########
    # READ_QUERY_FROM_FILE, query_file =True, "num_table_4.txt"
    ################ DUMP_TO_NNGP ###################
    DUMP_TO_NNGP=True
    ################ DUMP_SUB #######################
    DUMP_SUB, subqueries_file, total_file = True, "sub.txt", "total.txt"
    subqueries_card_file="sub.card"
    total_card_file="total.card"
    total_x_file="x"
    ################ DUMP_CHOSEN ###################
    # DUMP_CHOSEN=True
    # chosen_file="chosen.txt"
    # DUMP_VALID=True
    # valid_file="valid.txt"

# model = examplepkg.Estimator("imdb", "data path", "train query path")
# 
# model.load_model()
# model.predict([1, 3, 5])
# 
# sys.exit(-1)

opts,args = getopt.getopt(sys.argv[1:], '-s:-d:-q:-a:-e:-l:-m:-i:-k:-j:-t:-n:-p:-r:-b:-o:-c:', ['schema=', 'datapath=', 'querypath=', 'train_sql_path=', 'test_sql_path=', 'logpath=', 'manual_split=', 'model=','id=', 'skew_split_keys=', 'is_join_query=', 'model_type=', 'num_negs=', 'epochs=', 'learning_rate=', 'batch_size=', 'model_save_path=', 'predict_file=', 'config='])

for opt_name, opt_value in opts:
    if opt_name in ('-s', '--schema'):
        SCHEMA_NAME = opt_value
    if opt_name in ('-d', '--datapath'):
        DATA_PATH = opt_value
    if opt_name in ('-q', '--querypath'):
        QUERY_PATH = opt_value
    if opt_name in ('-a', '--train_sql_path'):
        TRAIN_SQL_PATH = opt_value
    if opt_name in ('-e', '--test_sql_path'):
        TEST_SQL_PATH = opt_value
    if opt_name in ('-l', '--logpath'):
        LOG_PATH = opt_value
    if opt_name in ('--manual_split'):
        MANUAL_SPLIT = opt_value
    if opt_name in ('-m', '--model'):
        MODEL_NAME = opt_value
    if opt_name in ('-i', '--id'):
        Train_Set_Id = opt_value
    if opt_name in ('-k', '--skew_split_keys'):
        SKEW_SPLIT_KEYS = opt_value
    if opt_name in ('-j', '--is_join_query'):
        IS_JOIN_QUERY = opt_value
    if opt_name in ('-t', '--model_type'):
        MODEL_TYPE = opt_value
    if opt_name in ('-n', '--num_negs'):
        NUM_NEGS = opt_value
    if opt_name in ('-p', '--epochs'):
        EPOCHS = opt_value
    if opt_name in ('-r', '--learning_rate'):
        LEARNING_RATE = opt_value
    if opt_name in ('-b', '--batch_size'):
        BATCH_SIZE = opt_value
    if opt_name in ('-o', '--model_save_path'):
        MODEL_SAVE_PATH = opt_value
    if opt_name in ('-c', '--predict_file'):
        PREDICT_FILE = opt_value
    if opt_name in ('--config'):
        CONFIG_FILE = opt_value
        

class Basepred:
    def __init__(self):
        self.cname = ""
        self.hb = -1
        self.lb = -1

class Joinpredeq:
    def __init__(self):
        self.t1 = ""
        self.t2 = ""
        self.c = ""

class TrueCardModel:
    def __init__(self, args_list):
        assert len(args_list) == 2
        self.predict_file = args_list[1]
        f = open(self.predict_file, "r")
        self.cards = f.readlines()
        self.cards  = [ float(c.strip("\n")) for c in self.cards ]
        f.close()
        self.cardi = 0
        self.server_info = {}
    def load_model(self):
        pass
    def predict(self, plans):
        size = len(plans)
        res_cards = self.cards[self.cardi: self.cardi + size]
        self.cardi = self.cardi + size
        res = np.array(res_cards)
        return res
    def set_server_info(self, key=None, value=None):
        if key is not None:
            self.server_info[key] = value
    def get_server_info(self, key=None):
        if key is not None:
            return self.server_info[key]

class NNGPModel:
    def __init__(self):
        from ood import TTTModel, ERMModel, CORALModel, SERVER_DANNModel, MIXUPModel, IRMModel, GROUPDROModel, MASKModel
        # t = torch.tensor([1, 2, 3])
        # t = t.to('cuda')
        # print("is cuda initialized?", t)
        print("SCHEMA = ", SCHEMA_NAME, "DATA_PATH = ", DATA_PATH, "QUERY_PATH = ", QUERY_PATH, "TRAIN_SQL_PATH = ", TRAIN_SQL_PATH, "TEST_SQL_PATH = ", TEST_SQL_PATH, "LOG_PATH = ", LOG_PATH, "MODEL = ", MODEL_NAME, "SKEW_SPLIT_KEYS = ", SKEW_SPLIT_KEYS, "IS_JOIN_QUERY = ", IS_JOIN_QUERY, "MODEL_TYPE = ", MODEL_TYPE, 'NUM_NEGS = ', NUM_NEGS)
        print(PREDICT_FILE)
        # LOG files.
        if READ_QUERY_FROM_FILE:
            self.fquery = open(query_file, "r")
        if DUMP_SUB:
            assert subqueries_file is not None
            assert total_file is not None
            assert subqueries_card_file is not None
            assert total_card_file is not None
            self.fsub = open(subqueries_file, "w")
            self.ftotal = open(total_file, "w")
            self.fsubcard = open(subqueries_card_file, "w")
            self.ftotalcard = open(total_card_file, "w")
        if DUMP_CHOSEN:
            assert chosen_file is not None
            self.fchosen = open(chosen_file, "w")
        if DUMP_VALID:
            assert valid_file is not None
            self.fvalid = open(valid_file, "w")
        # patterns.
        self.__qpat = re.compile(r'^.*(SELECT|select)(.*)(FROM|from)(.*)(WHERE|where)(.*)$')
        self.__lepat = re.compile(r'(\s*[a-zA-Z_]+\.[a-zA-Z_]+\s*)<=?(\s*[\-0-9]+\s*)')
        self.__gepat = re.compile(r'(\s*[a-zA-Z_]+\.[a-zA-Z_]+\s*)>=?(\s*[\-0-9]+\s*)')
        self.__eqnumpat = re.compile(r'^(\s*[a-zA-Z_]+\.[a-zA-Z_]+\s*)=(\s*[\-0-9\.]+\s*)$')
        self.__eqpat = re.compile(r'^(\s*[a-zA-Z_]+\.[a-zA-Z_]+\s*)=(\s*[a-zA-Z_]+\.[a-zA-Z_]+\s*)$')
        if MODEL_NAME == 'erm':
            print('[LOG]: ', 'Initializing ERM model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'erm', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE ]
            self.current_model = ERMModel(self.args) 
            self.current_model.load_model()
        elif MODEL_NAME == 'ttt':
            print('[LOG]: ', 'Initializing TTT model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'ttt', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = TTTModel(self.args, DEBUG_MODE, total_x_file) 
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'coral':
            print('[LOG]: ', 'Initializing CORAL model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'coral', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = CORALModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'mask':
            print('[LOG]: ', 'Initializing MASK model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'mask', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = MASKModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'dann':
            print('[LOG]: ', 'Initializing DANN model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'dann', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = SERVER_DANNModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'mixup':
            print('[LOG]: ', 'Initializing MIXUP model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'mixup', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = MIXUPModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'irm':
            print('[LOG]: ', 'Initializing IRM model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'irm', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH , '--config', CONFIG_FILE]
            self.current_model = IRMModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'groupdro':
            print('[LOG]: ', 'Initializing groupdro model.')
            self.predict_file = open(PREDICT_FILE, "w")
            self.args = [ '--is_join_query', IS_JOIN_QUERY, '--skew_split_keys', SKEW_SPLIT_KEYS, '--model_type', MODEL_TYPE, '--model_name', 'groupdro', '--schema_query_path', QUERY_PATH, '--schema_data_path', DATA_PATH, '--train_sql_path', TRAIN_SQL_PATH, '--test_sql_path', TEST_SQL_PATH, '--num_negs', NUM_NEGS, '--epochs', EPOCHS, '--learning_rate', LEARNING_RATE, '--batch_size', BATCH_SIZE, '--model_save_path', MODEL_SAVE_PATH, '--config', CONFIG_FILE]
            self.current_model = GROUPDROModel(self.args)
            self.current_model.load_model()
            print('[LOG]: ', 'Initialize complete.')
        elif MODEL_NAME == 'true':
            print('[LOG]: ', 'Initialize true card model.')
            self.args = [ '--predict_file', PREDICT_FILE]
            self.current_model = TrueCardModel(self.args)
            self.current_model.load_model()
        else:
            assert False, "Model name not recognized."
        print('[LOG]: ', 'Initialize complete.')

    def encodeSQL(self, s, revridmap):
        # if self.current_model.args.model_type == 'CEB_MSCN':
        if MODEL_TYPE == 'CEB_MSCN':
            self.current_model.set_server_info(key="revridmap", value=revridmap)
            s = s.strip("\n").strip(" ").strip(";").strip("explain").strip("EXPLAIN")
            print(s)
            res = [ s ]
            for i in range(0, 2 ** len(revridmap) - 2):
                res.append( None )
            return res
            # return self.current_model.model.featurizer.
        elif MODEL_NAME == 'true':
            res = [ s]
            for i in range(0, 2 ** len(revridmap) - 2):
                res.append( None )
            return res
        else:
            return sql2nngpformat(s, revridmap)


    def sql2nngpformat(self, s, revridmap):
        # print("[LOG]: SQL format: ", s)
        m = self.__qpat.match(s)
        # self.ts = m.group(4).strip().split(',')
        # self.ts = map(lambda x: x.strip(), ts)
        self.qs = []
        self.ts = []
        self.predmap = {}
        self.joinpreds = []
        self.eqclass = []

        if DUMP_TO_NNGP:
            print("[LOG]: Pattern match: ", m.group(1), m.group(2),m.group(3), m.group(4), m.group(5),  m.group(6))

        ll = len(revridmap.keys())
        for li in range(1, ll + 1):
            self.ts.append(revridmap[str(li)])
        # print("[LOG]: Base tables: ", self.ts)
        self.rinfo = m.group(6).strip(';').strip().lower().split('and')
        self.rinfo = map(lambda x: x.strip(), self.rinfo)
        self.rinfo = [ ri for ri in self.rinfo ]
        if DUMP_TO_NNGP:
            print("[LOG]: Rinfo: ")
            for r in self.rinfo:
                print(r)
        
        # print(self.rinfo)
        # print(self.rinfo)
        self.predmap = {}
        self.joinpreds = []
        # print("[LOG]: rinfo: ", self.rinfo)
        for t in self.ts:
            self.predmap[t] = []
        # print(type(self.rinfo))
        # print(len(self.rinfo))
        for rii, ri in enumerate(self.rinfo):
            if self.__eqnumpat.match(ri) is not None:
                m = self.__eqnumpat.match(ri)
                if DUMP_TO_NNGP:
                    print(rii, ri, " matches eqnumpat .", m.group(1), m.group(2))
                lpart = m.group(1).strip()
                rpart = m.group(2).strip()
                tcname = []
                lb = int(rpart)
                hb = int(rpart)
                tcname = lpart.split(".")
                hflag = True
                lflag = True
                if tcname[0] not in self.predmap.keys():
                    self.predmap[tcname[0]] = []
                found = False 
                for p in self.predmap[tcname[0]]:
                    if p.cname == tcname[1]:
                        if hflag:
                            p.hb = hb
                        elif lflag:
                            p.lb = lb
                        found = True
                        break
                if not found:
                    bp = Basepred()
                    bp.cname = tcname[1]
                    bp.hb = hb
                    bp.lb = lb
                    self.predmap[tcname[0]].append(bp)
            elif self.__lepat.match(ri) is not None:
                m = self.__lepat.match(ri)
                if DUMP_TO_NNGP:
                    print(rii,ri, " matches lepat .", m.group(1), m.group(2))
                lpart = m.group(1).strip()
                rpart = m.group(2).strip()
                tcname = []
                hb = -1
                lb = -1
                hflag = False
                lflag = False
                try:
                    hb = int(rpart)
                    tcname = lpart.split(".")
                    hflag = True
                    lflag = False
                except:
                    # We only support column <= const or column >= const
                    lb = int(lpart)
                    tcname = rpart.split(".")
                    hflag = False
                    lflag = True
                if tcname[0] not in self.predmap.keys():
                    self.predmap[tcname[0]] = []
                found = False 
                for p in self.predmap[tcname[0]]:
                    if p.cname == tcname[1]:
                        if hflag:
                            p.hb = hb
                        elif lflag:
                            p.lb = lb
                        found = True
                        break
                if not found:
                    bp = Basepred()
                    bp.cname = tcname[1]
                    bp.hb = hb
                    bp.lb = lb
                    self.predmap[tcname[0]].append(bp)
            elif self.__gepat.match(ri) is not None:
                m = self.__gepat.match(ri)
                if DUMP_TO_NNGP:
                    print(rii, ri, " matches gepat .", m.group(1), m.group(2))
                lpart = m.group(1).strip()
                rpart = m.group(2).strip()
                tcname = []
                hb = -1
                lb = -1
                hflag = False
                lflag = False
                try:
                    lb = int(rpart)
                    tcname = lpart.split(".")
                    lflag = True
                    hflag = False
                    # print('[LOG]: > detected. ', tcname, lb)
                except:
                    # We only support column <= const or column >= const
                    hb = int(lpart)
                    tcname = rpart.split(".")
                    lflag = False
                    hflag = True
                if tcname[0] not in self.predmap.keys():
                    self.predmap[tcname[0]] = []
                found = False 
                for p in self.predmap[tcname[0]]:
                    if p.cname == tcname[1]:
                        if hflag:
                            p.hb = hb
                        elif lflag:
                            p.lb = lb
                        found = True
                        break
                if not found:
                    bp = Basepred()
                    bp.cname = tcname[1]
                    bp.hb = hb
                    bp.lb = lb
                    self.predmap[tcname[0]].append(bp)
            elif self.__eqpat.match(ri) is not None:
                m = self.__eqpat.match(ri)
                if DUMP_TO_NNGP:
                    print(rii,  ri, " matches eqpat .", m.group(1), m.group(2))
                lpart = m.group(1).strip()
                rpart = m.group(2).strip()
                # Only table1.column = table2.column is permitted.
                # column name must be the same.
                ltcname = lpart.split(".")
                rtcname = rpart.split(".")
                if ltcname[1] != rtcname[1]:
                    print("Parse error: column name in eq join pred must be the same.")
                    print(ri)
                    print(ltcname, rtcname)
                    sys.exit(0)
                # jp = Joinpredeq()
                # jp.t1 = ltcname[0]
                # jp.t2 = rtcname[0]
                # jp. = ltcname[1]
                # self.joinpreds.append(jp)
                eqcfound = False
                ltfound = False
                rtfound = False
                for eqc in self.eqclass:
                    ltfound = False
                    rtfound = False
                    for eq in eqc:
                        if eq[0] == ltcname[0] and eq[1] == ltcname[1]:
                            eqcfound = True
                            ltfound = True
                        if eq[0] == rtcname[0] and eq[1] == rtcname[1]:
                            eqcfound = True
                            rtfound = True
                    if ltfound and rtfound:
                        break
                    if ltfound and not rtfound:
                        eqc.append((rtcname[0], rtcname[1]))
                        break
                    if not ltfound and rtfound:
                        eqc.append((ltcname[0], ltcname[1]))
                        break
                    if not ltfound and not rtfound:
                        continue
                if not eqcfound:
                    eqc = []
                    eqc.append((ltcname[0], ltcname[1]))
                    eqc.append((rtcname[0], rtcname[1]))
                    self.eqclass.append(eqc)
            else:
                assert False, f"rinfo [ {ri}] can not be matched!\nsql: {s} \n revridmap: {revridmap}"
        
        # DUMP base preds. 
        if DUMP_TO_NNGP:
            for tname in self.ts:
                for bp in self.predmap[tname]:
                    print(bp.cname, bp.hb, bp.lb)
        # generate joinpreds
        self.joinpreds = []
        # for eqc in self.eqclass:
        #     for j in range(1, len(eqc)):
        #         i = 0
        #         jp = Joinpredeq()
        #         jp.t1 = eqc[i][0]
        #         jp.t2 = eqc[j][0]
        #         jp.c = eqc[i][1]
        #         self.joinpreds.append(jp)
        #         if DUMP_TO_NNGP:
        #             print(jp.t1, jp.t2, jp.c)

        # # TYPE 1: Expand all the join predicates.
        # for eqc in self.eqclass:
        #     for i in range(0, len(eqc) - 1):
        #         for j in range(i + 1, len(eqc)):
        #             jp = Joinpredeq()
        #             jp.t1 = eqc[i][0]
        #             jp.t2 = eqc[j][0]
        #             jp.c = eqc[i][1]
        #             self.joinpreds.append(jp)
        #             if DUMP_TO_NNGP:
        #                 print(jp.t1, jp.t2, jp.c)

        # TYPE 2: Only preserve n - 1 join predicates
        for eqc in self.eqclass:
            for i in range(0, len(eqc) - 1):
                j = i + 1
                jp = Joinpredeq()
                jp.t1 = eqc[i][0]
                jp.t2 = eqc[j][0]
                jp.c = eqc[i][1]
                self.joinpreds.append(jp)
                if DUMP_TO_NNGP:
                    print(jp.t1, jp.t2, jp.c)

        # Queries
        #print(self.ts)
        qs = []
        for li in range(1, ll + 1):
            cc = combinations(self.ts, li)
            for c in cc:
                # Create the query string
                flist = ",".join(c)
                bpredstr = "@".join([ "#".join([ bp.cname + "," + str(bp.hb) + "," + str(bp.lb) for bp in self.predmap[tname]]) for tname in c])
                jpreds = filter( lambda jpred: jpred.t1 in c and jpred.t2 in c, self.joinpreds)
                jpredstr =  "#".join( [ jpred.t1 + "," + jpred.t2 + "," + jpred.c for jpred in jpreds ] ) 
                qs.append(flist + "@" + bpredstr + "@" + jpredstr)
    
        if DUMP_TO_NNGP:
            print("[LOG]: Convert results:")
            print(qs[-1])
        return qs

    def out_chosen_subqueries(self, message):
        cards = message[0]["ChosenSubqueries"]
        if DUMP_CHOSEN:
            for i in range(0, len(self.qs)):
                if cards[i] > 0:
                    self.fchosen.write(self.qs[i] + "@" + str(cards[i]) + "\n")
                else:
                    self.fchosen.write("#\n")
            self.fchosen.flush()
        print("Output chosen subqueries done.")

    def out_subqueries(self, message):
        inds = message[0]["ValidSubqueries"]
        if DUMP_VALID:
            # print(inds)
            for i in range(0, len(self.qs)):
                if inds[i] == 1:
                    self.fvalid.write(self.qs[i] + "@123\n")
                else:
                    self.fvalid.write("#\n")
            self.fvalid.flush()
        print("Output valid subqueries done.")

    def enumerate_subset(self, message):
        # Given an SQL query, enumerate all possible subplans
        # print("Enumerating subset...")
        # print(message)
        pstr = message[0]["Plan"]
        revridmap  = message[2]["Revrelidmap"]
        qs = self.encodeSQL(pstr, revridmap)
        if READ_QUERY_FROM_FILE:
            assert self.fquery is not None
            line = self.fquery.readline().strip("\n")
            arr = line.split("@")
            qs = [ "@".join(arr[:-1]) for q in qs ]
        # if MODEL_NAME == 'nngp':
        # print("Send model to predict:")
        # for q in qs:
        #     print(q)
        res = self.current_model.predict(qs)
        res = res.reshape(res.shape[0])

        if DUMP_SUB:
            self.fsub.write("\n".join(qs) + "\n")
            self.ftotal.write(qs[-1] + "\n")
            self.fsubcard.write("\n".join( [ str(2 ** e) for e in res ] ) + '\n')
            self.ftotalcard.write( str( 2 ** res[-1]) + "\n" )
            self.fsub.flush()
            self.ftotal.flush()
            self.fsubcard.flush()
            self.ftotalcard.flush()

        # res[res < 0] = 0.001
        # res = np.relu(res)
        # Add a fake uncertainty. 
        # filter large prediction
        # note that here in res is log representation
        res[res > 60] = 60
        res = [ res, res.copy()]
        # else: 
        #    res = self.current_model.train(qs) 
        print("Prediction shape:", res[0].shape)
        # print("Prediction result:")
        # print(",".join([ str(r) for r in res[0]]))
        if MODEL_NAME != "true":
            for r in res[0]:
                self.predict_file.write(str(r) + "\n")
            self.predict_file.flush()
        # f = open(OUTPUT_PATH + "/mean.txt", "a")
        # f2 = open(OUTPUT_PATH + "/std.txt", "a")
        # f.write("# Start of one query\n")
        # f.write("\n".join([ str(2 ** r) for r in res[0]]) + "\n")
        # f2.write("# Start of one query\n")
        # f2.write("\n".join([ str(r) for r in res[1]]) + "\n")
        return res
        # print("Done.")

    def predict(self, message):
        if self.current_model is None:
            return math.nan

        # print("Predict...")
        # print(message)
        res = self.current_model.predict([message[0]["Plan"]])
        print("Predict result:", res)
        print("Shape of predict result:", res.shape)
        # print("res[0][0]:", np.absolute(res[0][0]))
        # print("res[0][1]:", np.absolute(res[0][1]))
        return np.absolute(res[0][0])
        # print("Done.")
        return 0

    def load_model(self):
        try:
            print("Loading model..")
            self.current_model.load_model()
            print("Done.")

        except Exception as e:
            print("Faild to load NNGP model. \nException:", sys.exec_info()[0])
            raise e

class NNGPJSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            str_buf += self.request.recv(1024).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            
            null_loc = str_buf.find("\n")
            #if (null_loc := str_buf.find("\n")) != -1:
            if null_loc != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON:", json_msg)
                        break


                
class NNGPJSONHandler(NNGPJSONTCPHandler):
    def setup(self):
        self.__messages = []
    
    def handle_json(self, data):
        if "final" in data:
            message_type = self.__messages[0]["type"]
            self.__messages = self.__messages[1:]

            if message_type == "query":
                result = self.server.NNGP_model.enumerate_subset(self.__messages)
                print("Prediciton complete. Prepare to send back to pg.")
                self.request.sendall(struct.pack("%ud" % (len(result[0]) + len(result[1])), *result[0], *result[1]))
                self.request.close()
            elif message_type == "predict":
                # print(self.__messages)
                result = self.server.NNGP_model.predict(self.__messages)
                # print(result)
                self.request.sendall(struct.pack("d", result))
                self.request.close()
            elif message_type == "valid":
                self.server.NNGP_model.out_subqueries(self.__messages)
            elif message_type == "chosen":
                self.server.NNGP_model.out_chosen_subqueries(self.__messages)
            elif message_type == "load model":
                path = self.__messages[0]["path"]
                self.server.NNGP_model.load_model()
            else:
                print("Unknown message type:", message_type)
            
            return True

        self.__messages.append(data)
        return False


def start_server(listen_on, port):
    # from examplepkg import Estimator
    # from constants import (PG_OPTIMIZER_INDEX, DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH, SCHEMA_NAME, DATA_PATH, TRAIN_SQL_PATH, CHUNK_SIZE, USE_AUX, Q_ERROR_THRESHOLD, COEF_VAR_THRESHOLD, PLOGLEVEL)
    # from estimator import Estimator
    # import Estimator,MSCN,TreeLSTM
    # sys.exit(-1)
    model = NNGPModel()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), NNGPJSONHandler) as server:
        server.NNGP_model = model
        server.serve_forever()


if __name__ == "__main__":
    print("entering main...")
    from multiprocessing import Process

    port = 8001
    # port = 8000
    listen_on = "localhost"
    # t = torch.tensor([1, 2, 3])
    # t = t.to('cuda')
    # print("cuda...", t)

    # import multiprocessing
    # multiprocessing.set_start_method('spawn')

    print(f"Listening on {listen_on} port {port}")
   
    # multiprocessing.set_start_method('spawn')
    server = Process(target=start_server, args=[listen_on, port])
    
    print("Spawning server process...")
    print("After Spawning Server Process...(test)")
    server.start()
