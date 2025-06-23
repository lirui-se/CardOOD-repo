import os
import random
from cardinality_estimation.featurizer import Featurizer
import math
from query_representation.utils import get_query_splits, load_qdata_parallel, load_qdata, func_sql_to_qrep, func_sql_to_qrep_with_ordered_sub, slices, extract_join_clause
from query_representation.query import parse_sql
from multiprocessing import Process, Queue
import numpy as np
import torch
 #from scripts.sql_to_qrep import func_sql_to_qrep
import json
import time
import yaml
# import pdb
from decimal import Decimal
from datetime import date
import pdb
import sys

DEBUG_API = True
GLOBAL_QS = 1
GLOBAL_QN = 100

def convert_decimal_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, tuple):
        return tuple([convert_decimal_to_float(item) for item in obj])
    elif isinstance(obj, date):
        return obj.isoformat()
    else:
        return obj

def featurizer_join_str_to_real_join(joinstr, aliases):
    # return joinstr
    join_tabs = joinstr.split("=")
    join_tabs.sort()
    real_join_tabs = []

    for jt in join_tabs:
        jt = jt.replace(" ", "")
        jsplits = jt.split(".")
        tab_alias = jsplits[0]
        if tab_alias not in aliases:
            print("tab alias not in self.aliases: ", tab_alias)
            # pdb.set_trace()
        real_jkey = aliases[tab_alias] + "." + jsplits[1]
        real_jkey = real_jkey.replace(" ", "")
        real_join_tabs.append(real_jkey)

    return "=".join(real_join_tabs)



def featurizer_update_workload_stats_interface(pid, qs, queue, feat_separate_alias, db_name):
    cmp_ops = set()
    regex_cols = set()
    regex_templates = set()
    tables = set()
    featurizer_joins = set()
    aliases = dict()
    for qrep in qs:
        cur_columns = []
        for node, info in qrep["join_graph"].nodes(data=True):
            if "pred_types" not in info:
                print("pred types not in info!")
            for i, cmp_op in enumerate(info["pred_types"]):
                cmp_ops.add(cmp_op)
                if "like" in cmp_op:
                    regex_cols.add(info["pred_cols"][i])
                    regex_templates.add(qrep["template_name"])
            tables.add(info["real_name"])

            if node not in aliases:
                aliases[node] = info["real_name"]
                node2 = "".join([n1 for n1 in node if not n1.isdigit()])
                aliases[node2] = info["real_name"]
            
            for col in info["pred_cols"]:
                cur_columns.append(col)

        joins = extract_join_clause(qrep["sql"])
        for join_str in joins:
            # get rid of whitespace
            # joinstr = joinstr.replace(" ", "")
            join_str = featurizer_join_str_to_real_join(join_str, aliases)
            if not feat_separate_alias and "dsb" not in db_name:
                join_str = ''.join([ck for ck in join_str if not ck.isdigit()])
            keys = join_str.split("=")
            keys.sort()
            keys = ",".join(keys)
            featurizer_joins.add(keys)
    queue.put( [ pid, (cmp_ops, regex_cols, regex_templates, tables, featurizer_joins, aliases) ])


def featurizer_update_workload_stats(featurizer, qreps, parallel=True):
    if not parallel:
        featurizer.update_workload_stats(qreps)
        return 
    nthreads = 40
    # pdb.set_trace()
    qreps_list = slices(qreps, nthreads)
    p_list = []
    queue = Queue()
    for i, q in enumerate(qreps_list):
        p = Process(target=featurizer_update_workload_stats_interface, args=(i, q, queue, featurizer.feat_separate_alias, featurizer.db_name))
        p_list.append(p)
        p.start()
    for pid, p in enumerate(p_list):
        cmp_ops, regex_cols, regex_templates, tables, joins, aliases = queue.get()[1]
        featurizer.cmp_ops.update(cmp_ops)
        featurizer.regex_cols.update(regex_cols)
        featurizer.regex_templates.update(regex_templates)
        featurizer.tables.update(tables)
        featurizer.joins.update(joins)
        featurizer.aliases.update(aliases)
    for pid, p in enumerate(p_list):
        p.join()
    print(f"api.py::featurizer_update_workload_stats::process(Main): Receive from {len(p_list)} processes.")
    print("max pred vals: {}".format(featurizer.max_pred_vals))
    print("Seen comparison operators: ", featurizer.cmp_ops)
    print("Tables: ", featurizer.tables)
    return



def featurizer_update_column_stats(args, cfg, featurizer, qreps):
    featdata_fn = os.path.join(cfg["data"]["query_dir"], "dbdata.json")
    if args.regen_featstats or not os.path.exists(featdata_fn):
        # we can assume that we have db stats for any column in the db
        print("regenerate column statistics.")
        featurizer.update_column_stats(qreps)
        ATTRS_TO_SAVE = ['aliases', 'cmp_ops', 'column_stats', 'joins',
                'max_in_degree', 'max_joins', 'max_out_degree', 'max_preds',
                'max_tables', 'regex_cols', 'tables', 'join_key_stats',
                'primary_join_keys', 'join_key_normalizers',
                'join_key_stat_names', 'join_key_stat_tmps'
                'max_tables', 'regex_cols', 'tables',
                'mcvs']

        featdata = {}
        for k in dir(featurizer):
            if k not in ATTRS_TO_SAVE:
                continue
            attrvals = getattr(featurizer, k)
            if isinstance(attrvals, set):
                attrvals = list(attrvals)
            featdata[k] = attrvals
        # pdb.set_trace()
        featdata = convert_decimal_to_float(featdata)
        if args.save_featstats:
            f = open(featdata_fn, "w")
            json.dump(featdata, f)
            f.close()
    else:
        import chardet
        print("load existing column statistics.")

        with open(featdata_fn, 'rb') as ff:
            raw_data = ff.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            print(f"Detected encoding: {encoding}")
        f = open(featdata_fn, "r", encoding=encoding)
        featdata = json.load(f)
        f.close()
        featurizer.update_using_saved_stats(featdata)
        # pdb.set_trace()

def get_featurizer(args, cfg, trainqs, valqs, testqs, all_evalqs):
# def get_featurizer(args, cfg, train_qfns, val_qfns, test_qfns, eval_qfns):
    # prepare statistics 
    feat_type = None
    card_type = "subplan"
    qdir_name = os.path.basename(cfg["data"]["query_dir"])
    bitmap_dir = cfg["data"]["bitmap_dir"]
    if args.alg in ["mscn", "mscn_joinkey", "mstn"]:
        feat_type = "set"
    else:
        feat_type = "combined"

    start_time = time.time()
    # step 1. creat empty featurizer
    featurizer = Featurizer(**cfg["db"])
    end_time = time.time()
    print(f"step 1: creat featurizer elapsed time = {end_time - start_time} s.")

    # step 2. update_column_stats. 
    # this will create self.column_stats.
    # the keys are columns extracted directly from sql query.
    # column_stats: { "i_category" : ..., "d1.d_year": ... }
    featurizer_update_column_stats(args, cfg, featurizer, trainqs + valqs + testqs + all_evalqs)
    end_time = time.time()
    print(f"step 2: update_column_stats elapsed time = {end_time - start_time} s.")

    # Look at the various keyword arguments to setup() to change the
    # featurization behavior; e.g., include certain features etc.
    # these configuration properties do not influence the basic statistics
    # collected in the featurizer.update_column_stats call; Therefore, we don't
    # include this in the cached version

    # step 3. ** converts the dictionary into keyword args
    # caution: in this step, will change featurizer.column_stats.keys() from 
    #       d1.d_year to d.d_year
    # print(cfg["featurizer"])
    featurizer.setup(
            **cfg["featurizer"],
            loss_func = cfg["model"]["loss_func_name"],
            featurization_type = feat_type,
            # bitmap_dir = cfg["data"]["bitmap_dir"],
            card_type = card_type
            )
    end_time = time.time()
    print(f"step 3: setup elapsed time = {end_time - start_time} s.")

    # step 4: just updates stuff like max-num-tables etc. for some implementation
    # things
    featurizer.update_max_sets(trainqs+valqs+testqs+all_evalqs)
    end_time = time.time()
    # 0.052 s
    print(f"step 4: Update_max_sets elapsed time = {end_time - start_time} s.")

    # step 5: update_workload_stats
    # featurizer.update_workload_stats(trainqs+valqs+testqs+all_evalqs)
    featurizer_update_workload_stats(featurizer, trainqs+valqs+testqs+all_evalqs)
    end_time = time.time()
    # 81.99 s
    print(f"step 5: Update_workload_stats elapsed time = {end_time - start_time} s.")

    # step 6: init_feature_mapping
    featurizer.init_feature_mapping()
    end_time = time.time()
    # 81.99s
    print(f"step 6: init_feature_mapping elapsed time = {end_time - start_time} s.")

    # step 7: update_ystats
    # feat_onlyseen_maxy = 1
    if cfg["featurizer"]["feat_onlyseen_maxy"]:
        featurizer.update_ystats(trainqs,
                max_num_tables=cfg["model"]["max_num_tables"])
    else:
        featurizer.update_ystats(trainqs+valqs+testqs+all_evalqs,
                max_num_tables = cfg["model"]["max_num_tables"])
    end_time = time.time()
    print(f"step 7: update_ystats elapsed time = {end_time - start_time} s.")

    # step 8: update_seen_preds
    featurizer.update_seen_preds(trainqs)
    end_time = time.time()
    # 83.465s
    print(f"step 8:update_seen_preds elapsed time = {end_time - start_time} s.")
    return featurizer

def load_pg_est(qreps, est_dir, qs, qn):
    f_dict = {}
    for qrep in qreps:
        fname = qrep["fname"]
        tpl_name = qrep["template_name"]
        if tpl_name not in f_dict.keys():
            f_dict[tpl_name] = {}
            tbl_name = est_dir + "/" + tpl_name + "_" + str(qs) + "_" + str(qn) + "-sub.tbl"
            est_name = est_dir + "/" + tpl_name + "_" + str(qs) + "_" + str(qn) + "-sub.est"
            ftbl = open(tbl_name, "r")
            fest = open(est_name, "r")
            tbl_lines = ftbl.readlines()
            est_lines = fest.readlines()
            tnum = len(qrep["join_graph"].nodes())
            qnum = 2 ** tnum - 1
            tbl_vs = tbl_lines
            for qi in range(qs + 1, qs + 1 + qn):
                est_vs = est_lines[(qi - 1) * qnum:qi * qnum]
                # print(tpl_name, qi, len(tbl_vs), len(est_vs))
                f_dict[tpl_name][qi] = {  frozenset(v1.strip("\n").split(",")): int(v2.strip("\n"))  for v1, v2 in zip(tbl_vs, est_vs) }
                # print(qi,f_dict[tpl_name][qi])
                # print(tpl_name, qi,len(f_dict[tpl_name][qi].keys()), qnum)
                assert len(f_dict[tpl_name][qi].keys()) == qnum
        qno = int(fname.split(".")[0].split("-")[1])
        te = f_dict[tpl_name][qno]
        for node in qrep["subset_graph"].nodes():
            key = frozenset(node)
            ekey = "expected"
            assert ekey in qrep["subset_graph"].nodes()[node]["cardinality"].keys()
            assert key in te.keys(), key
            qrep["subset_graph"].nodes()[node]["cardinality"][ekey] = te[key]
            # qrep["subset_graph"].nodes()[node]["cardinality"][ekey] = random.randint(100, 200)
            # print(node, key, te[key])


def dump_statistics(path, qreps, label, model_name):
    tn_list = [ ("fn", "tables",  "is_total_query", "card", "est")]
    for qrep in qreps:
        max_len = -1
        curr_d = {}
        node_names = list(qrep["subset_graph"].nodes())
        node_names.sort()
        for node in node_names:
            if len(node) > max_len:
                max_len = len(node)
        for node in node_names:
            if len(node) == max_len:
                tn_list.append( ( qrep["name"], node, True, qrep["subset_graph"].nodes()[node]["cardinality"]["actual"],qrep["subset_graph"].nodes()[node]["cardinality"]["expected"]))
            else:
                tn_list.append( (qrep["name"], node, False, qrep["subset_graph"].nodes()[node]["cardinality"]["actual"], qrep["subset_graph"].nodes()[node]["cardinality"]["expected"]) )
    fn = os.path.join(path, model_name + "_"+ label+ ".csv")
    with open(fn, "w") as f:
        for t in tn_list:
            f.write( ";".join( [str(e) for e in t] ) + "\n")

def update_statistics(path, d, label, model_name):
    fn = os.path.join(path, model_name + "_" +label + ".csv")
    os.rename( fn, fn + ".bak")
    tn_list = []
    with open(fn + ".bak", "r") as f:
        lines = f.readlines()
        for line in lines:
            tn_list.append( line.strip("\n").split(";") )
    valid_q = 0
    card_col_index = 3
    for k in d.keys():
        temp = d[k]
        d[k] = []
        for e in temp:
            if isinstance(e, torch.Tensor):
                d[k].append(e.item())
            elif isinstance(e, np.ndarray):
                d[k].append(e.item())
            else:
                d[k].append(e)
    for t in tn_list[1:]:
        card = int(t[card_col_index])
        if card >= 10 ** 8:
            pass
        elif card <= 1:
            pass
        else:
            valid_q += 1
    with open(fn, "w") as f:
        # process the head. 
        t = tn_list[0]
        tn_list = tn_list[1:]
        keys = list(d.keys())
        for k in keys:
            t.append(k)
        f.write(";".join( t ) + "\n")
        # check if valid.
        for k in keys:
            assert len(d[k]) == valid_q or len(d[k]) == len(tn_list), k + "," + str(len(d[k])) + "," + str(len(tn_list)) + "," + "\n".join( [ str(e) for e in tn_list ] )
            if len(d[k]) == valid_q:
                for i, t in enumerate(tn_list):
                    card = int(t[card_col_index])
                    if card >= 10 ** 8 or card <= 1:
                        d[k].insert(i, -1)
                    else:
                        pass
        # process the data
        for i, t in enumerate(tn_list):
            for k in keys:
                t.append(str( d[k][i]))
                if k == 'origin':
                    assert int(d[k][i]) == int(t[card_col_index])
                if k == 'Y_':
                    assert int(d[k][i])< 0 or int(d[k][i]) / (int(t[card_col_index])) < 1.01, k + "," + str(d[k][i]) + "," + str(t[card_col_index])
            f.write(";".join( t ) + "\n")

def get_statistics(qreps):
    tq = 0
    valid_tq = 0
    valid_qrep = 0
    valid_qrep_tq = 0
    tpl_set = set()
    for qrep in qreps:
        is_qrep_valid = True
        tq += len(qrep["subset_graph"].nodes())
        for node in qrep["subset_graph"].nodes():
            if qrep["subset_graph"].nodes[node]["cardinality"]["actual"] < 10 ** 8:
                valid_tq += 1
            else:
                is_qrep_valid = False
        if is_qrep_valid:
            valid_qrep += 1
            valid_qrep_tq += len(qrep["subset_graph"].nodes())
        tpl_set.add(qrep["template_name"])
    # return [ selected queries, all queries, valid qreps, all queries for valid qrep, valid all queries ] 
    # if timeout qrep has non-timeout subquery, then valid_qrep_tp < valid_tq
    # else, valid_qrep_tp = valid_tq
    return [len(qreps), tq, valid_qrep, valid_qrep_tq, valid_tq, tpl_set]
    # print(f"len(qreps) = {len(qreps)}, total queries = {tq}, valid queries = {valid_tq}")


def reorganize_qfns(qfns_list, qs=None, qn=None):
    # reorganize the file names.
    d = {}
    for qfns in qfns_list:
        tmp_name = qfns.split("/")[-1].split("-")[0]
        if tmp_name not in d.keys():
            d[tmp_name] = {"list": [], "dict": {}}
        d[tmp_name]["list"].append(qfns)
    res_list = []
    for k in d.keys():
        # sort by qno.
        for x in d[k]["list"]:
            key=int(x.split("/")[-1].split(".")[0].split("-")[1])
            d[k]["dict"][key] = x
        # d[k].sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[1]))
        if qs is not None and qn is not None:
            for ind in range(qs, qs + qn):
                if ind not in d[k]["dict"].keys():
                    print("Error! ", ind, " not found in tpl ", k)
                else:
                    res_list.append(d[k]["dict"][ind])
            # print(d[k][0:qn])
        else:
            res_list += d[k]["list"]
    return res_list

def get_testds_from_sql(args, model, server_info, query_lines):
    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())


    est_dir = cfg["data"]["pg_est_dir"]
    qstart = cfg["data"]["pg_est_qs"]
    qnum = cfg["data"]["pg_est_qn"]

    cachef = open(cfg["data"]["pg_cache_file"], "r")
    flines = cachef.readlines()
    cachef.close()
    flines = [ l.strip("\n") for l in flines ]
    arr = flines[0].split(",") 
    tpl=arr[0]
    qno=arr[1]
    print("processing tpl", tpl, "query id (from 0)", qno)
    cachef = open(cfg["data"]["pg_cache_file"], "w")
    cachef.write(tpl + "," + str(int(qno) + 1) + "\n")
    cachef.flush()
    cachef.close()


    qreps = []
    for l in query_lines:
        qrep = func_sql_to_qrep_with_ordered_sub(l, server_info, tpl=tpl,qno=qno, est_dir=est_dir, qstart=qstart,qnum=qnum)
        qreps.append(qrep)
    if args.model_name == 'ttt':
        # when ceb dataset class try to encode a qrep, it will sort the subqueries, so 
        # that destroy the original order (which is our intend)
        evalds = model.get_ttt_testds(qreps, num_negs=args.num_negs, sort_sub=False)
        # for x, y in zip(evalds.X, evalds.Y):
        #     print(x["table"].sum(dim=0), x["join"].sum(dim=1), x["pred"].sum(dim=1), y)
    else:
        # when ceb dataset class try to encode a qrep, it will sort the subqueries, so 
        # that destroy the original order (which is our intend)
        evalds = model.get_testds(qreps, sort_sub=False)
    return evalds

def log_predicts_stat(args, predictions_train, predictions_val):
    # read configurations 
    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())
    # print(yaml.dump(cfg, default_flow_style=False))

    # import pdb; pdb.set_trace()
    # Step 1: read qrep data
    eval_qdirs = cfg["data"]["eval_query_dir"].split(",")
    train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])

    if cfg["db"]["db_name"] == "dsb":
        # qn => number of queries for each template. max: 4000
        train_qfns = reorganize_qfns(train_qfns, qs=GLOBAL_QS, qn=GLOBAL_QN)
        for ei, _ in enumerate(eval_qfns):
            eval_qfns[ei] = reorganize_qfns(eval_qfns[ei], qs=GLOBAL_QS, qn=GLOBAL_QN)

    update_statistics(cfg["data"]["query_dir"], predictions_train, "train", args.model_name)
    # dump_statistics(testqs, "test")
    update_statistics(cfg["data"]["eval_query_dir"], predictions_val, "val", args.model_name)




# mainly a wrapper for main.py
def load_model_datasets(args):
    # read configurations 
    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())
    # print(yaml.dump(cfg, default_flow_style=False))

    # import pdb; pdb.set_trace()
    # Step 1: read qrep data
    eval_qdirs = cfg["data"]["eval_query_dir"].split(",")
    train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])

    if cfg["db"]["db_name"] == "dsb":
        # qn => number of queries for each template. max: 4000
        train_qfns = reorganize_qfns(train_qfns, qs=GLOBAL_QS, qn=GLOBAL_QN)
        # print("\n".join(train_qfns))
        for ei, _ in enumerate(eval_qfns):
            eval_qfns[ei] = reorganize_qfns(eval_qfns[ei], qs=GLOBAL_QS, qn=GLOBAL_QN)
            # print("\n".join(eval_qfns[ei]))

    # split="1"
    # tpl1 = "025"
    # tpl2 = "100"
    # tpl3 = "102"
    # test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-3.pkl' ]



    # split="2"
    # tpl1 = "050"
    # tpl2 = "091"
    # tpl3 = "099"
    # test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-3.pkl' ]

    # tpl1 = "018"
    # tpl2 = "072"
    # tpl3 = "084"
    # tpl4 = "100"
    # tpl5 = "102"
    # test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-3.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-1.pkl', 
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-2.pkl',
    #               f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-3.pkl' ]

#     split="3"
#     # --------------------------------- testing templates. --------------------------------------------
#     tpl1 = "018"
#     tpl2 = "027"
#     tpl3 = "050"
#     test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-1.pkl', 
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-2.pkl',
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-3.pkl',
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-1.pkl', 
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-2.pkl',
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-3.pkl',
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-1.pkl', 
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-2.pkl',
#                   f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-3.pkl' ]
# 
    # --------------------------------- training templates. -------------------------------------------
    #  tpl1 = "019"
    #  tpl2 = "025"
    #  tpl3 = "040"
    #  tpl4 = "072"
    #  tpl5 = "084"
    #  test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-1.pkl', 
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-2.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl1}/tpl{tpl1}-3.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-1.pkl', 
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-2.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl2}/tpl{tpl2}-3.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-1.pkl', 
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-2.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl3}/tpl{tpl3}-3.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-1.pkl', 
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-2.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl4}/tpl{tpl4}-3.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-1.pkl', 
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-2.pkl',
    #                f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/train/tpl{tpl5}/tpl{tpl5}-3.pkl' ]

    split="4"
    tpl1 = "018"
    tpl2 = "027"
    tpl3 = "050"
    test_qfns = [ f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-1.pkl', 
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-2.pkl',
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl1}/tpl{tpl1}-3.pkl',
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-1.pkl', 
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-2.pkl',
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl2}/tpl{tpl2}-3.pkl',
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-1.pkl', 
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-2.pkl',
                  f'/home/lirui/codes/PG_CardOOD/CEB/ceb/queries/dsb_split_{split}/test/tpl{tpl3}/tpl{tpl3}-3.pkl' ]




    # step 2. load qreps. 
    args.parallel_load = False
    evalqs = []
    for eval_qfn in eval_qfns:
        evalqs.append(load_qdata_parallel(eval_qfn, parallel=args.parallel_load))
    trainqs = load_qdata_parallel(train_qfns, parallel=args.parallel_load)
    testqs = load_qdata_parallel(test_qfns, parallel=args.parallel_load)
    valqs = load_qdata_parallel(val_qfns, parallel=args.parallel_load)
    all_evalqs = []
    for e0 in evalqs:
        all_evalqs += e0

    train_stats = get_statistics(trainqs)
    test_stats  = get_statistics(testqs)
    val_stats   = get_statistics(valqs)
    # for qs in all_evalqs:
    #     eval_stats_  = get_statistics([qs])
    #     print(f"{eval_stats_[0]}, {eval_stats_[1]}, {eval_stats_[2]}, {eval_stats_[3]}, {eval_stats_[4]}")
    eval_stats = get_statistics(all_evalqs)

    print(f"Selected queries: {train_stats[0]} train, {test_stats[0]} test, {val_stats[0]} val, {eval_stats[0]} eval.")
    print(f"All queries: {train_stats[1]} train, {test_stats[1]} test, {val_stats[1]} val, {eval_stats[1]} eval.")
    print(f"Valid qreps: {train_stats[2]} train, {test_stats[2]} test, {val_stats[2]} val, {eval_stats[2]} eval.")
    print(f"Totol qreps: {train_stats[3]} train, {test_stats[3]} test, {val_stats[3]} val, {eval_stats[3]} eval.")
    print(f"Valid total queries: {train_stats[4]} train, {test_stats[4]} test, {val_stats[4]} val, {eval_stats[4]} eval.")
    print(f"tpl set: {train_stats[5]} train, {test_stats[5]} test,{val_stats[5]} val, {eval_stats[5]} eval.")

    
    if cfg["data"]["load_est"] == 1:
        load_pg_est(trainqs, cfg["data"]["pg_est_dir"], cfg["data"]["pg_est_qs"], cfg["data"]["pg_est_qn"])
        load_pg_est(all_evalqs, cfg["data"]["pg_est_dir"], cfg["data"]["pg_est_qs"], cfg["data"]["pg_est_qn"])

    if args.dump_prediction == 1:
        dump_statistics(cfg["data"]["query_dir"], trainqs, "train", args.model_name)
        # dump_statistics(testqs, "test")
        dump_statistics(cfg["data"]["eval_query_dir"], all_evalqs, "val", args.model_name)
        # sys.exit(0)

    # import pdb; pdb.set_trace()
    # Step 3: Initialize features 
    if args.alg in ["xgb", "fcnn", "mscn", "mscn_joinkey", "mstn"]:
        featurizer = get_featurizer(args, cfg, trainqs, valqs, testqs, all_evalqs)
        # featurizer = get_featurizer(args, cfg, train_qfns, val_qfns, test_qfns, eval_qfns)
    else:
        featurizer = None
    from cardinality_estimation import get_alg

    # Step 3: Initialize model 
    if args.model_name == 'ttt':
        # return type of model net is slightly different. will return hidden output + final output 
        model = get_alg(args.alg, cfg, return_rep=True, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        # ds should contain x, x_neg, y.
        trainds = model.get_ttt_trainds(trainqs, num_negs=args.num_negs)
        evalds = model.get_ttt_testds(all_evalqs, num_negs=args.num_negs)
        debugds = model.get_ttt_testds(testqs, num_negs=args.num_negs)
        # pdb.set_trace()

        # for x, y in zip(evalds.X, evalds.Y):
        #     print(x["table"].sum(dim=0), x["join"].sum(dim=1), x["pred"].sum(dim=1), y)
    elif args.model_name == 'coral':
        model = get_alg(args.alg, cfg, return_rep=True, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        train_env = model.get_train_env(trainqs)
        evalds = model.get_testds(all_evalqs)
        debugds = model.get_testds(testqs)
    elif args.model_name == 'irm':
        model = get_alg(args.alg, cfg, return_rep=False, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        train_env = model.get_train_env(trainqs)
        evalds = model.get_testds(all_evalqs)
        debugds = model.get_testds(testqs)
    elif args.model_name == 'dann':
        model = get_alg(args.alg, cfg, return_rep=True, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        trainds = model.get_grp_trainds(trainqs)
        evalds = model.get_testds(all_evalqs)
        debugds = model.get_testds(testqs)
    elif args.model_name == 'groupdro':
        model = get_alg(args.alg, cfg, return_rep=False, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        trainds = model.get_grp_trainds(trainqs)
        evalds = model.get_testds(all_evalqs)
        debugds = model.get_testds(testqs)
    else:
        model = get_alg(args.alg, cfg, return_rep=False, lr=args.learning_rate, mb_size=args.batch_size)
        model.add_featurizer(featurizer)
        trainds = model.get_trainds(trainqs)
        evalds = model.get_testds(all_evalqs)
        debugds = model.get_testds(testqs)
    # if DEBUG_API:
    #     for x, y, info in trainds:
    #         pass

    if args.model_name == 'irm' or args.model_name == 'coral':
        model.init_net(train_env.datasets[0][0])
    else:
        model.init_net(trainds[0])
    # pdb.set_trace()
    if args.model_name == 'irm' or args.model_name == 'coral':
        return model, train_env, evalds, debugds
    elif args.model_name == 'ttt':
        return model, trainds, evalds, debugds 
    elif args.model_name == 'erm':
        return model, trainds, evalds, debugds 
    elif args.model_name == 'dann':
        return model, trainds, evalds, debugds 
    else:
        return model, trainds, evalds
