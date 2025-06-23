import torch
from torch.utils import data
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
import time
import copy
import multiprocessing as mp
import math
import pickle

from query_representation.utils import *

import pdb

DEBUG_MODE = False

DATASET_DEBUG_MODE = True

def load_mscn_features(qfeat_fn):
    with open(qfeat_fn, "rb") as f:
        data = pickle.load(f)
    return data["x"], data["y"]

def load_ttt_mscn_features(qfeat_fn):
    with open(qfeat_fn, "rb") as f:
        data = pickle.load(f)
    return data["x"], data["x_neg"], data["y"]


def QueryDataset__get_ttt_feature_vectors(queue_rec, queue_send):
    pid, dataset, samples, cur_qidx, cur_qi = queue_rec.get()
    qidx = cur_qidx
    qi = cur_qi
    X= []
    X_neg=[]
    Y=[]
    sample_info=[]

    for qrep in samples:
        x,x_neg,y,cur_info = dataset._get_ttt_query_features(qrep, qidx, qi)
        qidx += len(y)
        qi += 1
        X += x
        X_neg += x_neg
        Y += y
        sample_info += cur_info
    if dataset.featurizer.featurization_type == "combined":
        X = to_variable(X, requires_grad=False).float()
        X_neg = to_variable(X_neg, requires_grad=False).float() 
    elif dataset.featurizer.featurization_type == "set":
        # don't need to do anything, since padding+masks is handled later
        pass
    # queue_send.put( [pid, (X, X_neg, Y, sample_info) ] )
    # queue_send.put( [pid, ([1],[2],[3],[4]) ] )
    queue_send.put( [pid, (pickle.dumps(X), pickle.dumps(X_neg), pickle.dumps(Y), pickle.dumps(sample_info)) ] )
    # res = queue_send.get()
    # empty = queue_send.empty()
    # print(f"dataset.py::QueryDataset__get_ttt_feature_vectors::process({pid}): Processing {len(samples)} samples, X size = {len(X)}.")

# def get_feature_vectors_parallel_interface(featdir, queue_rec, queue_send):
#     pid, qhl = queue_rec.get()
#     all_X = []
#     all_Y = []
#     for qhash in qhl:
#         featfn = os.path.join(featdir, qhash) + ".pkl"
#         x, y = load_mscn_features(featfn)
#         all_X = all_X + x
#         all_Y = all_Y + y
#     # sample0 = [ 1 for _ in range(20108032)]
#     # sample1 = [ 1 for _ in range(20108032)]
#     # queue_send.put( [pid, sample0, sample1 ] )
#     queue_send.put( [pid, pickle.dumps(all_X), pickle.dumps(all_Y)] )
#     # if DATASET_DEBUG_MODE:
#     #     print(f"dataset.py::get_feature_vectors_parallel_interface::process({pid}): size = {len(qhl)}, {len(all_X)}, {len(all_Y)}, load finished.")
# 


def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    else:
        arr = Variable(arr, requires_grad=requires_grad)
    return arr

def mscn_collate_fn_together(data):
    start = time.time()
    alldata = defaultdict(list)

    for di in range(len(data)):
        try:
            for feats in data[di][0]:
                for k,v in feats.items():
                    alldata[k].append(v)
        except Exception:
            for feats in data[di]:
                for k,v in feats.items():
                    alldata[k].append(v)

    xdata = {}
    for k,v in alldata.items():
        if k == "flow":
            if len(v[0]) == 0:
                xdata[k] = v
            else:
                xdata[k] = torch.stack(v)
        else:
            xdata[k] = torch.stack(v)

    ys = [d[1] for d in data]
    ys = torch.cat(ys)
    infos = [d[2] for d in data]
    return xdata,ys,infos

def mscn_collate_fn(data):
    '''
    TODO: faster impl.
    '''
    start = time.time()
    alltabs = []
    allpreds = []
    alljoins = []

    flows = []
    ys = []
    infos = []

    maxtabs = 0
    maxpreds = 0
    maxjoins = 0

    for d in data:
        alltabs.append(d[0]["table"])
        if len(alltabs[-1]) > maxtabs:
            maxtabs = len(alltabs[-1])

        allpreds.append(d[0]["pred"])
        if len(allpreds[-1]) > maxpreds:
            maxpreds = len(allpreds[-1])

        alljoins.append(d[0]["join"])
        if len(alljoins[-1]) > maxjoins:
            maxjoins = len(alljoins[-1])

        flows.append(d[0]["flow"])
        ys.append(d[1])
        infos.append(d[2])

    tf,pf,jf,tm,pm,jm = pad_sets(alltabs, allpreds,
            alljoins, maxtabs,maxpreds,maxjoins)

    flows = torch.stack(flows).float()

    ys = to_variable(ys, requires_grad=False).float()
    data = {}
    data["table"] = tf
    data["pred"] = pf
    data["join"] = jf
    data["flow"] = flows
    data["tmask"] = tm
    data["pmask"] = pm
    data["jmask"] = jm

    return data,ys,infos

def _handle_set_padding(features, max_set_vals):

    if len(features) == 0:
        return None, None

    features = np.vstack(features)
    num_pad = max_set_vals - features.shape[0]
    assert num_pad >= 0

    mask = np.ones_like(features).mean(1, keepdims=True)
    features = np.pad(features, ((0, num_pad), (0, 0)), 'constant')
    mask = np.pad(mask, ((0, num_pad), (0, 0)), 'constant')
    features = np.expand_dims(features, 0)
    mask = np.expand_dims(mask, 0)

    return features, mask

def pad_sets(all_table_features, all_pred_features,
        all_join_features, maxtabs, maxpreds, maxjoins):

    tf = []
    pf = []
    jf = []
    tm = []
    pm = []
    jm = []

    assert len(all_table_features) == len(all_pred_features) == len(all_join_features)

    for i in range(len(all_table_features)):
        table_features = all_table_features[i]
        # print(len(table_features))
        pred_features = all_pred_features[i]
        join_features = all_join_features[i]

        pred_features, predicate_mask = _handle_set_padding(pred_features,
                maxpreds)
        table_features, table_mask = _handle_set_padding(table_features,
                maxtabs)
        join_features, join_mask = _handle_set_padding(join_features,
                maxjoins)

        if table_features is not None:
            tf.append(table_features)
            tm.append(table_mask)

        if pred_features is not None:
            pf.append(pred_features)
            pm.append(predicate_mask)

        if join_features is not None:
            jf.append(join_features)
            jm.append(join_mask)

    tf = to_variable(tf,
            requires_grad=False).float().squeeze()
    extra_dim = len(tf.shape)-1
    tm = to_variable(tm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    pf = to_variable(pf,
            requires_grad=False).float().squeeze()
    extra_dim = len(pf.shape)-1
    pm = to_variable(pm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    jf = to_variable(jf,
            requires_grad=False).float().squeeze()
    extra_dim = len(jf.shape)-1

    jm = to_variable(jm,
            requires_grad=False).byte().squeeze().unsqueeze(extra_dim)

    if maxtabs == 1:
        tm = tm.unsqueeze(1)

    if maxjoins == 1:
        jm = jm.unsqueeze(1)

    if maxpreds == 1:
        pm = pm.unsqueeze(1)

    return tf, pf, jf, tm, pm, jm


class QueryDataset(data.Dataset):
    def __init__(self, samples, featurizer,
            load_query_together, load_padded_mscn_feats=False,
            max_num_tables = -1,
            subplan_mask = None,
            feat_dir = "./mscn_features",
            join_key_cards=False,
            with_info = True,
            sort_sub=True):
        '''
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subplans.
        @load_query_together: each sample will be a list of all the feature
        vectors belonging to all the subplans of a query.
        @subplan_mask: [], same length as samples;
        '''
        # yaml: load_query_together = 0
        self.load_query_together = load_query_together
        self.with_info = with_info
        self.sort_sub = sort_sub
        if self.load_query_together:
            self.start_idxs, self.idx_lens = self._update_idxs(samples)

        self.load_padded_mscn_feats = load_padded_mscn_feats
        self.featurizer = featurizer
        self.max_num_tables = max_num_tables

        if "whitening" in self.featurizer.ynormalization:
            self.featurizer.update_means(samples)

        self.join_key_cards = join_key_cards

        self.save_mscn_feats = False
        self.subplan_mask = subplan_mask

        # yaml: load_padded_mscn_feats = 1
        if self.load_padded_mscn_feats:
            fkeys = list(dir(self.featurizer))
            fkeys.sort()
            attrs = ""
            for k in fkeys:
                attrvals = getattr(featurizer, k)
                if not hasattr(attrvals, "__len__") and \
                    "method" not in str(attrvals):
                    attrs += str(k) + str(attrvals) + ";"

            attrs += "padded"+str(self.load_padded_mscn_feats)
            # print(attrs)
            self.feathash = deterministic_hash(attrs)
            self.featdir = os.path.join(feat_dir, str(self.feathash))
            if os.path.exists(self.featdir):
                print("features saved before")
                if not self.featurizer.use_saved_feats:
                    print("going to delete feature directory, and save again")
                    # delete these and save again
                    self.save_mscn_feats = True
                    os.remove(self.featdir)
            else:
                print("Features not saved before.")

            if self.featurizer.use_saved_feats:
                self.save_mscn_feats = True
                make_dir(feat_dir)
                make_dir(self.featdir)

        # yaml: max_num_tables = -1
        if self.max_num_tables != -1:
            self.save_mscn_feats = False
            self.featurizer.use_saved_feats = False

        # yaml: load_query_together = 0
        if self.load_query_together:
            self.save_mscn_feats = False

        # shorthands
        self.ckey = self.featurizer.ckey
        self.minv = self.featurizer.min_val
        self.maxv = self.featurizer.max_val
        self.feattype = self.featurizer.featurization_type

        # TODO: we may want to avoid this, and convert them on the fly. Just
        # keep some indexing information around.
        # self.X, self.Y, self.info = self._get_feature_vectors(samples)
        # self.X, self.Y, self.info = self._get_feature_vectors_par(samples)
        self.X, self.Y, self.info = self._get_feature_vectors_switch(samples)
        
        # memory the original card. 
        self.label = []
        for s in samples:
            node_name = list(s["subset_graph"].nodes())
            node_name.sort()
            for node in node_name:
                self.label.append( (s["subset_graph"].nodes()[node]["cardinality"]["actual"]) )

        if self.load_query_together:
            self.num_samples = len(samples)
        else:
            self.num_samples = len(self.X)

    def _update_idxs(self, samples):
        qidx = 0
        idx_starts = []
        idx_lens = []
        for i, qrep in enumerate(samples):
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            idx_starts.append(qidx)
            nodes = list(qrep["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            idx_lens.append(len(nodes))
            qidx += len(nodes)
        return idx_starts, idx_lens

    def _load_mscn_features(self, qfeat_fn):
        # with open(qfeat_fn, "rb") as f:
        #     data = pickle.load(f)
        # return data["x"], data["y"]
        return load_mscn_features(qfeat_fn)

    def _save_mscn_features(self, x,y,qfeat_fn):
        data = {"x":x, "y":y}
        with open(qfeat_fn, "wb") as f:
            pickle.dump(data, f)

    def _get_sample_info(self, qrep, dataset_qidx, query_idx):
        sample_info = []
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, node in enumerate(node_names):
            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(node)
            cur_info["template_no"] = qrep["template_name"]
            sample_info.append(cur_info)

        return sample_info

    def _get_query_features_joinkeys(self, qrep, sbitmaps,
            jbitmaps,
            dataset_qidx,
            query_idx):
        X = []
        Y = []
        sample_info = []

        edges = list(qrep["subset_graph"].edges())
        edges.sort(key = lambda x: str(x))

        for edge_idx, subset_edge in enumerate(edges):

            # find the appropriate node from which this edge starts
            subset = subset_edge[1]
            if subset == SOURCE_NODE:
                continue
            ## not needed
            larger_subset = subset_edge[0]
            assert len(larger_subset) > len(subset)

            x,y = self.featurizer.get_subplan_features_joinkey(qrep,
                    subset, subset_edge, bitmaps=sbitmaps,
                    join_bitmaps=jbitmaps)

            if self.featurizer.featurization_type == "set" \
                and self.load_padded_mscn_feats:
                start = time.time()
                tf,pf,jf,tm,pm,jm = \
                    pad_sets([x["table"]], [x["pred"]], [x["join"]],
                            self.featurizer.max_tables,
                            self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = tf
                x["join"] = jf
                x["pred"] = pf
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm

            x["flow"] = to_variable(x["flow"], requires_grad=False).float()

            X.append(x)
            Y.append(y)

            cur_info = {}
            cur_info["num_tables"] = len(subset)
            cur_info["dataset_idx"] = dataset_qidx + edge_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(subset)
            sample_info.append(cur_info)

        return X, Y, sample_info

    def _get_query_features_nodes(self, qrep, sbitmaps,
            jbitmaps,
            dataset_qidx,
            query_idx):
        '''
        @qrep: one pickle object.
        '''

        if self.subplan_mask is not None:
            submask = self.subplan_mask[query_idx]

        X = []
        Y = []
        sample_info = []

        # now, we will generate the actual feature vectors over all the
        # subplans. Order matters --- dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        # print(" =========== SOURCE_NODE ============= ")
        # print(SOURCE_NODE)
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        if self.sort_sub:
            node_names.sort()
        else:
            # do not sort the node_names.
            # use the original order of the node_names.
            # but be aware that, qrep["subset_graph"].nodes() returns an unordered list.
            # thus should use the order information stored in the node attribute to help 
            node_names_ord = list(range(len(node_names)))
            for n in node_names:
                ind = qrep["subset_graph"].nodes()[n]["ind"]
                node_names_ord[ind] = n
            node_names = node_names_ord
            print("\n".join( [ str(n) for n in node_names ] ) )

        for node_idx, node in enumerate(node_names):

            if self.max_num_tables != -1 \
                    and self.max_num_tables < len(node):
                continue

            if self.subplan_mask is not None and \
                    list(node) not in submask:
                continue

            x,y = self.featurizer.get_subplan_features(qrep,
                    node, bitmaps=sbitmaps,
                    join_bitmaps=jbitmaps)
            if x is None:
                continue

            if self.featurizer.featurization_type == "set" \
                and self.load_padded_mscn_feats:
                start = time.time()
                tf,pf,jf,tm,pm,jm = \
                    pad_sets([x["table"]], [x["pred"]], [x["join"]],
                            self.featurizer.max_tables,
                            self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = tf
                x["join"] = jf
                x["pred"] = pf
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm

            if self.featurizer.featurization_type == "set":
                x["flow"] = to_variable(x["flow"], requires_grad=False).float()

            X.append(x)
            Y.append(y)

            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(node)
            cur_info["template_no"] = qrep["template_name"]
            sample_info.append(cur_info)

        return X, Y, sample_info

    def _get_query_features(self, qrep, dataset_qidx,
            query_idx):
        '''
        @qrep: qrep dict.
        '''

        # yaml: sample_bitmap = 0
        if self.featurizer.sample_bitmap or \
                self.featurizer.join_bitmap:
            assert self.featurizer.bitmap_dir is not None

            bitdir = os.path.join(self.featurizer.bitmap_dir, qrep["workload"],
                    "sample_bitmap")

            bitmapfn = os.path.join(bitdir, qrep["name"])

            if not os.path.exists(bitmapfn):
                print(bitmapfn, " not found")
                sbitmaps = None
            else:
                with open(bitmapfn, "rb") as handle:
                    sbitmaps = pickle.load(handle)

        else:
            sbitmaps = None

        # old code
        # yaml: join_bitmap = 0
        if self.featurizer.join_bitmap:
            bitdir = os.path.join(self.featurizer.bitmap_dir, qrep["workload"],
                    "join_bitmap")
            bitmapfn = os.path.join(bitdir, qrep["name"])

            if not os.path.exists(bitmapfn):
                print(bitmapfn, " not found")
                # pdb.set_trace()
                jbitmaps = None
            else:
                with open(bitmapfn, "rb") as handle:
                    jbitmaps = pickle.load(handle)
        else:
            jbitmaps = None

        # self.join_key_cards = False
        if self.join_key_cards:
            return self._get_query_features_joinkeys(qrep,
                    sbitmaps,
                    jbitmaps,
                    dataset_qidx,
                    query_idx)
        else:
            return self._get_query_features_nodes(qrep,
                    sbitmaps,
                    jbitmaps,
                    dataset_qidx,
                    query_idx)

    def _get_feature_vectors_par(self, samples):

        start = time.time()
        X = []
        Y = []
        sample_info = []

        nump = 16

        batchsize = 200
        outbatch = math.ceil(len(samples) / batchsize)

        dsqidx = 0
        pool = mp.Pool(nump)
        for i in range(outbatch):
            startidx = i*batchsize
            endidx = startidx+batchsize
            endidx = min(endidx, len(samples))
            print(startidx, endidx)
            qreps = samples[startidx:endidx]

            par_args = []
            for qi, qrep in enumerate(qreps):
                par_args.append((qrep, dsqidx, startidx+qi))
                dsqidx += len(qrep["subset_graph"].nodes())

            print("par args: ", len(par_args))

            # with mp.Pool(nump) as p:
            res = pool.starmap(self._get_query_features, par_args)
            for r in res:
                X += r[0]
                Y += r[1]
                sample_info += r[2]

        pool.close()
        pdb.set_trace()

        print("Extracting features took: ", time.time() - start)
        # TODO: handle this somehow
        if self.featurizer.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        Y = to_variable(Y, requires_grad=False).float()

        return X,Y,sample_info

    def _get_ttt_feature_vectors_par(self, samples):

        if len(samples) < 100:
            return self._get_ttt_feature_vectors(samples)

        start = time.time()
        X = []
        X_neg = []
        Y = []
        sample_info = []

        nthreads = 40

        self.featurizer.total_query = 0
        self.featurizer.timeout_query = 0

        # do not read from file in parallel mode. 

        dsqidx = 0
        qi = 0
        sample_list = slices(samples, nthreads)
        p_list = []
        queue_rec = Queue()
        queue_send = Queue()

        
        print(f"dataset.py::QueryDataset::_get_ttt_feature_vectors_par::process(Main): Starging {len(sample_list)} processes, processing {len(samples)} data.")
        for qid, qreps in enumerate(sample_list):
            queue_rec.put( [ qid, self, qreps, dsqidx, qi ])
            for qrep in qreps:
                qi += 1
                for node in qrep["subset_graph"].nodes():
                    true_card = qrep["subset_graph"].nodes()[node]['cardinality']['actual']
                    if true_card >= 10 ** 8:
                        continue
                    if true_card < 1:
                        continue
                    dsqidx += 1

        for qid, qreps in enumerate(sample_list):
            p = Process(target=QueryDataset__get_ttt_feature_vectors, args=(queue_rec, queue_send))
            p_list.append(p)
            p.start()
        
        X_dict, X_neg_dict, Y_dict, info_dict = dict(), dict(), dict(), dict()
        for p in p_list:
            # empty = queue_send.empty()
            # print(f"dataset.py::QueryDataset::_get_ttt_feature_vectors_par::process(Main): Prepare to Receive from processes xxx, size = xxx. empty = {empty}")
            
            res = queue_send.get()
            pid = res[0]
            X_dict[pid], X_neg_dict[pid], Y_dict[pid], info_dict[pid] = res[1]
            print(f"dataset.py::QueryDataset::_get_ttt_feature_vectors_par::process(Main): Receive from processes {pid},type={type(pid)}, size = {len(X_dict[pid])}, qreps = {len(sample_list[pid])}.")
        for p in p_list:
            p.join()
        for qid, _ in enumerate(sample_list):
            print("processing chunk", qid, "...")
            X += pickle.loads(X_dict[qid])
            X_neg += pickle.loads(X_neg_dict[qid])
            Y += pickle.loads(Y_dict[qid])
            sample_info += pickle.loads(info_dict[qid])
        print(f"dataset.py::QueryDataset::_get_ttt_feature_vectors_par::process(Main): Receive from {len(p_list)} processes, size = {len(X)}, {len(X_neg)}, {len(Y)}, {len(sample_info)}.")
                
        print("Extracting features took: ", time.time() - start)
        # TODO: handle this somehow
        if self.featurizer.featurization_type == "combined":
            # X = to_variable(X, requires_grad=False).float()
            pass
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        # Y = to_variable(Y, requires_grad=False).float()

        # Y = torch.cat(Y, dim=0)
        Y = to_variable(Y, requires_grad=False).float().view(-1, 1)
        return X,X_neg,Y,sample_info

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()
        X = []
        Y = []
        sample_info = []

        self.featurizer.total_query = 0 
        self.featurizer.timeout_query = 0
        # to parallel load. 
        qidx = 0
        for i, qrep in enumerate(samples):
            if i % 100 == 0:
                print(f"Processing qrep {i}...")
            qhash = str(deterministic_hash(qrep["sql"]))

            if self.save_mscn_feats and \
                    "job" not in qrep["template_name"]:
                featfn = os.path.join(self.featdir, qhash) + ".pkl"
                if os.path.exists(featfn):
                    try:
                        x,y = self._load_mscn_features(featfn)
                        cur_info = self._get_sample_info(qrep, qidx, i)
                    except Exception as e:
                        print(e)
                        print("features could not be loaded in try")
                        # pdb.set_trace()
                        x,y,cur_info = self._get_query_features(qrep, qidx, i)
                        self._save_mscn_features(x,y,featfn)
                else:
                    x,y,cur_info = self._get_query_features(qrep, qidx, i)
                    self._save_mscn_features(x,y,featfn)
            else:
                x,y,cur_info = self._get_query_features(qrep, qidx, i)
            qidx += len(y)
            X += x
            Y += y
            sample_info += cur_info

        print("Extracting features took: ", time.time() - start)

        if self.featurizer.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        print(f"total queries = {self.featurizer.total_query}, timeout query = {self.featurizer.timeout_query}.")
        Y = to_variable(Y, requires_grad=False).float().view(-1, 1)
        return X,Y,sample_info
    
    # def _get_feature_vectors_switch(self, samples):
    #     # determine whether to get feature vectors. 
    #     if self.save_mscn_feats and \
    #             "job" not in samples[0]["template_name"]:
    #         qhash = str(deterministic_hash(samples[0]["sql"]))
    #         featfn = os.path.join(self.featdir, qhash) + ".pkl"
    #         if os.path.exists(featfn):
    #             try:
    #                 return self._get_feature_vectors_parallel(samples)
    #             except Exception as exp:
    #                 print(f"error when load features. {exp}")
    #                 raise exp
    #     return self._get_feature_vectors(samples)

    def _get_feature_vectors_switch(self, samples):
        # determine whether to get feature vectors. 
        if self.featurizer.parallel_gen_feats:
            return self._get_feature_vectors_par(samples)
        else:
            return self._get_feature_vectors(samples)
        

        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        if self.load_query_together:
            # assert False, "needs to be implemented"
            start_idx = self.start_idxs[index]
            end_idx = start_idx + self.idx_lens[index]

            return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                    self.info[start_idx:end_idx]
        else:
            if self.with_info:
                return self.X[index], self.Y[index], self.info[index]
            else:
                return self.X[index], self.Y[index]


class GrpQueryDataset(QueryDataset):
    def __init__(self, samples, featurizer,
            load_query_together, load_padded_mscn_feats=False,
            max_num_tables = -1,
            subplan_mask = None,
            feat_dir = "./mscn_features",
            join_key_cards=False,
            with_info = True,
            sort_sub=True):
        super(GrpQueryDataset, self).__init__(samples, featurizer, load_query_together, 
                load_padded_mscn_feats, max_num_tables, subplan_mask, feat_dir,
                join_key_cards, with_info, sort_sub)
        self.n_groups = 0
        self.group_map = dict()
        for qrep in samples:
            tpl = qrep['template_name']
            if tpl not in self.group_map.keys():
                self.group_map[tpl] = self.n_groups
                # nodes = list(qrep["subset_graph"].nodes())
                self.n_groups += 1
        self.group_counts = torch.zeros(size=(self.n_groups,), dtype=torch.float32)
        for qrep in samples:
            tpl = qrep['template_name']
            self.group_counts[ self.group_map[tpl]] += 1

    
    def group_count(self):
        return self.group_counts

    def __getitem__(self, index):
        tpl = self.info[index]['template_no']
        group_idx = self.group_map[tpl]

        if self.load_query_together:
            # assert False, "needs to be implemented"
            start_idx = self.start_idxs[index]
            end_idx = start_idx + self.idx_lens[index]
            return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                    self.info[start_idx:end_idx], group_idx
        else:
            if self.with_info:
                return self.X[index], self.Y[index], self.info[index], group_idx
            else:
                return self.X[index], self.Y[index], group_idx

class TTTQueryDataset(QueryDataset):
    def __init__(self, samples, featurizer,
            load_query_together, load_padded_mscn_feats=False,
            max_num_tables = -1,
            subplan_mask = None,
            feat_dir = "./mscn_features",
            join_key_cards=False,
            with_info = True,
            num_negs=3,
            sort_sub=True):
        # super().__init__(samples, featurizer, load_query_together, load_padded_mscn_feats, max_num_tables,
        #                  subplan_mask, feat_dir, join_key_cards, with_info)
        '''
        basically copied from super().__init__(). 
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subplans.
        @load_query_together: each sample will be a list of all the feature
        vectors belonging to all the subplans of a query.
        @subplan_mask: [], same length as samples;
        '''
        # yaml: load_query_together = 0
        self.load_query_together = load_query_together
        self.with_info = with_info
        self.num_negs = num_negs
        self.sort_sub= sort_sub
        if self.load_query_together:
            self.start_idxs, self.idx_lens = self._update_idxs(samples)

        self.load_padded_mscn_feats = load_padded_mscn_feats
        self.featurizer = featurizer
        self.max_num_tables = max_num_tables

        if "whitening" in self.featurizer.ynormalization:
            self.featurizer.update_means(samples)

        self.join_key_cards = join_key_cards

        self.save_mscn_feats = False
        self.subplan_mask = subplan_mask

        # yaml: load_padded_mscn_feats = 1
        if self.load_padded_mscn_feats:
            fkeys = list(dir(self.featurizer))
            fkeys.sort()
            attrs = ""
            for k in fkeys:
                attrvals = getattr(featurizer, k)
                if not hasattr(attrvals, "__len__") and \
                    "method" not in str(attrvals):
                    attrs += str(k) + str(attrvals) + ";"

            attrs += "padded"+str(self.load_padded_mscn_feats)
            # print(attrs)
            self.feathash = deterministic_hash(attrs)
            self.featdir = os.path.join(feat_dir, str(self.feathash))
            if os.path.exists(self.featdir):
                print("features saved before")
                if not self.featurizer.use_saved_feats:
                    print("going to delete feature directory, and save again")
                    # delete these and save again
                    self.save_mscn_feats = True
                    os.remove(self.featdir)
            else:
                print("Features not saved before.")

            if self.featurizer.use_saved_feats:
                self.save_mscn_feats = True
                make_dir(feat_dir)
                make_dir(self.featdir)

        # yaml: max_num_tables = -1
        if self.max_num_tables != -1:
            self.save_mscn_feats = False
            self.featurizer.use_saved_feats = False

        # yaml: load_query_together = 0
        if self.load_query_together:
            self.save_mscn_feats = False

        # shorthands
        self.ckey = self.featurizer.ckey
        self.minv = self.featurizer.min_val
        self.maxv = self.featurizer.max_val
        self.feattype = self.featurizer.featurization_type

        # TODO: we may want to avoid this, and convert them on the fly. Just
        # keep some indexing information around.
        # self.X, self.Y, self.info = self._get_feature_vectors(samples)
        # self.X, self.Y, self.info = self._get_feature_vectors_par(samples)
        self.X, self.X_neg, self.Y, self.info = self._get_ttt_feature_vectors_switch(samples)

        self.label = []
        for s in samples:
            node_name = list(s["subset_graph"].nodes())
            node_name.sort()
            for node in node_name:
                self.label.append( (s["subset_graph"].nodes()[node]["cardinality"]["actual"]) )


        if self.load_query_together:
            self.num_samples = len(samples)
        else:
            self.num_samples = len(self.X)

    def _get_ttt_feature_vectors_switch(self, samples):
        # pdb.set_trace()
        if self.featurizer.parallel_gen_feats:
            # currently has some problem: can not ensure a stable featurization. 
            return self._get_ttt_feature_vectors_par(samples)
        else:
            # 912: def _get_ttt_feature_vectors(self, samples)
            return self._get_ttt_feature_vectors(samples)
            

    def _load_ttt_mscn_features(self, qfeat_fn):
        # with open(qfeat_fn, "rb") as f:
        #     data = pickle.load(f)
        # return data["x"], data["y"]
        return load_ttt_mscn_features(qfeat_fn)

    def _save_ttt_mscn_features(self, x,x_neg,y,qfeat_fn):
        data = {"x":x, "x_neg": x_neg, "y":y}
        with open(qfeat_fn, "wb") as f:
            pickle.dump(data, f)

    def _get_ttt_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()
        X = []
        X_neg = []
        Y = []
        sample_info = []

        self.featurizer.total_query = 0
        self.featurizer.timeout_query = 0

        # to parallel load. 
        qidx = 0
        for i, qrep in enumerate(samples):
            if i % 100 == 0:
                print(f"Processing qrep {i}...")
            qhash = str(deterministic_hash(qrep["sql"]))

            if self.save_mscn_feats and \
                    "job" not in qrep["template_name"]:
                featfn = os.path.join(self.featdir, qhash) + ".pkl"
                if os.path.exists(featfn):
                    try:
                        x,x_neg,y = self._load_ttt_mscn_features(featfn)
                        cur_info = self._get_sample_info(qrep, qidx, i)
                    except Exception as e:
                        print(e)
                        print("features could not be loaded in try")
                        # pdb.set_trace()
                        x,x_neg,y,cur_info = self._get_ttt_query_features(qrep, qidx, i)
                        self._save_ttt_mscn_features(x,x_neg,y,featfn)
                else:
                    x,x_neg,y,cur_info = self._get_ttt_query_features(qrep, qidx, i)
                    self._save_ttt_mscn_features(x,x_neg,y,featfn)
            else:
                x,x_neg,y,cur_info = self._get_ttt_query_features(qrep, qidx, i)
            qidx += len(y)
            X += x
            X_neg += x_neg
            Y += y
            sample_info += cur_info

        print("Extracting features took: ", time.time() - start)

        if self.featurizer.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
            X_neg = to_variable(X_neg, requires_grad=False).float() 
        elif self.featurizer.featurization_type == "set":
            # don't need to do anything, since padding+masks is handled later
            pass

        Y = to_variable(Y, requires_grad=False).float().view(-1, 1)
        print(f"Totally {self.featurizer.total_query} queries, {self.featurizer.timeout_query} time out queries.")
        return X,X_neg,Y,sample_info
    
    # def _get_feature_vectors_switch(self, samples):
    #     # determine whether to get feature vectors. 
    #     if self.save_mscn_feats and \
    #             "job" not in samples[0]["template_name"]:
    #         qhash = str(deterministic_hash(samples[0]["sql"]))
    #         featfn = os.path.join(self.featdir, qhash) + ".pkl"
    #         if os.path.exists(featfn):
    #             try:
    #                 return self._get_feature_vectors_parallel(samples)
    #             except Exception as exp:
    #                 print(f"error when load features. {exp}")
    #                 raise exp
    #     return self._get_feature_vectors(samples)

    def _get_ttt_query_features(self, qrep, dataset_qidx, query_idx):
        '''
        @qrep: qrep dict.
        '''

        # yaml: sample_bitmap = 0
        if self.featurizer.sample_bitmap or \
                self.featurizer.join_bitmap:
            assert self.featurizer.bitmap_dir is not None

            bitdir = os.path.join(self.featurizer.bitmap_dir, qrep["workload"],
                    "sample_bitmap")

            bitmapfn = os.path.join(bitdir, qrep["name"])

            if not os.path.exists(bitmapfn):
                print(bitmapfn, " not found")
                sbitmaps = None
            else:
                with open(bitmapfn, "rb") as handle:
                    sbitmaps = pickle.load(handle)

        else:
            sbitmaps = None

        # old code
        # yaml: join_bitmap = 0
        if self.featurizer.join_bitmap:
            bitdir = os.path.join(self.featurizer.bitmap_dir, qrep["workload"],
                    "join_bitmap")
            bitmapfn = os.path.join(bitdir, qrep["name"])

            if not os.path.exists(bitmapfn):
                print(bitmapfn, " not found")
                # pdb.set_trace()
                jbitmaps = None
            else:
                with open(bitmapfn, "rb") as handle:
                    jbitmaps = pickle.load(handle)
        else:
            jbitmaps = None

        # self.join_key_cards = False
        if self.join_key_cards:
            return self._get_query_features_joinkeys(qrep, sbitmaps, jbitmaps, dataset_qidx, query_idx)
        else:
            return self._get_ttt_query_features_nodes(qrep, sbitmaps, jbitmaps, dataset_qidx, query_idx)

    def _get_ttt_query_features_nodes(self, qrep, sbitmaps, jbitmaps, dataset_qidx, query_idx):
        '''
        @qrep: one pickle object.
        '''

        if self.subplan_mask is not None:
            submask = self.subplan_mask[query_idx]

        X = []
        X_negs = []
        Y = []
        sample_info = []

        # now, we will generate the actual feature vectors over all the
        # subplans. Order matters --- dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        # print(" =========== SOURCE_NODE ============= ")
        # print(SOURCE_NODE)
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        if self.sort_sub:
            node_names.sort()
        else:
            # do not sort the node_names.
            # use the original order of the node_names.
            # but be aware that, qrep["subset_graph"].nodes() returns an unordered list.
            # thus should use the order information stored in the node attribute to help 
            node_names_ord = list(range(len(node_names)))
            for n in node_names:
                ind = qrep["subset_graph"].nodes()[n]["ind"]
                node_names_ord[ind] = n
            node_names = node_names_ord

        if DEBUG_MODE:
            sql = qrep["sql"]
            print(f"sql = {sql}")
        for node_idx, node in enumerate(node_names):

            if self.max_num_tables != -1 \
                    and self.max_num_tables < len(node):
                continue

            if self.subplan_mask is not None and \
                    list(node) not in submask:
                continue

            x,x_neg,y = self.featurizer.get_ttt_subplan_features(qrep, node, bitmaps=sbitmaps, join_bitmaps=jbitmaps, num_negs=self.num_negs)
            if x is None:
                continue
            if DEBUG_MODE and len(node) > 1: 
                table = x["table"]
                pred = x["pred"]
                join = x["join"]
                flow = x["flow"]
                neg_table = x_neg["table"]
                neg_pred = x_neg["pred"]
                neg_join = x_neg["join"]
                neg_flow = x_neg["flow"]
                print(f"node = {node}")
                print(f"node = {node}, table = {table}, pred = {pred}, join = {join}, flow = {flow}")
                print(f"neg_table = {neg_table}, neg_pred = {neg_pred}, neg_join = {neg_join}, neg_flow = {neg_flow}")
            # x_neg["flow"] = [ 0 for _ in range(len(x_neg["table"])) ]
            x_neg["tmask"] = [ 0 for _ in range(len(x_neg["table"])) ]
            x_neg["pmask"] = [ 0 for _ in range(len(x_neg["table"])) ]
            x_neg["jmask"] = [ 0 for _ in range(len(x_neg["table"])) ]

            # pdb.set_trace()
            if self.featurizer.featurization_type == "set" \
                and self.load_padded_mscn_feats:
                start = time.time()
                tf,pf,jf,tm,pm,jm = \
                    pad_sets([x["table"]], [x["pred"]], [x["join"]],
                            self.featurizer.max_tables,
                            self.featurizer.max_preds,
                            self.featurizer.max_joins)
                x["table"] = tf
                x["join"] = jf
                x["pred"] = pf
                # relevant masks
                x["tmask"] = tm
                x["pmask"] = pm
                x["jmask"] = jm
                
                # process x_neg
                assert len(x_neg["table"]) == len(x_neg["pred"]) == len(x_neg["join"])
                for ti in range(len(x_neg["table"])):
                    t = x_neg["table"][ti]
                    p = x_neg["pred"][ti]
                    j = x_neg["join"][ti]
                    tf, pf, jf, tm, pm, jm = \
                        pad_sets( [ t ], [ p ], [ j ], 
                                    self.featurizer.max_tables,
                                    self.featurizer.max_preds,
                                    self.featurizer.max_joins)
                    x_neg["table"][ti] = tf
                    x_neg["pred"][ti] = pf
                    x_neg["join"][ti] = jf
                    x_neg["tmask"][ti] = tm
                    x_neg["pmask"][ti] = pm
                    x_neg["jmask"][ti] = jm

            # import pdb; pdb.set_trace()
            if self.featurizer.featurization_type == "set":
                x["flow"] = to_variable(x["flow"], requires_grad=False).float()
                for ti in range(len(x_neg["flow"])):
                    x_neg["flow"][ti] = to_variable(x_neg["flow"][ti], requires_grad=False).float()

            # transform x_neg
            for k in x_neg.keys():
                x_neg[k] = torch.stack(x_neg[k], dim=0)
                if len(x_neg[k].shape) == 3:
                    x_neg[k] = x_neg[k].permute(1, 0, 2)
                elif len(x_neg[k].shape) == 2:
                    # do nothing. 
                    # x_neg[k] = x_neg[k].permute(1, 0)
                    # import pdb; pdb.set_trace()
                    pass

            X.append(x)
            X_negs.append(x_neg)
            Y.append(y)

            cur_info = {}
            cur_info["num_tables"] = len(node)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx
            cur_info["node"] = str(node)
            cur_info["template_no"] = qrep["template_name"]
            sample_info.append(cur_info)
        if DEBUG_MODE:
            pass
            # import pdb; pdb.set_trace()
        return X, X_negs, Y, sample_info

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        if self.load_query_together:
            # assert False, "needs to be implemented"
            start_idx = self.start_idxs[index]
            end_idx = start_idx + self.idx_lens[index]

            return self.X[start_idx:end_idx],  self.Y[start_idx:end_idx], self.X_neg[start_idx:end_idx], \
                    self.info[start_idx:end_idx]
        else:
            if self.with_info:
                return self.X[index], self.Y[index], self.X_neg[index], self.info[index]
            else:
                return self.X[index], self.Y[index], self.X_neg[index]
