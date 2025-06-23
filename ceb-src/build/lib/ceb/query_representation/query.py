import networkx as nx
from networkx.readwrite import json_graph

import query_representation
from query_representation.utils import *
import time
import itertools
import json
import pdb
import pickle
import copy

import query_representation.utils

DEBUG_MODE=False

def get_subset_cache_name(sql):
    return str(deterministic_hash(sql)[0:5])

def parse_sql(sql, user, db_name, db_host, port, pwd, timeout=False,
        compute_ground_truth=True, subset_cache_dir="./subset_cache/"):
    '''
    @sql: sql query string.

    @ret: python dict with the keys:
        sql: original sql string
        join_graph: networkX graph representing query and its
        join_edges. Properties include:
            Nodes:
                - table
                - alias
                - predicate matches
            Edges:
                - join_condition

            Note: This is the only place where these strings will be stored.
            Each of the subplans will be represented by their nodes within
            the join_graph, and we can use these properties to reconstruct the
            appropriate query for each subplan.

        subset_graph: networkX graph representing each subplan as a node.

        Properties of each subplan will include all the cardinality data that
        will need to be computed:
            - true_count
            - pg_count
            - total_count
    '''
    start = time.time()
    join_graph = query_representation.utils.extract_join_graph(sql)
    subset_graph = query_representation.utils.generate_subset_graph(join_graph)

    # print("query has",
    #       len(join_graph.nodes), "relations,",
    #       len(join_graph.edges), "joins, and",
    #       len(subset_graph), " possible subplans.",
    #       "took:", time.time() - start)

    ret = {}
    ret["sql"] = sql
    ret["join_graph"] = join_graph
    ret["subset_graph"] = subset_graph

    ret["join_graph"] = nx.adjacency_data(ret["join_graph"])
    ret["subset_graph"] = nx.adjacency_data(ret["subset_graph"])
    # get true cardinality. 
    
    return ret

def load_qrep(fn):
    assert ".pkl" in fn, f"{fn}"
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"]["adjacency"] = []
    query["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def save_qrep(fn, cur_qrep):
    assert ".pkl" in fn
    qrep = copy.deepcopy(cur_qrep)
    qrep["join_graph"] = nx.adjacency_data(qrep["join_graph"])
    qrep["subset_graph"] = nx.adjacency_data(qrep["subset_graph"])

    with open(fn, "wb") as f:
        pickle.dump(qrep, f)

def get_tables(qrep):
    '''
    ret:
        @tables: list of table names in the query
        @aliases: list of corresponding aliases in the query.
        (each table has an alias here.)
    '''
    tables = []
    aliases = []
    for node in qrep["join_graph"].nodes(data=True):
        aliases.append(node[0])
        tables.append(node[1]["real_name"])

    return tables, aliases

def get_predicates(qrep):
    '''
    ret:
        @predicates: list of the predicate strings in the query
        We also break the each predicate string into @pred_cols, @pred_types,
        and @pred_vals and return those as separate lists.
    '''
    predicates = []
    pred_cols = []
    pred_types = []
    pred_vals = []
    for node in qrep["join_graph"].nodes(data=True):
        info = node[1]
        if len(info["predicates"]) == 0:
            continue
        predicates.append(info["predicates"])
        pred_cols.append(info["pred_cols"])
        pred_types.append(info["pred_types"])
        pred_vals.append(info["pred_vals"])

    return predicates, pred_cols, pred_types, pred_vals

def get_joins(qrep):
    '''
    '''
    joins = []
    for einfo in qrep["join_graph"].edges(data=True):
        join = einfo[2]["join_condition"]
        joins.append(join)
    return joins

def get_postgres_cardinalities(qrep):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    pred_dict = {}
    for alias_key in qrep["subset_graph"].nodes():
        info = qrep["subset_graph"].nodes()[alias_key]
        est = info["cardinality"]["expected"]
        pred_dict[(alias_key)] = est

    return pred_dict

def get_true_cardinalities(qrep):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    pred_dict = {}
    for alias_key in qrep["subset_graph"].nodes():
        info = qrep["subset_graph"].nodes()[alias_key]
        true_card = info["cardinality"]["actual"]
        pred_dict[(alias_key)] = true_card

    return pred_dict

def subplan_to_sql(qrep, subplan_node):
    '''
    @ests: dict; key: label of the subplan. value: cardinality estimate.
    '''
    sg = qrep["join_graph"].subgraph(subplan_node)
    subsql = query_representation.utils.nx_graph_to_query(sg)
    return subsql

def merge_eqm(eqc, lind, rind):
    eqc_list = []
    for i, eqm in enumerate(eqc):
        if i != lind and i != rind:
            eqc_list.append(eqm)
    eqc_list.append( eqc[lind] + eqc[rind] )
    return eqc_list

def get_eqid(eqc, col):
    eqid = -1
    for i, eqm in enumerate(eqc):
        if col in eqm:
            assert eqid != -1, f"Col {col} occurs in eqc[{eqid}] = {eqc[eqid]} and eqc[{i}] = {eqc[i]}"
            eqid = i

def get_eqname(eqid):
    return "C" + str(eqid)

def get_eqidfromname(eqname):
    return int(eqname[1:])

def get_joineqname(eqtables, l_tables, r_tables):
    join_cols = []
    for i, tables in enumerate(eqtables):
        # whether intersected with left tables
        if bool(tables & l_tables) and bool(tables & r_tables):
            # a join column found 
            join_cols.append(get_eqname(i))
    return join_cols

def get_truecol_from_joineq(cols, tables, eqname):
    for t in tables:
        for c_eqname, c in cols[t]:
            if c_eqname == eqname:
                return c
    return None


def subplan_to_agg_sql(qrep, subplan_node):
    sg = qrep["join_graph"].subgraph(subplan_node)
    if DEBUG_MODE:
        print("==== query.py::cnt_sql_to_agg_sql ====")
        for node in sg.nodes:
            # type(sg.edges) = EdgeView, type(sg.edges(node)) = EdgeDataView
            edges = sg.edges(node)
            cond = [ sg.edges[edge]["join_condition"] for edge in edges ]
            # print(f"type(sg.edges) = {type(sg.edges)}, type(sg.edges(node)) = {type(edges)}")
                # conds = [ edge["join_condition"] for edge in edges ]
            # print(f"node = {node}, type(node) = {type(node)}, type(str(node)) = {type(str(node))}, edges = {edges}, cond = {cond}")
            print(f"node = {node}, node_prop = {sg.nodes[node]}, edges = {edges}, cond = {cond}")
    # 1. Construct the equivalence class
    join_conditions = []
    eqc = []
    eqtables = []
    cols = {}


    for edge in sg.edges:
        jc = sg.edges[edge]["join_condition"]
        for join in jc.split(","):
            join_conditions.append(join)
    if DEBUG_MODE:
        print(f"all the join conditions = {join_conditions}")
    
    for join in join_conditions:
        c1, c2 = join.split("=")[0].strip(), join.split("=")[1].strip()
        assert "." in c1 and "." in c2, f"Column {c1} or {c2} not in #table.#column format!"
        t1 = c1.split(".")[0].strip()
        t2 = c2.split(".")[0].strip()
        lind, rind = -1, -1
        for i,eqm in enumerate(eqc):
            if c1 in eqm:
                lind = i
            if c2 in eqm: 
                rind = i
        if lind > 0 and rind > 0:
            if lind != rind:
                eqc = merge_eqm(eqc, lind, rind)
        if lind > 0 and rind < 0:
            eqc[lind].append(c2)
        if lind < 0 and rind > 0:
            eqc[rind].append(c1)
        if lind < 0 and rind < 0:
            eqc.append( [c1, c2] )
    for eqid, eqm in enumerate(eqc):
        for c in eqm:
            t = c.split(".")[0].strip()
            if t not in cols.keys():
                cols[t] = set()
            for eqname, cname in cols[t]:
                assert eqname != get_eqname(eqid),  f"Table {t} contains two cols {c} and {cname} in the same equivalence class!"
            cols[t].add((get_eqname(eqid), c))
    for eqid, eqm in enumerate(eqc):
        tables = set()
        for c in eqm:
            t = c.split(".")[0].strip()
            tables.add(t)
        eqtables.append(tables)

    curr_sql = None
    tid = len(subplan_node)
    curr_tables = set()
    output_eqname = []
    set_subplan_node = set(subplan_node)
    for nodei, node in enumerate(sg.nodes):
        snode = set([node])
        other_nodes = set_subplan_node.difference(snode)
        nodei_with_rest_join_cols = get_joineqname(eqtables, snode, other_nodes)
        with_rest_join_cols = nodei_with_rest_join_cols
        true_cols = [ get_truecol_from_joineq(cols, snode, eqname) for eqname in nodei_with_rest_join_cols ]
        select_cols = [" count(*) as c " ] + [ e1 + " as " + e2 for e1, e2 in zip(true_cols, nodei_with_rest_join_cols) ]
        select_clause = ",".join(select_cols)
        from_clause = sg.nodes[node]["real_name"] + " as " + node
        where_clause = " and ".join(sg.nodes[node]["predicates"])
        if len(where_clause) > 0:
            where_clause = " where " + where_clause
        group_clause = ",".join(true_cols)
        if len(group_clause) > 0:
            group_clause = " group by " + group_clause
        agg_sql = f"(select {select_clause} from {from_clause} {where_clause} {group_clause}) as T{nodei}"
        if curr_sql is None:
            left_table_id = 0
            curr_sql = agg_sql
        else:
            # 1. join T{left_table_id} with T{nodei}
            with_nodei_join_cols = get_joineqname(eqtables, curr_tables, snode)
            select_cols = [ f"T{left_table_id}.c * T{nodei}.c as c" ]
            left_output_cols = output_eqname[nodei - 1]
            right_output_cols = nodei_with_rest_join_cols
            temp_output_cols = set(left_output_cols) | set(right_output_cols)
            for toc in temp_output_cols:
                if get_truecol_from_joineq(cols, curr_tables, toc) is not None:
                    select_cols.append(f"T{left_table_id}.{toc}")
                else:
                    select_cols.append(f"T{nodei}.{toc}")
            select_clause = ",".join(select_cols)
            from_clause = f"{curr_sql}, {agg_sql}"
            join_clause = " and ".join([ f"T{left_table_id}.{col} = T{nodei}.{col}" for col in with_nodei_join_cols ])
            if len(join_clause) > 0:
                where_clause = " where " + join_clause
            else:
                where_clause = join_clause
            join_sql = f"( select {select_clause} from {from_clause} {where_clause}) as TEMP"

            # 2. group by T{left_table_id} \Join T{nodei} on with_rest_join_cols
            with_rest_join_cols = get_joineqname(eqtables, curr_tables | snode, set_subplan_node.difference(curr_tables))
            select_cols = [ f"sum(TEMP.c) as c"] + [ f"TEMP.{col}" for col in with_rest_join_cols  ]
            select_clause = ",".join(select_cols)
            from_clause = join_sql
            group_clause = ",".join([ f"TEMP.{col}" for col in with_rest_join_cols  ])
            if len(group_clause) > 0:
                group_clause = " group by " + group_clause 
            left_table_id = tid
            tid += 1
            curr_sql = f"(select {select_clause} from {from_clause} {group_clause}) as T{left_table_id}"
        output_eqname.append(with_rest_join_cols)
        curr_tables.add(node)
    return f"select sum(c) from {curr_sql}"
