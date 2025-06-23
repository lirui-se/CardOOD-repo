import os
import pdb
import sys
from networkx.readwrite import json_graph
from query_representation.utils import *
from query_representation.query import *
from query_representation.constants import *
import multiprocessing
import time
import clickhouse_driver as chd
import duckdb

import threading

class ClickThread(threading.Thread):
    def __init__(self, func, args=()):
        super(ClickThread, self).__init__()
        self.func = func
        self.args = args
        self.default_res = [[ 10 ** 8 ]] 
        self.res = None

    def run(self):
        self.res = self.func(*self.args)

    def get_result(self):
        try:
            if self.res is not None:
                return self.res
            else:
                return self.default_res
        except Exception:
            return self.default_res

if len(sys.argv) < 5:
    print("Usage: python3 sql_to_qrep.py <machine> <chunk_num> <start_chunk_id> <end_chunk_id>")
    print("\tmachine: 1006, 1066, sefe01, 1031")
    print("\tchunk_num: number of groups for one machine. default = 10")
    print("\tstart_chunk_id & end_chunk_id: value from 0 ~ chunk_num - 1. (will execute start_chunk_id & end_chunk_id)")
    sys.exit(1)

machine = sys.argv[1]
chunk_num = int(sys.argv[2])
start_chunk_id = int(sys.argv[3])
end_chunk_id = int(sys.argv[4])

assert machine in set(["1006", "1066", "sefe01", "1031"]), f"Not a valid machine: {machine}"
assert start_chunk_id >= 0 and start_chunk_id <= chunk_num - 1, f"Invalid start_chunk_id: {start_chunk_id}. Should in [0, {chunk_num} - 1]"
assert end_chunk_id >= 0 and end_chunk_id <= chunk_num - 1, f"Invalid end_chunk_id: {end_chunk_id}. Should in [0, {chunk_num} - 1]"

if machine == 'sefe01':
    home="/home/lirui"
elif machine == '1006':
    home="/home/lirui"
elif machine == '1066':
    home="/home/kfzhao"
elif machine == '1031':
    home="/home/shfang"
path=f"{home}/codes/PG_CardOOD/CardDATA/dsb/query_templates_pg"
datapath=f"{home}/codes/PG_CardOOD/CardDATA/dsb/code/data"


qid=['013', '018', '019', '025', '027', '040', '050', '072', '084', '085', '091', '099', '100', '101', '102']
# qid=['018']

# or operator: 013, 085

qid=[ '018', '019', '025', '027', '040', '050', '072', '084', '091', '099', '100', '101', '102']
# qid=[ '11c' ]

# Example:
# OUTPUT_DIR="./queries/joblight/all_joblight/"
# INPUT_FN = "./queries/joblight.sql"
# OUTPUT_FN_TMP = "joblight-{i}.sql"


DEBUG_MODE=False
GEN_QREP=False

def load_empty_qdata(fns):
    qreps = []
    for qfn in fns:
        qrep = load_qrep(qfn)
        qrep["fname"] = os.path.basename(qfn)
        qreps.append(qrep)
    return qreps

def load_empty_qdata_interface(pid, fns, queue):
    qreps = load_empty_qdata(fns)
    queue.put([ pid, qreps ])
    # print(f"utils.py::load_empty_qdata_interface::process({pid}): size = {len(preds)}, load finished.")

def load_empty_qdata_parallel(fns):
    nthreads = 40
    if len(fns) < 10000:
        return load_empty_qdata(fns)
    if DEBUG_MODE:
        print(f"Totally {len(fns)} files to load.")
    fns_list = slices(fns, nthreads) 
    p_list = []
    queue = Queue()
    for i, f in enumerate(fns_list):
        p = Process(target=load_empty_qdata_interface, args=(i, f, queue))
        p_list.append(p)
        p.start()
    # CAUTION: MUST CONSUME THE QUEUE FIRST IN THE MAIN PROCESS!
    #   The child process may wait for queue data being consumed. 
    #   If not consume queue data here, the child process will never exit. 
    res = []
    for p in p_list:
        pid, preds = queue.get()
        res += preds
        print(f"utils.py::load_empty_qdata_parallel::process(Main): Receive from pid = {pid}, size = {len(preds)}.")
    for pid, p in enumerate(p_list):
        p.join()
        if DEBUG_MODE:
            # This message must in order. 
            print(f"Process {pid} successfully finished.")
    print(f"All {len(p_list)} processes finished.")
    return res

# for qrep in all_qreps:
#     fname = qrep["fname"]
#     arr = fname.split("-")

def slice_by_group_size(l, group_size):
    length = len(l)
    indx = 0
    res = []
    while indx < length:
        res.append( l[indx:indx + group_size])
        indx = indx + group_size
    return res

def load_duckdb(db_name):
    tables = { "dsb": ["call_center", "promotion", "catalog_page", "reason", "catalog_returns", "ship_mode", "catalog_sales", "store", "customer_address", "store_returns", "customer", "store_sales", "customer_demographics", "time_dim", "date_dim", "warehouse", "dbgen_version", "web_page", "household_demographics", "web_returns", "income_band", "web_sales", "inventory", "web_site", "item"] }
    conn = duckdb.connect(database = ":memory:")
    with open(datapath + "/create_tables.sql") as file:
        schema_sql = file.read()
    try:
        result = conn.execute(schema_sql).fetchall()
        for t in tables[db_name]:
            load_sql = f"insert into {t} select * from read_csv('{t}.csv');"
            result = conn.execute(load_sql).fetchall()
    except Exception as exp:
        print(f"Fail to load dataset {db_name}.")
    return conn

def get_card_interface(pid, ichunk, queue, dir_path):
    ipc, qfns = queue.get()
    print(f"get_card_interface(pid {ipc}): ichunk = {ichunk}, {len(qfns)} ready to load & exe.")
    qreps = load_empty_qdata(qfns)
    # if machine == '1066':
    #     conn = load_duckdb()
    # if machine == '1031':
    #     conn = load_duckdb()
    for q in qreps:
        basename = q["fname"]
        tnum = len(q["join_graph"].nodes())
        for node in q['subset_graph'].nodes():
            if len(node) > 6 and len(node) < tnum:
                card = 10 ** 8
                exe_time = 10 ** 8
            else:
                # if 'inventory' in node:
                #     cnt_sql = subplan_to_agg_sql(q, node)
                # else:
                #     cnt_sql = subplan_to_sql(q, node)
                cnt_sql = subplan_to_sql(q, node)
                start_time = time.time()
                if machine == '1006':
                    card = execute(cnt_sql, 'dsb', basename)[0][0]
                elif machine == 'sefe01':
                    # card = execute(cnt_sql, 'dsb', basename)[0][0]
                    card = execute_clickhouse(cnt_sql, 'dsb', basename)[0][0]
                elif machine == '1066':
                    card = execute(cnt_sql, 'dsb', basename)[0][0]
                    # card = execute_duckdb(conn, cnt_sql, 'dsb', basename)[0][0]
                elif machine == '1031':
                    card = execute(cnt_sql, 'dsb', basename)[0][0]
                    # card = execute_duckdb(conn, cnt_sql, 'dsb', basename)[0][0]
                else:
                    print("Not implement yet!")
                    raise NotImplementedError(f"Execute on {machine} not implemented.")
                end_time = time.time()
                exe_time = end_time - start_time
            q['subset_graph'].nodes[node]['cardinality'] = {}
            q['subset_graph'].nodes[node]['cardinality']['actual'] = card 
            q['subset_graph'].nodes[node]['cardinality']['expected'] = card 
            q['subset_graph'].nodes[node]['cardinality']['total'] = card 
            q['subset_graph'].nodes[node]['exec_time'] = {'actual': exe_time}
        fname = q["fname"].split("-")[0]
        path = f"{dir_path}/{fname}/{basename}"
        save_qrep(path, q)
    print(f"ichunk = {ichunk}, ipc = {ipc}, {len(qfns)} queries finished.")

def get_card_parallel(machine, all_qfns, chunk_num, start_chunk_id, end_chunk_id, dir_path):
    if machine == '1006':
        nthreads = 40
    elif machine == 'sefe01':
        # clickhouse requires more memory. 
        nthreads = 25
    elif machine == '1066':
        nthreads = 10
    elif machine == '1031':
        nthreads = 10
    group_num = chunk_num
    group_size = int(len(all_qfns) / group_num)    
    template_num = 13
    # 52000 queries in total. group by 2600 queries. totally 20 groups. 
    all_qfns_list = slice_by_group_size(all_qfns, group_size)
    assert len(all_qfns_list) == group_num and len(all_qfns_list[0]) == group_size, f"Error for all_qreps_list = {len(all_qfns_list)} and all_qreps_list[0] = {len(all_qfns_list[0])}"
    # for duckdb, prepare the dataschema. 
    for ichunk, chunk in enumerate(all_qfns_list):
        print(f"==== get_card_parallel(main): chunk {ichunk} / {len(all_qfns_list)} started ====")
        print(f"==== get_card_parallel(main): {template_num} templates, each with {group_size / template_num} queries ====")
        if ichunk < start_chunk_id or ichunk > end_chunk_id:
            print(f"==== get_card_parallel(main): chunk {ichunk} skipped. ==== ")
            print(f"==== get_card_parallel(main): {template_num} templates, each with {group_size / template_num} queries ====")
            continue
        # inside each group of 2600 queries, each of 50 threads running 65 queries. 
        parallel_chunk = slice_by_group_size(chunk, int(group_size / nthreads))
        queue = multiprocessing.Queue()
        start = time.time()
        process_list = []
        for ipc, pchunk in enumerate(parallel_chunk):
            p = multiprocessing.Process(target=get_card_interface, args=(ipc, ichunk, queue, dir_path))
            process_list.append(p)
            p.start()
        for ipc, pchunk in enumerate(parallel_chunk):
            queue.put((ipc, pchunk))
        for ipc, p in enumerate(process_list):
            p.join()
        end = time.time()
        print(f"==== get_card_parallel(main): chunk {ichunk} / {len(all_qfns_list)} finished, ({end - start}) seconds ====")
        print(f"==== get_card_parallel(main): {template_num} templates, each with {group_size / template_num} queries ====")

def pg_connection_string(db_name):
    return f"dbname={db_name} user=lirui host=localhost port=5433"

def execute(sql, db_name, basename):
    try:
        con = pg.connect(pg_connection_string(db_name))
    except BaseException as exp:
        print(f"Fail to connect: {basename}, {sql}, {exp}")
        return [[10 ** 8]]
        raise exp
    cursor = con.cursor()
    try:
        cursor.execute(sql)
    except BaseException as exp:
        print(f"Fail to execute: {basename}, {sql}, {exp}")
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()
        print("returning arbitrary large value for now")
        return [[10 ** 8]]
    
    try:
        exp_output = cursor.fetchall()
    except BaseException as exp:
        print(f"Fail to fetch result.{exp}")
        exp_output = None
    cursor.close()
    con.close()
    return exp_output


def execute_duckdb(conn, sql, db_name, basename):
    raise NotImplementedError("execute_duckdb is not implemented well. It does not support timeout as well as canceling a query.")
    try:
        result = conn.execute(sql).fetchall()[0][0]
    except Exception as exp:
        print(f"Fail to execute: {basename}, {sql}, {exp}")
        print("returning arbitrary large value for now.")
        return [[10 ** 8]]
    return result

def execute_click_thread(client, sql):
    try:
        res = client.execute(sql)
    except Exception as exp:
        print(f"Fail to execute: {sql}, {exp}")
        print("returning arbitrary large value for now.")
        return [[10 ** 8]]
    return res

def execute_clickhouse(sql, db_name, basename):
    try:
        client = chd.Client(host='127.0.0.1', port='2233', database=db_name, connect_timeout=70)
    except Exception as exp:
        print(f"Fail to connect: {basename}, {sql}, {exp}")
        return [[10 ** 8]]

    query_thread = ClickThread(execute_click_thread, args=(client, sql))
    query_thread.start()
    cnt = 0
    # time out = 30s . 
    while cnt < 30:
        if not query_thread.is_alive():
            break
        time.sleep(1)
        # print(f"waiting...{cnt}")
        cnt += 1
    client.disconnect()
    res = query_thread.get_result()

    return res



def translate(OUTPUT_DIR, INPUT_FN, SQL_NAME):
    make_dir(OUTPUT_DIR)

    with open(INPUT_FN, "r") as f:
        data = f.read()

    print(OUTPUT_DIR)
    print(INPUT_FN)
    print(SQL_NAME)
    queries = data.split(";")
    for i, sql in enumerate(queries):
        output_fn = OUTPUT_DIR + "/" + SQL_NAME + "-" + str(i+1) + ".pkl"
        if "SELECT" not in sql and "select" not in sql:
            continue
        qrep = func_sql_to_qrep(sql, compute_ground_truth = True)
        # Convert join_graph and subset_graph into sql count query.

        # Convert sql count query into aggregation query. 

        save_qrep(output_fn, qrep)

        if DEBUG_MODE:
            print("==== qrep ====")
            print(f"qrep.keys() = {qrep.keys()} ")
            joingraph = qrep['join_graph']
            all_nodes = []
            for node in joingraph.nodes:
                all_nodes.append(node)
                print(f"join graph node = {node}, nodes[node] = {joingraph.nodes[node]}")
            for edge in joingraph.edges:
                print(f"join graph edge = {edge}, edges[edge] = {joingraph.edges[edge]}")
            print("==== sql ====")
            all_sql = subplan_to_sql(qrep, all_nodes)
            agg_sql = subplan_to_agg_sql(qrep, all_nodes)
            all_card = execute(all_sql, "dsb")[0][0]
            agg_card = execute(agg_sql, "dsb")[0][0]
            print(f"all_sql = {all_sql}, all_card = {all_card}")
            print(f"agg_sql = {agg_sql}, agg_card = {agg_card}")
            subgraph = qrep['subset_graph']
            nodes = subgraph.nodes
            print(f"subgraph.nodes.type = {type(nodes)}")
            for node in nodes:
                pass
                # print(f"node = {node}, nodes[node] = {nodes[node]}, type(nodes[node]) = {type(nodes[node])}")
                # print(f"sql = {subplan_to_sql(qrep, node)}")
            sys.exit(0)



if DEBUG_MODE:
    id = '025'
    translate(f"{path}/sql/all_dsb", f"{path}/sql/tpl{id}.sql", f"tpl{id}")
    sys.exit(0)


# 1. Convert SQL queries to qrep data structure. 
if GEN_QREP:
    p_list = []
    for id in qid:
        p = multiprocessing.Process(target=translate, args=(f"{path}/sql/all_dsb/tpl{id}", f"{path}/sql/tpl{id}.sql", f"tpl{id}"))
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()
    print("Generate qrep done.")

# 2. Collect statistics for qrep data structure.
all_qfns = []
qfn_lists = []
for id in qid:
    qdir = f"{path}/sql/all_dsb/tpl{id}"
    qfns = list(glob.glob(qdir + "/*.pkl"))
    k_qfns = [ (int(os.path.basename(qfn).split("-")[1].split(".")[0]), qfn) for qfn in qfns ]
    k_qfns.sort(key=lambda x : x[0])
    qfns = [ x[1] for x in k_qfns ]
    qfn_lists.append(qfns)

# Assume that every template has the same number of queries.
qfn_len = len(qfn_lists[0])

for ind in range(qfn_len):
    for qfn in qfn_lists:
        all_qfns.append(qfn[ind])

print(f"len(qnfs) = {qfn_len}, len(all_qfns) = {len(all_qfns)}, e.g. {all_qfns[0]}")

# all_qreps = load_empty_qdata_parallel(all_qfns)
# 
# # Total subqueries. 
# n_sub = 0
# n_queries = 0
# for qrep in all_qreps:
#     n_queries += 1
#     ssg = qrep["subset_graph"]
#     for node in ssg.nodes():
#         n_sub += 1
# import pdb; pdb.set_trace()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# prepare to running on different machines. 
# partition by 4 chunks,
# chunk 1: 1006
# chunk 2: 1066
# chunk 3: sefe01
# chunk 4: 1031
qfns_chunk_size = int(len(all_qfns) / 4)
chunk_1006, chunk_1066, chunk_sefe01, chunk_1031 = list(chunks(all_qfns, qfns_chunk_size))
chunk_dict = { "1006": chunk_1006, "1066": chunk_1066, "sefe01": chunk_sefe01, "1031": chunk_1031 }
for k in chunk_dict.keys():
    assert len(chunk_dict[k]) == qfns_chunk_size


get_card_parallel(machine, chunk_dict[machine], chunk_num, start_chunk_id, end_chunk_id, f"{path}/sql/all_dsb_card")
