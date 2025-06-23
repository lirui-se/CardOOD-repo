import clickhouse_driver as chd
import threading
import time

class ClickThread(threading.Thread):
    def __init__(self, func, args=()):
        super(ClickThread, self).__init__()
        self.func = func
        self.args = args
        self.default_res = [[ 10 ** 8 ]] 

    def run(self):
        self.res = self.func(*self.args)

    def get_result(self):
        try:
            return self.res
        except Exception:
            return self.default_res


def execute_click_thread(client, sql):
    print("Executint sql...")
    try:
        res = client.execute(sql)
        print("Still waiting ...")
    except Exception as exp:
        print(f"Fail to execute: {sql}, {exp}")
        print("returning arbitrary large value for now.")
        return [[10 ** 8]]
    return res

def execute_click(sql, db_name):
    try:
        client = chd.Client(host='localhost', port='2233', database=db_name, connect_timeout=70)
    except Exception as exp:
        print(f"Fail to connect: {sql}, {exp}")
        return [[10 ** 8]]
    query_thread = ClickThread(execute_click_thread, args=(client, sql))
    query_thread.start()
    cnt = 0
    while cnt < 10:
        if not query_thread.is_alive():
            break
        time.sleep(1)
        print(f"waiting...{cnt}")
        cnt += 1
    client.disconnect()
    res = query_thread.get_result()[0][0]
    print(res)
    return res

sql="SELECT COUNT(*) FROM date_dim AS d1 , item AS item , store_returns AS store_returns , store_sales AS store_sales WHERE  d1.d_moy = 8 AND  d1.d_year = 1999 AND d1.d_date_sk = store_sales.ss_sold_date_sk AND item.i_item_sk = store_sales.ss_item_sk AND store_sales.ss_customer_sk = store_returns.sr_customer_sk AND store_sales.ss_item_sk = store_returns.sr_item_sk AND store_sales.ss_ticket_number = store_returns.sr_ticket_number"
sql="select count(*) from catalog_sales;"
execute_click(sql, "dsb")
