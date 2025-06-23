import os
import pandas as pd

UCI_DATA_PATH = '/home/kfzhao/data/UCI'

### UCI Tables
def load_higgs(data_path: str, filename: str="HIGGS.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	col_types = ['numerical'] * len(col_names)
	df_dataset = pd.read_csv(csv_file, header=None, usecols=[22, 23, 24, 25, 26, 27, 28],
							 names=col_names, nrows=nrows)
	return df_dataset, col_types

def load_forest(data_path: str, filename: str="forest.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	col_types = ['numerical'] * len(col_names)
	df_dataset = pd.read_csv(csv_file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
							 names=col_names, nrows=nrows)
	return df_dataset, col_types


### TPC-DS Tables
TPCDS_CLEAN_DATA_DIR = "/home/kfzhao/data/rdb/TPCDS_clean"

def LoadTPCDS_store_sales(data_path, filename="store_sales.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['item_sk', 'customer_sk', 'store_sk', 'promo_sk', 'quantity', 'wholesale_cost', 'list_price', 'sales_price',
				 'ext_discount_amt', 'ext_sales_price', 'ext_wholesale_cost', 'ext_list_price', 'ext_tax', 'ext_coupon_amt',
				 'net_paid', 'net_paid_inc_tax', 'net_profit']
	col_types = ['numerical'] * 17
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[2, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadTPCDS_store(data_path, filename = "store.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['store_sk', 'number_employees', 'floor_space', 'market_id', 'devision_id', 'company_id', 'tax_percentage']
	col_types = ['numerical'] * 7
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 6, 7, 10, 14, 18, 28], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'store_sk'
	return df_dataset, col_types, primary_key


def LoadTPCDS_item(data_path, filename = "item.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['item_sk', 'current_price', 'wholesale_cost', 'brand_id', 'class_id', 'category_id', 'manufact_id']
	col_types = ['numerical'] * 7
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 5, 6, 7, 9, 11, 13], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'item_sk'
	return df_dataset, col_types, primary_key

def LoadTPCDS_customer(data_path, filename = "customer.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['customer_sk', 'birth_day', 'birth_month', 'birth_year']
	col_types = ['numerical'] * 4
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 11, 12, 13], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'customer_sk'
	return df_dataset, col_types, primary_key

def LoadTPCDS_promotion(data_path, filename= "promotion.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['promo_sk', 'item_sk', 'cost', 'response_target']
	col_types = ['numerical'] * 6
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 4, 5, 6], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'promo_sk'
	return df_dataset, col_types, primary_key

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def LoadDSB_table(data_path, table_name):
    tables = [
        "customer_address",
        "customer_demographics",
        "date_dim",
        "warehouse",
        "ship_mode",
        "time_dim",
        "reason",
        "income_band",
        "item",
        "store",
        "call_center",
        "customer",
        "web_site",
        "store_returns",
        "household_demographics",
        "web_page",
        "promotion",
        "catalog_page",
        "inventory",
        "catalog_returns",
        "web_returns",
        "web_sales",
        "catalog_sales",
        "store_sales"
    ]
    table_cols = read_json_file(data_path + "/tables.json")
    primary_cols = read_json_file(data_path + "/pks.json")
    csv_file = os.path.join(data_path, table_name + ".csv")
    col_names = table_cols[table_name]
    col_types = [ 'numerical' ] * len(col_names)
    df_dataset = pd.read_csv(csv_file, header=1, delimiter=',', names=tables[table_name], nrows=None)
    primary_key = ''



### IMDB Tables
def LoadIMDB_title(data_path, filename="title.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'kind_id', 'product_year', 'imdb_id']
	col_types = ['numerical'] * 4
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 3, 4, 5], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_cast_info(data_path, filename="cast_info.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['person_id', 'movie_id', 'person_role_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2, 3], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info(data_path, filename="movie_info.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_id', 'movie_id', 'info_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_companies(data_path, filename="movie_companies.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'company_id', 'company_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2, 3], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info_idx(data_path, filename="movie_info_idx.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_idx_id','movie_id', 'info_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_idx_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info_idx2(data_path, filename="movie_info_idx.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_idx_id','movie_id']
	col_types = ['numerical'] * 2
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_idx_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_keyword(data_path, filename="movie_keyword.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'keyword_id']
	col_types = ['numerical'] * 2
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key
