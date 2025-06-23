import os
import pandas as pd
from db.table import Table, JoinInfo
import networkx as nx
import db.datasets as datasets

class DBSchema(object):
	# fields:
	# 1. primary_key_list
	# 2. pk_code_lists [ { v1 -> 0, v2 -> 1, v3 -> 2, ... }, {  } ]
	# 3. fk_code_dicts_list [ { column1 -> { }, column2 -> {}, ... } , {  }   ]
	# 4. tables
	# 5. schema_name
	# 6. tid_to_table_name
	# 7. table_name_to_tid
	# 8. all_join_infos
	# 9. tid_to_join_infos
	# 10. table_pair_to_join_infos
	# 11. all_join_table_pairs
	# 12. join_graph
	# 13. all_join_triples
	# 14. all_join_col_names

	def __init__(self, df_dataset_list, col_types_list, table_name_list, primary_key_list):
		self.primary_key_list = primary_key_list
		self.pk_code_lists = list()

		# map primary key column to categorical encoding
		#############
		# Try not construct code_dict and map the primary key column into categorical encoding.
		############# 
		for df, col_types in zip(df_dataset_list, col_types_list):
			self.pk_code_lists.append({})

		# for df, col_types, primary_key in zip(df_dataset_list, col_types_list, primary_key_list):
		# 	if not primary_key:
		# 		self.pk_code_lists.append({})
		# 		continue
		# 	cate = pd.Categorical(df[primary_key])
		# 	code_dict = dict([(category, code) for code, category in enumerate(cate.categories)])
		# 	self.pk_code_lists.append(code_dict)
		# 	df[primary_key] = cate.codes

		# prepare the fk categorical code for each table
		# fk and pk name must be the same. 
		self.fk_code_dicts_list = list()
		#############
		# Try not construct fk_code_dicts
		#############
		for t2_id, df in enumerate(df_dataset_list):
			self.fk_code_dicts_list.append({})

		# for t2_id, df in enumerate(df_dataset_list):
		# 	fk_code_dicts = {}
		# 	for t1_id, primary_key in enumerate(primary_key_list):
		# 		if t2_id == t1_id:
		# 			continue
		# 		if primary_key in df.columns:
		# 			pk_code_dict = self.pk_code_lists[t1_id]
		# 			fk_code_dicts[primary_key] = pk_code_dict
		# 	self.fk_code_dicts_list.append(fk_code_dicts)
		self.tables = list()
		for df, col_types, table_name, primary_key, fk_code_dicts in \
				zip(df_dataset_list, col_types_list, table_name_list, primary_key_list, self.fk_code_dicts_list):
			table = Table(df, col_types, table_name, primary_key=primary_key, fk_code_dicts=fk_code_dicts)
			self.tables.append(table)

		# create meta data
		self.schema_name = "_".join([table.table_name for table in self.tables])
		self.tid_to_table_name, self.table_name_to_tid = dict(), dict()
		for tid, table in enumerate(self.tables):
			self.tid_to_table_name[tid] = table.table_name
			self.table_name_to_tid[table.table_name] = tid
		for k in self.table_name_to_tid.keys():
			print(k, self.table_name_to_tid[k])
		### Try to remove all the join infos. (seems not used in anywhere.)
		self.all_join_infos = list()
		self.tid_to_join_infos = dict()
		self.table_pair_to_join_infos = dict()

		# captures the (PK-FK) join information in the schema
		for t1_id in range(len(self.tables)):
			for t2_id in range(t1_id + 1, len(self.tables)):
				table1, table2 = self.tables[t1_id], self.tables[t2_id]
				for col_name in table1.df.columns:
					if col_name not in table2.df.columns: continue
					join_info = JoinInfo(t1_name=table1.table_name, t1_id=t1_id,
										 t2_name=table2.table_name, t2_id=t2_id, col_name=col_name,
										 col_type=table1.col_types[table1.df.columns.get_loc(col_name)])
					self.all_join_infos.append(join_info)
					if t1_id not in self.tid_to_join_infos.keys():
						self.tid_to_join_infos[t1_id] = list()
					self.tid_to_join_infos[t1_id].append(join_info)
					if t2_id not in self.tid_to_join_infos.keys():
						self.tid_to_join_infos[t2_id] = list()
					self.tid_to_join_infos[t2_id].append(join_info)
					if (t1_id, t2_id) not in self.table_pair_to_join_infos.keys():
						self.table_pair_to_join_infos[(t1_id, t2_id)] = list()
					self.table_pair_to_join_infos[(t1_id, t2_id)].append(join_info)

		self.all_join_table_pairs = list(self.table_pair_to_join_infos.keys())
		self.join_graph = nx.Graph()
		self.join_graph.add_edges_from(self.all_join_table_pairs)
		self.all_join_triples = [(join_info.t1_id, join_info.t2_id, join_info.col_name) for join_info in
								 self.all_join_infos]
		self.all_join_col_names = [join_info.col_name for join_info in self.all_join_infos]


	def print_schema_info(self):
		print("<" * 80)
		for t_id, table in enumerate(self.tables):
			print("Table {}: {}".format(t_id, table.table_name))
			print("Columns", table.df.columns)
			print("PK name: {}".format(table.primary_key))
		#print("FK name: {}".format(','.join(table)))
		# print("Join infos:", self.all_join_infos)
		print(">" * 80)


def load_schema(schema_name: str, data_path: str):
	if schema_name == 'imdb_simple':
		load_funcs = [datasets.LoadIMDB_title(data_path), datasets.LoadIMDB_cast_info(data_path),
					  datasets.LoadIMDB_movie_info(data_path),
					  datasets.LoadIMDB_movie_companies(data_path), datasets.LoadIMDB_movie_info_idx2(data_path),
					  datasets.LoadIMDB_movie_keyword(data_path)]
		table_name_list = ['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_info_idx', 'movie_keyword']
	elif schema_name == 'dsb':
		# load_funcs = [ datasets.
		pass
	else:
		raise NotImplementedError('Do not support Schema {}!'.format(schema_name))
	df_dataset_list, col_types_list, pk_name_list = list(), list(), list()
	local_vars = locals()
	for load_func in load_funcs:
		local_vars['df_dataset'], local_vars['col_types'], local_vars['pk'] = load_func
		df_dataset_list.append(local_vars['df_dataset'])
		col_types_list.append(local_vars['col_types'])
		pk_name_list.append(local_vars['pk'])

	schema = DBSchema(df_dataset_list, col_types_list, table_name_list, pk_name_list)
	schema.print_schema_info()
	return schema

def load_table(table_name: str, data_path:str):
	if table_name == 'forest':
		df_dataset, col_types = datasets.load_forest(data_path=data_path)
	elif table_name == 'higgs':
		df_dataset, col_types = datasets.load_higgs(data_path=data_path)
	else:
		raise NotImplementedError('Do not support table {}!'.format(table_name))
	table = Table(df=df_dataset, col_types=col_types, table_name=table_name)
	return table


if __name__ == "__main__":
	from db.parser import JoinQueryParser
	from encoder.transform import JoinQueryDataset, TTTJoinQueryDataset
	from torch.utils.data import DataLoader

	data_path = '/home/kfzhao/data/rdb/imdb_clean'
	query_path = '/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_824_FP'

	schema = load_schema(schema_name='imdb_simple', data_path= data_path)
	parser = JoinQueryParser(schema)
	all_queries, all_cards, all_query_infos = parser.load_queries(query_path= query_path)
	print(len(all_queries))
	train_dataset = TTTJoinQueryDataset(schema=schema, queries=all_queries, cards=all_cards, encoder_type='dnn')
	#train_dataset = JoinQueryDataset(schema=schema, queries=all_queries, cards=all_cards, encoder_type='dnn')
	train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	for x, y, x_neg in train_loader:
		print(x.shape, y.shape, x_neg.shape)
