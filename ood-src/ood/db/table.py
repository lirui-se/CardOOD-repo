import collections
import random
import pandas as pd
import numpy as np
import math
from typing import List

Address = collections.namedtuple('Address', ['start', 'end'])
QueryInfo = collections.namedtuple('QueryInfo', ['num_table', 'num_joins', 'num_predicates', 'is_equal_join', 'is_multi_key', 'template_no', 'table_comb', 'distri'])
JoinInfo = collections.namedtuple('JoinInfo', ['t1_name', 't1_id', 't2_name', 't2_id', 'col_name', 'col_type'])

class Table(object):
	def __init__(self, df, col_types, table_name:str, primary_key: str = '', fk_code_dicts=None):
		self.df = df
		self.table_name = table_name
		self.col_types = col_types

		self.num_cols = len(df.columns)
		self.num_rows = len(df.index)
		self.all_col_ranges = np.zeros(shape=(self.num_cols, 2))
		self.all_col_denominator = np.zeros(shape=(self.num_cols,))
		self.df.fillna(-1, inplace=True)
		# self.all_col_df = []
		self.categorical_codes_dict = dict()
		self.primary_key = primary_key

		for i in range(self.num_cols):
			col_name = self.df.columns[i]
			single_col_df = self.df.iloc[:, i]
			single_col_df = single_col_df.sort_values()
			# self.all_col_df.append(single_col_df)
			if col_types[i] == 'categorical':
				# categorical type
				cate = pd.Categorical(single_col_df)
				#print(len(single_col_df.unique()))
				# col_name is a fk, map to code
				if fk_code_dicts is not None and col_name in fk_code_dicts.keys():
					self.categorical_codes_dict[col_name] = fk_code_dicts[col_name]
				else:
					self.categorical_codes_dict[col_name] = \
						dict([(category, code) for code, category in enumerate(cate.categories)]) # {category : code}
			else: # numerical type
				self.all_col_ranges[i][0] = single_col_df.min()
				self.all_col_ranges[i][1] = single_col_df.max()
				denominator = self.all_col_ranges[i][1] - self.all_col_ranges[i][0]
				self.all_col_denominator[i] = denominator if denominator > 0 else 1e-6

	def subquery_sample(self, pred_list: List):
		# Given q, sample a query q' where C(q') <= C(q) always holds, for test time training
		num_sample_pred = random.randint(1, len(pred_list))
		sample_pred_idx = random.sample(list(range(len(pred_list))), k=num_sample_pred)
		new_pred_list = list()
		for idx, pred in enumerate(pred_list):
			if idx not in sample_pred_idx:
				new_pred_list.append(pred)
				continue
			col_name = pred[0]
			col_idx = self.df.columns.get_loc(col_name)
			if self.col_types[col_idx] == 'categorical':
				cat_set = pred[1]
				if len(cat_set) == 1:  # directly insert the predicate
					new_pred_list.append(pred)
				new_cat_set = random.sample(cat_set, k=random.randint(1, len(cat_set) - 1))
				new_pred_list.append((col_name, new_cat_set))
			else:
				upper, lower = pred[1], pred[2]
				mid = random.uniform(upper, lower)
				if upper == mid == lower:  # directly insert the predicate
					new_pred_list.append(pred)
				elif mid > lower:
					new_pred_list.append((col_name, mid, lower))
				else:
					new_pred_list.append((col_name, upper, mid))
		return new_pred_list
