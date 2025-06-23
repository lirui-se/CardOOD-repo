import os
from db.table import Table, QueryInfo, JoinInfo
from db.schema import DBSchema

class QueryParser(object):
    def __init__(self, table: Table):
        self.df = table.df
        self.col_types = table.col_types
        print(self.df.columns)

    def parse_predicates(self, pred_str: str):
        pred_list = list()
        if not pred_str:
            return pred_list
        predicates = pred_str.split("#")
        for predicate in predicates:
            col_name = predicate.split(",")[0].strip()
            # print("=======================================")
            # print(self.df.columns)
            col_idx = self.df.columns.get_loc(col_name)
            if self.col_types[col_idx] == 'categorical': # categorical type
                cat_set = [int(_.strip()) for _ in predicate.split(",")[1:]]
                pred_list.append((col_name, cat_set))
            else:  # numerical type
                upper, lower = float(predicate.split(",")[1].strip()), float(predicate.split(",")[2].strip())
                pred_list.append((col_name, upper, lower))
        return pred_list

    def parse_line(self, line: str):
        pred_str, card = line.split("@")[0].strip(), int(line.split("@")[1].strip())
        pred_list = self.parse_predicates(pred_str)
        return pred_list, card

    def load_queries(self, query_path: str):
        sub_dirs = os.listdir(query_path)
        all_queries, all_cards = list(), list()
        all_query_infos = list()
        for i, sub_dir in enumerate(sorted(sub_dirs)):
            with open(os.path.join(query_path, sub_dir), "r") as in_file:
                print(os.path.join(query_path, sub_dir))
                for line in in_file:
                    # print(line)
                    pred_list, card = self.parse_line(line)
                    all_queries.append(pred_list)
                    all_cards.append(card)
                    all_query_infos.append(
                        QueryInfo(num_table=1, num_joins=0, num_predicates=len(pred_list), is_equal_join=False,
                                  is_multi_key=False, template_no=i, distri=i, table_comb='forest'))
                in_file.close()
        return all_queries, all_cards, all_query_infos

class JoinQueryParser(object):
    def __init__(self, schema: DBSchema):
        self.schema = schema
        self.tables = schema.tables
        self.table_name_to_tid = schema.table_name_to_tid
        self.table_parsers = [QueryParser(table) for table in self.tables]

    def parse_line_return_name(self, line: str):
        terms = line.strip().split('@')
        table_str, join_str, card = terms[0].strip(), terms[-2].strip(), int(terms[-1].strip())
        table_names = table_str.split(',')
        if not isinstance(table_names, list):
            table_names = [ table_names ]
        table_ids = [self.table_name_to_tid[table_name] for table_name in table_names]
        assert len(table_ids) + 3 == len(terms), "Query Format Error!"
        all_pred_str = terms[1: len(table_ids) + 1]
        all_pred_list, join_infos = list(), list()
        for t_id, pred_str in zip(table_ids, all_pred_str):
            pred_list = self.table_parsers[t_id].parse_predicates(pred_str.strip())
            all_pred_list.append(pred_list)
        join_str = [] if not join_str else join_str.split('#')
        # print(join_str)
        for join in join_str:
            t1_name, t2_name, col_name = join.split(',')[0].strip(), join.split(',')[1].strip(), join.split(',')[
                2].strip()
            t_id = self.table_name_to_tid[t1_name]
            col_idx = self.tables[t_id].df.columns.get_loc(col_name)
            col_type = self.tables[t_id].col_types[col_idx]
            join_info = JoinInfo(t1_name= t1_name, t1_id=self.table_name_to_tid[t1_name],
                                 t2_name= t2_name, t2_id=self.table_name_to_tid[t2_name],
                                 col_name=col_name, col_type=col_type)
            join_infos.append(join_info)
        return [ t for t in sorted(table_names) ], table_ids, all_pred_list, join_infos, card



    def parse_line(self, line: str):
        terms = line.strip().split('@')
        table_str, join_str, card = terms[0].strip(), terms[-2].strip(), int(terms[-1].strip())
        table_names = table_str.split(',')
        table_ids = [self.table_name_to_tid[table_name] for table_name in table_names]
        assert len(table_ids) + 3 == len(terms), "Query Format Error!"
        all_pred_str = terms[1: len(table_ids) + 1]
        all_pred_list, join_infos = list(), list()
        for t_id, pred_str in zip(table_ids, all_pred_str):
            pred_list = self.table_parsers[t_id].parse_predicates(pred_str.strip())
            all_pred_list.append(pred_list)
        join_str = [] if not join_str else join_str.split('#')
        # print(join_str)
        for join in join_str:
            t1_name, t2_name, col_name = join.split(',')[0].strip(), join.split(',')[1].strip(), join.split(',')[
                2].strip()
            t_id = self.table_name_to_tid[t1_name]
            col_idx = self.tables[t_id].df.columns.get_loc(col_name)
            col_type = self.tables[t_id].col_types[col_idx]
            join_info = JoinInfo(t1_name= t1_name, t1_id=self.table_name_to_tid[t1_name],
                                 t2_name= t2_name, t2_id=self.table_name_to_tid[t2_name],
                                 col_name=col_name, col_type=col_type)
            join_infos.append(join_info)
        return table_ids, all_pred_list, join_infos, card

    def load_queries(self, query_path: str):
        sub_dirs = os.listdir(query_path)
        all_queries, all_cards = list(), list()
        all_query_infos = list()
        for i, sub_dir in enumerate(sorted(sub_dirs)):
            with open(os.path.join(query_path, sub_dir), "r") as in_file:
                for line in in_file:
                    table_names, table_ids, all_pred_list, join_infos, card = self.parse_line_return_name(line)
                    all_queries.append((table_ids, all_pred_list, join_infos))
                    all_cards.append(card)
                    table_pairs = set([(join_info.t1_id, join_info.t2_id) for join_info in join_infos])
                    is_multi_key = True if len(table_pairs) < len(join_infos) else False
                    num_predicates = sum([len(pred_list) for pred_list in all_pred_list])
                    all_query_infos.append(QueryInfo(num_table=len(table_ids), num_joins=len(join_infos),
                                                     num_predicates=num_predicates, is_equal_join=True,
                                                     is_multi_key=is_multi_key, template_no=i, distri=i, table_comb=",".join(table_names)))
                in_file.close()
        return all_queries, all_cards, all_query_infos

    def load_queries_by_query_pattern(self, query_path:str):
        # The difference with load_queries is that, all_queries[70]
        sub_dirs = os.listdir(query_path)
        all_queries_str = list()
        all_queries, all_cards = list(), list()
        all_query_infos = list()
        # pattern_indices = [ ]
        # table_numbers = [ ]
        # cnt = 0
        for i, sub_dir in enumerate(sorted(sub_dirs)):
            with open(os.path.join(query_path, sub_dir), "r") as in_file:
                #ind = [ cnt, -1 ]
                # t_num = -1
                for line in in_file:
                    all_queries_str.append(line.strip("\n"))
                    table_names, table_ids, all_pred_list, join_infos, card = self.parse_line_return_name(line)
                    # print(table_ids, all_pred_list, join_infos, card)
                    all_queries.append((table_ids, all_pred_list, join_infos))
                    all_cards.append(card)
                    table_pairs = set([(join_info.t1_id, join_info.t2_id) for join_info in join_infos])
                    is_multi_key = True if len(table_pairs) < len(join_infos) else False
                    num_predicates = sum([len(pred_list) for pred_list in all_pred_list])
                    all_query_infos.append(QueryInfo(num_table=len(table_ids), num_joins=len(join_infos),
                                                     num_predicates=num_predicates, is_equal_join=True,
                                                     is_multi_key=is_multi_key, template_no=i, distri=i, table_comb=",".join(table_names)))
                    # t_num = len(table_ids)
                    # cnt += 1
                # in_file.close()
                # ind[1] = cnt
                # pattern_indices.append((t_num, ind))
                # Note that table_number can not be used if 'distri' is specified as split_keys, 
                # since the queries in a same distribution group may not have the same table number. 
                # table_numbers.append(t_num)
        # print(pattern_indices)
        # return all_queries, all_cards, all_query_infos, pattern_indices
        return all_queries_str, all_queries, all_cards, all_query_infos


    # def load_queries_by_query_pattern_sql(self, query_path:str):
    #     assert os.path.exists(query_path), query_path
    #     sub_dirs = os.listdir(query_path)
    #     all_sql_queries = list()
    #     # pattern_indices = [ ]
    #     # table_numbers = [ ]
    #     # cnt = 0
    #     for sub_dir in sorted(sub_dirs):
    #         with open(os.path.join(query_path, sub_dir), "r") as in_file:
    #             #ind = [ cnt, -1 ]
    #             # t_num = -1
    #             for line in in_file:
    #                 all_sql_queries.append(line.strip("\n"))
    #     return all_sql_queries
