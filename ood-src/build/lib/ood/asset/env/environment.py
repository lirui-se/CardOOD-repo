from typing import List
from util import PredictionStatistics
from encoder.transform import QueryDataset, JoinQueryDataset, TTTJoinQueryDataset
from db.table import Table
from db.schema import DBSchema

# Trainer that uses environment: IRM, CORAL
class QueryEnvironment(object):
    # The class to build
    def __init__(self, table: Table, queries: List, cards: List, query_infos: List, part_keys: str= 'num_predicates', encoder_type: str = 'dnn'):
        pred_stat = PredictionStatistics()
        partition_query_indices = pred_stat.get_partitioned_indices(query_infos, part_keys=part_keys) 
        # {key: list(index)}
        self.datasets = list()
        for query_indices in partition_query_indices.values():
            partition_queries = [queries[idx] for idx in query_indices]
            partition_cards = [cards[idx] for idx in query_indices]
            partition_query_infos = [query_infos[idx] for idx in query_indices]
            self.datasets.append(QueryDataset(table, partition_queries, partition_cards, partition_query_infos, encoder_type))


class JoinQueryEnvironment(object):
    def __init__(self, schema: DBSchema, queries: List, cards: List, query_infos: List, part_keys: str = 'num_table', encoder_type: str = 'dnn'):
        pred_stat = PredictionStatistics()
        partition_query_indices = pred_stat.get_partitioned_indices(query_infos,
                                                                    part_keys=part_keys)  # {key: list(index)}
        self.datasets = list()
        for query_indices in partition_query_indices.values():
            partition_queries = [queries[idx] for idx in query_indices]
            partition_cards = [cards[idx] for idx in query_indices]
            partition_query_infos = [query_infos[idx] for idx in query_indices]
            self.datasets.append(
                JoinQueryDataset(schema, partition_queries, partition_cards, partition_query_infos, encoder_type))

           

class TTTJoinQueryEnvironment(object):
    def __init__(self, schema: DBSchema, queries: List, cards: List, query_infos: List, num_negs:int = 5, part_keys: str = 'num_table', encoder_type: str = 'dnn'):
        pred_stat = PredictionStatistics()
        partition_query_indices = pred_stat.get_partitioned_indices(query_infos,
                                                                    part_keys=part_keys)  # {key: list(index)}
        self.datasets = list()
        for query_indices in partition_query_indices.values():
            partition_queries = [queries[idx] for idx in query_indices]
            partition_cards = [cards[idx] for idx in query_indices]
            partition_query_infos = [query_infos[idx] for idx in query_indices]
            self.datasets.append(
                TTTJoinQueryDataset(schema, partition_queries, partition_cards, partition_query_infos, num_negs, encoder_type))
