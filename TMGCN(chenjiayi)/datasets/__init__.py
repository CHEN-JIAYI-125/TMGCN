# 只选「用哪一个数据集文件」。
# Teacher 是否加载：config 里 reply_distillation = 0 或 1

#from .dataset_graph_4 import TargetGraphDataset, target_graph_collate_fn
from .dataset_graph_4_rd import TargetGraphDatasetRD as TargetGraphDataset
from .dataset_graph_4_rd import target_graph_collate_fn
# from .dataset_graph_3 import TargetGraphDataset, target_graph_collate_fn
