# 只选「用哪一个模型文件」。不要在这里关超图。
# 超图 / 时序 / 回复：config 里 use_speaker / use_dialogue / use_temporal / use_reply = 0 或 1

# from .model_graph_5_2 import HeterogeneousStanceTracerV5
from .model_graph_5_2_rd import HeterogeneousStanceTracerV5RD as HeterogeneousStanceTracerV5
# from .model_graph_5_1 import HeterogeneousStanceTracerV5
