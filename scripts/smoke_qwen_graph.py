"""Exercise projection and target grounding in base and distilled forward paths."""
from types import SimpleNamespace
import torch
from models.model_graph_5_2 import HeterogeneousStanceTracerV5
from models.model_graph_5_2_rd import HeterogeneousStanceTracerV5RD


def main():
    for distilled in (False, True):
        cfg = SimpleNamespace(input_size=16, hidden_size=8, num_layers=1,
                              use_grounding=True, hypergraph_attention=False,
                              reply_distillation=distilled, cl_lambda=0.0)
        model = HeterogeneousStanceTracerV5RD(cfg) if distilled else HeterogeneousStanceTracerV5(cfg)
        batch = dict(x=torch.randn(3, 16), y=torch.tensor([0, 1, 2]),
                     target_knowledge=torch.randn(1, 4, 16), graph_ptr=torch.tensor([0, 3]))
        output = model(**batch)
        assert output['logits'].shape == (3, 3)
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        assert model.input_proj[0].weight.grad is not None
    fused = HeterogeneousStanceTracerV5(
        SimpleNamespace(input_size=16, hidden_size=8, num_layers=1,
                        use_grounding=True, hypergraph_attention=False,
                        use_sem_struct_fusion=True, cl_lambda=0.0)
    )
    batch = dict(x=torch.randn(3, 16), y=torch.tensor([0, 1, 2]),
                 target_knowledge=torch.randn(1, 4, 16), graph_ptr=torch.tensor([0, 3]))
    out = fused(**batch)
    assert out['logits'].shape == (3, 3)
    assert fused.classifier[0].in_features == 32  # 4 * hidden
    out['loss'].backward()
    skip_only = HeterogeneousStanceTracerV5(
        SimpleNamespace(input_size=16, hidden_size=8, num_layers=1,
                        use_grounding=True, hypergraph_attention=False,
                        use_sem_struct_fusion=True, use_sem_residual=False,
                        use_qwen_skip=True, cl_lambda=0.0)
    )
    assert skip_only.classifier[0].in_features == 32
    assert not hasattr(skip_only, 'sem_res')
    res_only = HeterogeneousStanceTracerV5(
        SimpleNamespace(input_size=16, hidden_size=8, num_layers=1,
                        use_grounding=True, hypergraph_attention=False,
                        use_sem_struct_fusion=True, use_sem_residual=True,
                        use_qwen_skip=False, cl_lambda=0.0)
    )
    assert res_only.classifier[0].in_features == 24  # 3 * hidden
    ungated = HeterogeneousStanceTracerV5(
        SimpleNamespace(input_size=16, hidden_size=8, num_layers=1,
                        use_grounding=True, hypergraph_attention=False,
                        use_sem_struct_fusion=True, use_gated_fusion=False,
                        cl_lambda=0.0)
    )
    ungated_out = ungated(**batch)
    assert ungated_out['logits'].shape == (3, 3)
    ungated_out['loss'].backward()
    baseline = HeterogeneousStanceTracerV5(SimpleNamespace(hidden_size=8))
    assert isinstance(baseline.input_proj, torch.nn.Identity)
    print('Qwen input/grounding projection and backward smoke passed')


if __name__ == '__main__':
    main()
