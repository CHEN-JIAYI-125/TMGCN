"""Exercise projection, grounding, and semantic-structure fusion."""
from types import SimpleNamespace
import torch
from models.model_graph_5_2 import HeterogeneousStanceTracerV5


def main():
    cfg = SimpleNamespace(
        input_size=16, hidden_size=8, num_layers=1,
        use_grounding=True, hypergraph_attention=False, cl_lambda=0.0,
    )
    model = HeterogeneousStanceTracerV5(cfg)
    batch = dict(
        x=torch.randn(3, 16), y=torch.tensor([0, 1, 2]),
        target_knowledge=torch.randn(1, 4, 16), graph_ptr=torch.tensor([0, 3]),
    )
    output = model(**batch)
    assert output['logits'].shape == (3, 3)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    assert model.input_proj[0].weight.grad is not None
    fused = HeterogeneousStanceTracerV5(
        SimpleNamespace(
            input_size=16, hidden_size=8, num_layers=1,
            use_grounding=True, hypergraph_attention=False,
            use_sem_struct_fusion=True, cl_lambda=0.0,
        )
    )
    out = fused(**batch)
    assert out['logits'].shape == (3, 3)
    assert fused.classifier[0].in_features == 32  # 4 * hidden
    out['loss'].backward()
    baseline = HeterogeneousStanceTracerV5(SimpleNamespace(hidden_size=8))
    assert isinstance(baseline.input_proj, torch.nn.Identity)
    print('Qwen input/grounding projection and backward smoke passed')


if __name__ == '__main__':
    main()
