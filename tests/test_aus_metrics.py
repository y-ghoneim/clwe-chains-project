import numpy as np
import pytest
import torch

from clwe_chains.align import fit
from clwe_chains.aus_metrics import (
    aus_scheme_map,
    frobenius_error,
    hausdorff,
    neighbourhood_overlap,
    precision_at_k,
)

torch.manual_seed(0)


def test_aus_scheme_map_equals_direct():
    """AUS Algorithm 1: each step fitted from the mapped source; the product is direct lstsq."""
    xs = [torch.randn(300, 20, dtype=torch.float64) for _ in range(4)]  # en, x1, x2, ar
    assert torch.allclose(aus_scheme_map(xs), fit(xs[0], xs[-1], "lstsq"), atol=1e-8)


def test_aus_scheme_map_single_step_is_lstsq():
    x, y = torch.randn(100, 10, dtype=torch.float64), torch.randn(100, 10, dtype=torch.float64)
    assert torch.allclose(aus_scheme_map([x, y]), torch.linalg.pinv(x) @ y)


def test_frobenius_error():
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert frobenius_error(u, v) == pytest.approx(2.0)


def test_hausdorff_symmetric_max():
    u = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    v = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    # u -> v: max(min(0,3), min(1, sqrt(10))) = 1; v -> u: max(0, 3) = 3.
    assert hausdorff(u, v) == pytest.approx(3.0)


def test_neighbourhood_overlap_identical_and_bounds():
    u = torch.randn(50, 8, dtype=torch.float64)
    assert neighbourhood_overlap(u, u.clone(), k=5) == pytest.approx(1.0)
    v = torch.randn(50, 8, dtype=torch.float64)
    assert 0.0 <= neighbourhood_overlap(u, v, k=5) < 1.0


def test_precision_at_k_cosine_and_euclidean():
    v = torch.eye(4, dtype=torch.float64)
    u = v.clone()
    u[3] = torch.tensor([1.0, 0, 0, 0.9], dtype=torch.float64)  # row 3 nearest to row 0
    for metric in ("cosine", "euclidean"):
        p = precision_at_k(u, v, ks=(1, 2), metric=metric)
        assert p[1] == pytest.approx(0.75) and p[2] == pytest.approx(1.0), metric
    assert np.isclose(precision_at_k(v, v, ks=(1,), metric="cosine")[1], 1.0)
