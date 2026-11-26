"""Skip `gpu` tests without CUDA and `data` tests without the real inputs.

Unmarked tests are synthetic and run on every machine. The laptop and the remote PC run the same
suite; each skips what it can't run.
"""

import pytest

from clwe_chains.config import load_config


def pytest_collection_modifyitems(config, items):
    gpu = [i for i in items if "gpu" in i.keywords]
    data = [i for i in items if "data" in i.keywords]
    if gpu:
        import torch  # only imported when a gpu test was collected; the import takes seconds

        if not torch.cuda.is_available():
            for item in gpu:
                item.add_marker(pytest.mark.skip(reason="needs CUDA"))
    if data:
        root = load_config().data_root
        if not root.exists():
            for item in data:
                item.add_marker(pytest.mark.skip(reason=f"data_root {root} not found"))
