import pytest
import torch

from onemb.hash_utils import HashFamily


@pytest.mark.parametrize("seed", [42, 451, 1984])
def test_reproducibility(seed: int, hash_family: HashFamily, inputs: torch.Tensor) -> None:
    torch.manual_seed(seed)
    hash_func = hash_family.draw()
    hashed_inputs_1 = hash_func(inputs)

    torch.manual_seed(seed)
    hash_func = hash_family.draw()
    hashed_inputs_2 = hash_func(inputs)

    assert torch.all(hashed_inputs_1 == hashed_inputs_2)
