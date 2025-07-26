import pytest
import torch

from onemb.embedding import HashEmbedding, UnifiedEmbedding
from onemb.hash_utils import HashFamily


def test_param_padding_index(
    num_embeddings: int,
    embedding_dim: int,
    hash_family: HashFamily,
    inputs: torch.Tensor,
) -> None:
    hash_embedding = HashEmbedding(num_embeddings, embedding_dim, padding_index=0)

    hash_funcs = [hash_family.draw() for _ in range(4)]
    inputs_embeds = hash_embedding(inputs, hash_funcs)

    assert torch.allclose(inputs_embeds[inputs == 0], torch.tensor(0.0))


@pytest.mark.parametrize("num_lookups", [1, 4])
def test_hash_embedding(
    num_embeddings: int,
    embedding_dim: int,
    hash_family: HashFamily,
    num_lookups: int,
    inputs: torch.Tensor,
) -> None:
    hash_embedding = HashEmbedding(num_embeddings, embedding_dim)

    hash_funcs = [hash_family.draw() for _ in range(num_lookups)]
    inputs_embeds = hash_embedding(inputs, hash_funcs)

    assert inputs_embeds.shape[:-1] == inputs.shape
    assert inputs_embeds.shape[-1] == hash_embedding.embedding_dim * num_lookups


def test_unified_embedding(
    num_embeddings: int,
    embedding_dim: int,
    features: dict,
    hash_family: HashFamily,
    inputs: torch.Tensor,
) -> None:
    unified_embedding = UnifiedEmbedding(num_embeddings, embedding_dim, features, hash_family)

    for name, num_lookups in features.items():
        assert unified_embedding(inputs, name).shape[-1] == embedding_dim * num_lookups
