import pytest
import torch


from onemb.embedding import HashEmbedding, UnifiedEmbedding
from onemb.hash_utils import HashFamily, HashFunc


@pytest.fixture
def num_embeddings() -> int:
    return 1024


@pytest.fixture
def embedding_dim() -> int:
    return 128


@pytest.fixture
def hash_embedding(num_embeddings, embedding_dim) -> HashEmbedding:
    return HashEmbedding(num_embeddings, embedding_dim)


@pytest.fixture
def hash_family() -> HashFamily:
    return HashFamily()


@pytest.fixture
def inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1_000, 1_000_000],
            [0, 0, 2, 1_000_000_000],
        ]
    )


@pytest.fixture
def features() -> dict[str, int]:
    return {"feature_1": 1, "feature_2": 4, "feature_3": 2}
