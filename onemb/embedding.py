import torch
import torch.nn as nn

from .hash_utils import HashFamily, HashFunc


class HashEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_index: int | None = None,  # avoid collision with `torch.nn.Embedding.padding_idx`
    ) -> None:
        super(HashEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.padding_index = padding_index

    def forward(self, inputs: torch.Tensor, hash_funcs: list[HashFunc]) -> torch.Tensor:
        inputs_embeds = torch.cat(
            [
                super(HashEmbedding, self).forward(hash_func(inputs) % self.num_embeddings)
                for hash_func in hash_funcs
            ],
            dim=-1,
        )

        if self.padding_index is not None:
            inputs_embeds.masked_fill_((inputs == self.padding_index).unsqueeze(-1), 0.0)

        return inputs_embeds


class UnifiedEmbedding(HashEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        features: dict[str, int],
        hash_family: HashFamily,
        padding_index: int | None = None,
    ) -> None:
        super(UnifiedEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_index=padding_index,
        )
        self.features = features
        self.hash_family = hash_family

        self.feature_hash_funcs = {}

        for name, num_lookups in features.items():
            self.add_feature(name, num_lookups)

    def forward(self, inputs: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.feature_hash_funcs:
            msg = f"Unknown feature {name!r}. Please use `add_feature` to register it before using."
            raise ValueError(msg)

        return super(UnifiedEmbedding, self).forward(inputs, self.feature_hash_funcs[name])

    def add_feature(self, name: str, num_lookups: int) -> None:
        if name in self.feature_hash_funcs:
            msg = f"Cannot overwrite existing feature {name!r}."
            raise ValueError(msg)

        self.feature_hash_funcs[name] = [self.hash_family.draw() for _ in range(num_lookups)]

    @property
    def num_features(self) -> int:
        return len(self.features)
