# One Embedding to Serve Them All
This is a pure PyTorch implementation of unified embedding from the paper "[Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems](https://arxiv.org/abs/2305.12102)".


### Installation

```bash
pip install git+https://github.com/ArtemF42/onemb
```


### Usage

```python
import torch
from onemb import HashFamily, UnifiedEmbedding

num_embeddings, embedding_dim = 1_000, 128

features = {
    "feature_1": 4,
    "feature_2": 5,
    "feature_3": 1,
}
inputs = torch.tensor(
    [
        [0, 1, 1_000, 1_000_000],
        [0, 0, 2, 1_000_000_000],
    ]
)

unified_embedding = UnifiedEmbedding(
    num_embeddings,
    embedding_dim,
    features,
    hash_family=HashFamily(),
    padding_index=0,
)

inputs_embeds = unified_embedding(inputs, "feature_1")
print(inputs_embeds.shape)  # (2, 4, 512)
```