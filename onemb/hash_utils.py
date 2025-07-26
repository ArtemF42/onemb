import torch
import torch.nn as nn


class HashFunc(nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> None:
        super(HashFunc, self).__init__()
        # allow saving hash function with its parent `nn.Module` for reproducibility
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("p", p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a * x + self.b) % self.p


class HashFamily:
    def __init__(self, p: int | torch.Tensor = 0x1FFFFFFFFFFFFFFF) -> None:
        self.p = p if isinstance(p, torch.Tensor) else torch.tensor(p)

    def draw(self) -> HashFunc:
        p = self.p  # alias

        a = torch.randint(1, p - 1, size=())
        b = torch.randint(0, p - 1, size=())

        return HashFunc(a, b, p)
