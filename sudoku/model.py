"""
sudoku/model.py
Deep-learning Sudoku solvers in PyTorch.

* SudokuCNN           - the original baseline: a plain stack of 3x3 convolutions.
* SudokuConstraintNet - a residual network whose blocks exchange information along the
                        three Sudoku constraint groups (row, column, 3x3 box). A 3x3 convolution
                        only sees neighbouring cells, but a Sudoku rule links a cell to its whole
                        row, column and box, so every block pools each group and broadcasts the
                        result back to its cells (a message-passing step over the constraint graph).

Both take a (B, 9, 9, 9) one-hot board (channel d = digit d+1, all-zero = blank) and return
(B, 9, 9, 9) logits over the 9 digits for every cell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SudokuCNN(nn.Module):
    """Baseline: Conv3x3 + BatchNorm + ReLU stack (the original repository model)."""

    def __init__(self, num_layers: int = 9, hidden_dim: int = 64):
        super().__init__()
        layers = [nn.Conv2d(9, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU()]
        layers.append(nn.Conv2d(hidden_dim, 9, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _group_pool(x: torch.Tensor):
    """Mean over each cell's row, column and 3x3 box, broadcast back to the 9x9 grid."""
    row = x.mean(dim=3, keepdim=True).expand_as(x)
    col = x.mean(dim=2, keepdim=True).expand_as(x)
    box = F.avg_pool2d(x, kernel_size=3, stride=3)
    box = box.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)
    return row, col, box


class ConstraintBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.mix = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.local(x)))
        row, col, box = _group_pool(h)
        h = self.bn2(self.mix(torch.cat([h, row, col, box], dim=1)))
        return F.relu(x + h)


class SudokuConstraintNet(nn.Module):
    def __init__(self, num_blocks: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(9, hidden_dim, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ConstraintBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_dim, 9, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def build_model(arch: str = "constraint", **kwargs) -> nn.Module:
    if arch == "cnn":
        return SudokuCNN(**kwargs)
    if arch == "constraint":
        return SudokuConstraintNet(**kwargs)
    raise ValueError(f"unknown architecture {arch!r}")


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    """Load a checkpoint written by sudoku/train.py (stores the architecture with the weights)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], **ckpt["arch_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    for arch, kw in [("cnn", {}), ("constraint", {})]:
        m = build_model(arch, **kw)
        out = m(torch.zeros(4, 9, 9, 9))
        print(f"{arch:10s} params {sum(p.numel() for p in m.parameters()):,}  output {tuple(out.shape)}")
