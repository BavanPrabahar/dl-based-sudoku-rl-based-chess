"""
chess_rl/model.py
Policy-value residual network (AlphaZero-style, small enough for a 6 GB laptop GPU).

input  (B, 17, 8, 8) planes from chess_rl.encoding
policy (B, NUM_ACTIONS) move logits (illegal moves are masked by the caller)
value  (B,) logit of the probability that the side to move solves the position (puzzle success
       during RL; used as the REINFORCE baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_rl.encoding import NUM_ACTIONS, NUM_PLANES


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 10, channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(32 * 64, NUM_ACTIONS))
        self.value = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(4 * 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.policy(h), self.value(h).squeeze(1)


def save_checkpoint(path, model, meta, half=True):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"arch_kwargs": model.arch_kwargs, "state_dict": sd, "meta": meta}, path)


def build(blocks=10, channels=128):
    m = ChessNet(blocks, channels)
    m.arch_kwargs = {"blocks": blocks, "channels": channels}
    return m


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    m = build(**ckpt["arch_kwargs"])
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return m.to(device).eval(), ckpt.get("meta", {})


if __name__ == "__main__":
    m = build()
    p, v = m(torch.zeros(2, NUM_PLANES, 8, 8))
    print(f"ChessNet {sum(x.numel() for x in m.parameters()):,} params, policy {tuple(p.shape)}, value {tuple(v.shape)}")
