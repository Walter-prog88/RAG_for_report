import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)   # (T,)

        tok_x = self.token_embed(input_ids)   # (B,T,d_model)
        pos_x = self.pos_embed(positions)     # (T,d_model)
        x = tok_x + pos_x                     # (B,T,d_model)

        logits = self.lm_head(x)              # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V),
                targets.view(B*T)
            )

        return logits, loss


vocab_size = 10
d_model = 8
block_size = 4

model = TinyLM(vocab_size, d_model, block_size)

tokens = torch.tensor([[1, 2, 3, 4]])
input_ids = tokens[:, :-1]   # [[1,2,3]]
targets   = tokens[:, 1:]    # [[2,3,4]]

logits, loss = model(input_ids, targets)

print("input_ids shape:", input_ids.shape)
print("targets shape:", targets.shape)
print("logits shape:", logits.shape)
print("loss:", loss.item())