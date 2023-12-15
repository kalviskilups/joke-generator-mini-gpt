#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperparameters import HyperparametersAndDataEncoding

@torch.no_grad()
def estimate_loss():
    """
    Estimates the loss on training and validation sets.
    
    Returns:
    - dict: Loss values for training and validation sets.
    """

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparams.evaluation_iterations)
        for k in range(hyperparams.evaluation_iterations):
            X, Y = hyperparams.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """
    Represents one head of self-attention.
    """

    def __init__(self, head_size):
        """
        Initializes Head object.

        Args:
        - head_size (int): Size of the attention head.
        """

        super().__init__()
        self.key = nn.Linear(hyperparams.embedding_dimension, head_size, bias = False)
        self.query = nn.Linear(hyperparams.embedding_dimension, head_size, bias = False)
        self.value = nn.Linear(hyperparams.embedding_dimension, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(hyperparams.block_size, hyperparams.block_size)))

        self.dropout = nn.Dropout(hyperparams.dropout_rate)

    def forward(self, x):
        """
        Forward pass of the self-attention head.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after self-attention.
        """

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Represents multiple heads of self-attention in parallel.
    """

    def __init__(self, num_heads, head_size):
        """
        Initializes MultiHeadAttention object.

        Args:
        - num_heads (int): Number of attention heads.
        - head_size (int): Size of each attention head.
        """

        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, hyperparams.embedding_dimension)
        self.dropout = nn.Dropout(hyperparams.dropout_rate)

    def forward(self, x):
        """
        Forward pass for multiple heads of self-attention.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after multi-head self-attention.
        """

        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    Represents a simple linear layer followed by a non-linearity.
    """

    def __init__(self, n_embd):
        """
        Initializes FeedFoward object.

        Args:
        - n_embd (int): Embedding dimension.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(hyperparams.dropout_rate),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after feedforward computation.
        """

        return self.net(x)

class Block(nn.Module):
    """
    Represents a Transformer block: communication followed by computation.
    """

    def __init__(self, n_embd, n_head):
        """
        Initializes Block object.

        Args:
        - n_embd (int): Embedding dimension.
        - n_head (int): Number of heads for self-attention.
        """

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after Transformer block computation.
        """

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    Represents the simplified GPT Language Model.
    """
    
    def __init__(self):
        """
        Initializes GPTLanguageModel object.
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(hyperparams.vocab_size, hyperparams.embedding_dimension)
        self.position_embedding_table = nn.Embedding(hyperparams.block_size, hyperparams.embedding_dimension)
        self.blocks = nn.Sequential(*[Block(hyperparams.embedding_dimension, n_head = hyperparams.num_heads) for _ in range(hyperparams.num_layers)])
        self.ln_f = nn.LayerNorm(hyperparams.embedding_dimension) # final layer norm
        self.lm_head = nn.Linear(hyperparams.embedding_dimension, hyperparams.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for the neural network layers.

        Args:
        - module (nn.Module): Neural network module.

        Returns:
        - None
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT Language Model.

        Args:
        - idx (torch.Tensor): Input indices.
        - targets (torch.Tensor): Target indices for computing loss (optional).

        Returns:
        - torch.Tensor: Logits.
        - torch.Tensor or None: Loss if targets are provided, otherwise None.
        """

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = hyperparams.device))
        x = tok_emb + pos_emb 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens using the GPT Language Model.

        Args:
        - idx (torch.Tensor): Input indices.
        - max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: Generated indices.
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

if __name__ == "__main__":
    
    hyperparams = HyperparametersAndDataEncoding('hyperparameters.json', 'jokes_parsed.txt')
    
    model = GPTLanguageModel()
    m = model.to(hyperparams.device)

    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr = hyperparams.learning_rate)

    for iter in range(hyperparams.max_iterations):
        if iter % hyperparams.evaluation_interval == 0 or iter == hyperparams.max_iterations - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = hyperparams.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype = torch.long, device = hyperparams.device)

    print("Writing data to test.txt")
    with open('test.txt', 'w') as file:
        file.write(hyperparams.decode_list(m.generate(context, max_new_tokens=10000)[0].tolist()))
