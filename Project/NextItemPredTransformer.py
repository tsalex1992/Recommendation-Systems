from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


@dataclass
class ModelDimensions:
    pre_trained_item_embeddings: Tensor
    model_input_length: int
    model_hidden_dim: int
    vocab_size: int
    n_attention_heads: int
    n_decoder_layers: int
    pre_trained_user_embeddings: Optional[Tensor]
    use_concat_user_embedding: Optional[bool] = False


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        """Multi-head attention layer

        Args:
            n_state: number of hidden units
            n_head: number of attention heads
        """
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x)
            v = self.value(x)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        # Output dimensions: [n_batch, n_ctx, n_head, n_head_dim]
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class ItemDecoder(nn.Module):
    def __init__(
        self,
        pre_trained_item_embeddings: Tensor,
        input_length: int,
        hidden_dim: int,
        n_attention_heads: int,
        n_layers: int,
        pre_trained_user_embeddings: Optional[Tensor] = None,
        use_concat_user_embedding: bool = False,
    ):
        """Item decoder

        Args:
            pre_trained_item_embeddings: pre-trained item embeddings, of shape [n_vocab, n_state] ordered by item id
            input_length: number of items in context
            hidden_dim: number of hidden units
            n_attention_heads: number of attention heads
            n_layers: number of decoder layers
            pre_trained_user_embeddings: pre-trained user embeddings, of shape [n_users, n_state] ordered by user id
        """
        super().__init__()

        # Note we need to reserve special tokens for start and for padding(which is similar to end)
        self.use_concat_user_embedding = use_concat_user_embedding
        self.items_embedding = nn.Embedding.from_pretrained(pre_trained_item_embeddings)
        self.users_embedding = (
            nn.Embedding.from_pretrained(pre_trained_user_embeddings)
            if pre_trained_user_embeddings is not None
            else None
        )
        self.positional_embedding = nn.Parameter(torch.empty(input_length, hidden_dim))
        # adds reshape layer after linear to get output of shape [batch_size, input_length, hidden_dim]
        self.time_embedding = nn.Sequential(
            nn.Linear(input_length, input_length * hidden_dim),
            nn.Unflatten(1, (input_length, hidden_dim)),
            nn.ReLU(),
        )

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(hidden_dim, n_attention_heads) for _ in range(n_layers)]
        )
        self.ln = LayerNorm(hidden_dim)

        mask = torch.empty(input_length, input_length).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        items: Tensor,
        user: Optional[Tensor],
        kv_cache: Optional[dict] = None,
        user_interactions_times: Optional[Tensor] = None,
        prediction_time: Optional[Tensor] = None,
    ):
        """
        items : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        user : torch.LongTensor, shape = (batch_size,)
            the user id
        kv_cache : dict, optional
            a dictionary of cached key/value projections
        user_interactions_times : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the time of ratings of each item for the user in the context in unix epoch seconds
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        # Embed items
        items = (
            self.items_embedding(items)
            + self.positional_embedding[offset : offset + items.shape[-1]]
        )
        if prediction_time is not None and user_interactions_times is not None:
            # add prediction time to the first time in context 0f each batch
            # Works as concat since the first interaction time is all zeros for SOS
            user_interactions_times[:, 0] += prediction_time
        if user_interactions_times is not None:
            items = items + self.time_embedding(user_interactions_times)

        if self.users_embedding is not None and user is not None:
            embed_user = self.users_embedding(user)
            if self.use_concat_user_embedding:
                # add the user embedding to the first item in context 0f each batch
                # Works as concat since the first embedding is all zeros for SOS
                items[:, 0, :] += embed_user
            else:
                # broadcast user embedding to all items in context
                embed_user = embed_user.unsqueeze(1).expand(-1, items.shape[1], -1)
                items = items + embed_user
        for block in self.blocks:
            items = block(items, mask=self.mask, kv_cache=kv_cache)

        items = self.ln(items)
        logits = (
            items @ torch.transpose(self.items_embedding.weight.to(items.dtype), 0, 1)
        ).float()

        return logits


class NextItemPredTransformer(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.vocab_size = dims.vocab_size
        self.decoder = ItemDecoder(
            self.dims.pre_trained_item_embeddings,
            # adding 2 for SOS and EOS
            self.dims.model_input_length + 2,
            self.dims.model_hidden_dim,
            self.dims.n_attention_heads,
            self.dims.n_decoder_layers,
            self.dims.pre_trained_user_embeddings,
            self.dims.use_concat_user_embedding,
        )

    def logits(
        self,
        items: torch.Tensor,
        user: Optional[torch.Tensor] = None,
        user_interactions_times: Optional[torch.Tensor] = None,
        prediction_time: Optional[torch.Tensor] = None,
    ):
        return self.decoder(
            items,
            user,
            user_interactions_times=user_interactions_times,
            prediction_time=prediction_time,
        )

    def forward(
        self,
        items: torch.Tensor,
        user: Optional[torch.Tensor] = None,
        user_interactions_times: Optional[torch.Tensor] = None,
        prediction_time: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(
            items,
            user,
            user_interactions_times=user_interactions_times,
            prediction_time=prediction_time,
        )

    # Add no grad to avoid memory leak
    @torch.no_grad()
    def predict(
        self,
        items: torch.Tensor,
        pred_index: int,
        user: Optional[torch.Tensor] = None,
        user_interactions_times: Optional[torch.Tensor] = None,
        prediction_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.decoder(
            items,
            user,
            user_interactions_times=user_interactions_times,
            prediction_time=prediction_time,
        )
        return logits[:, pred_index]

    @property
    def device(self):
        return next(self.parameters()).device
