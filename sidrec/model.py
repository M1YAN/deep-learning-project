import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import BOS, CODEBOOK_SIZES, HISTORY_LENGTH, PAD


class Generator(nn.Module):
    def __init__(self, width=96, layers=2, heads=4, dropout=0.1, *, codebook_sizes=CODEBOOK_SIZES):
        super().__init__()
        offsets = [BOS + 1 + sum(codebook_sizes[:level]) for level in range(len(codebook_sizes))]
        self.register_buffer("offsets", torch.tensor(offsets))
        self.embedding = nn.Embedding(BOS + 1 + sum(codebook_sizes), width, padding_idx=PAD)
        self.history_position = nn.Embedding(HISTORY_LENGTH * len(codebook_sizes), width)
        self.target_position = nn.Embedding(len(codebook_sizes), width)
        encoder_layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout, activation="gelu", batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            width, heads, 4 * width, dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in codebook_sizes)

    def encode(self, history, sids):
        pad_items = history.lt(0)
        tokens = (sids[history.clamp_min(0)] + self.offsets).masked_fill(
            pad_items.unsqueeze(-1), PAD,
        ).flatten(1)
        padding = tokens.eq(PAD)
        position = torch.arange(tokens.shape[1], device=tokens.device)
        embedded = self.embedding(tokens) + self.history_position(position)
        return self.encoder(embedded, src_key_padding_mask=padding), padding

    def decode(self, previous_codes, memory, padding):
        length = previous_codes.shape[1] + 1
        bos = previous_codes.new_full((len(previous_codes), 1), BOS)
        tokens = torch.cat((bos, previous_codes + self.offsets[:length - 1]), dim=1)
        position = torch.arange(length, device=tokens.device)
        embedded = self.embedding(tokens) + self.target_position(position)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        return self.decoder(embedded, memory, tgt_mask=causal, memory_key_padding_mask=padding)

    def loss(self, rows, sids, *, per_level=False):
        memory, padding = self.encode(rows[:, :-1], sids)
        labels = sids[rows[:, -1]]
        decoded = self.decode(labels[:, :-1], memory, padding)
        losses = torch.stack([
            F.cross_entropy(head(decoded[:, level]), labels[:, level])
            for level, head in enumerate(self.heads)
        ])
        return losses if per_level else losses.mean()


class CatalogBeam:
    """Dense trie transitions, with final transitions returning unique item indices."""

    def __init__(self, sids, device, *, codebook_sizes=CODEBOOK_SIZES):
        if len(sids) < 10:
            raise ValueError("At least ten catalog items are required for top-10 ranking")
        if sids.ndim != 2 or sids.shape[1] != len(codebook_sizes):
            raise ValueError("The SID length must match the number of codebooks")
        prefixes = {(): 0}
        self.transitions = []
        for level, size in enumerate(codebook_sizes):
            next_prefixes = {}
            table = np.full((len(prefixes), size), -1, dtype=np.int64)
            for item, codes in enumerate(sids):
                prefix = tuple(codes[:level + 1])
                code = prefix[-1]
                if not 0 <= code < size:
                    raise ValueError(f"Invalid code {code} at level {level}")
                state = prefixes[prefix[:-1]]
                if level + 1 == len(codebook_sizes):
                    if table[state, code] != -1:
                        raise ValueError("Duplicate SID in the catalog")
                    table[state, code] = item
                else:
                    table[state, code] = next_prefixes.setdefault(prefix, len(next_prefixes))
            self.transitions.append(torch.as_tensor(table, device=device))
            prefixes = next_prefixes

    def rank(self, model, rows, sids, beam_width, top_k=10):
        memory, padding = model.encode(rows[:, :-1], sids)
        batch = len(rows)
        scores = memory.new_zeros((batch, 1))
        states = rows.new_zeros((batch, 1))
        paths = rows.new_zeros((batch, 1, 0))

        for level, transitions in enumerate(self.transitions):
            width = scores.shape[1]
            hiddens = model.decode(
                paths.flatten(0, 1),
                memory.repeat_interleave(width, dim=0),
                padding.repeat_interleave(width, dim=0),
            )
            size = transitions.shape[1]
            logprobs = F.log_softmax(model.heads[level](hiddens[:, -1]), dim=-1)
            reachable = transitions[states.clamp_min(0)]
            candidates = (scores.unsqueeze(-1) + logprobs.reshape(batch, width, size))
            candidates = candidates.masked_fill(reachable.lt(0), -torch.inf)
            final = level == len(self.transitions) - 1
            limit = top_k if final else beam_width
            scores, indices = candidates.flatten(1).topk(min(limit, width * size), dim=1)
            states = reachable.flatten(1).gather(1, indices).masked_fill(~scores.isfinite(), -1)
            if not final:
                parents = indices // size
                prefixes = paths.gather(1, parents.unsqueeze(-1).expand(-1, -1, level))
                paths = torch.cat((prefixes, (indices % size).unsqueeze(-1)), dim=-1)

        return states
