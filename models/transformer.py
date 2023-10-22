import torch
from torch import nn, Tensor
import math

from models.transformer_pytorch import TorchTransformer
from data import consts


class Transformer(nn.Module):

    def __init__(self, embed_size, num_layers=3, dim_feedforward=1024):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.transformer = TorchTransformer(
            d_model=embed_size, nhead=8, dim_feedforward=dim_feedforward, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers)
        self.position_encoding = PositionalEncoding(embed_size)

        self.drop_out = nn.Dropout(p=0.1)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask,
                batch_first=False):
        """
        Shape(batch_first=False):
            - src: (src_sequence_length, batch_size, embed_size)
            - tgt: (tgt_sequence_length, batch_size, embed_size)
            - return: (tgt_sequence_length, batch_size, embed_size)
        """
        if batch_first:
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.embed_size or tgt.size(2) != self.embed_size:
            raise RuntimeError("the feature number of src and tgt must be equal to embed_size")

        src_pos_enc = self.position_encoding(src)
        tgt_pos_enc = self.position_encoding(tgt)

        out = self.transformer(src_pos_enc, tgt_pos_enc, src_mask, tgt_mask, None, src_padding_mask,
                               tgt_padding_mask, memory_key_padding_mask)
        return out if not batch_first else out.permute(1, 0, 2)

    def encode(self, embed_src: Tensor, src_mask):
        return self.transformer.encoder(self.position_encoding(embed_src), src_mask)

    def decode(self, embed_tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask=None):
        return self.transformer.decoder(self.position_encoding(embed_tgt), memory, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 150):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)  # (max_len, 1, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, pretrained_weight=None):
        super(TokenEmbedding, self).__init__()
        if pretrained_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weight, freeze=False, padding_idx=consts.PAD)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=consts.PAD)
        self.emb_size = emb_size
        self.weight = self.embedding.weight

    def forward(self, tokens: Tensor):
        dim1, dim2 = tokens.size()
        idx = tokens.reshape(-1).unsqueeze(-1)
        idx = idx.expand(dim1 * dim2, self.emb_size)
        embeddings = torch.gather(self.weight, dim=0, index=idx)
        embeddings = embeddings.reshape([dim1, dim2, -1])
        return embeddings * math.sqrt(self.emb_size)
        # return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
