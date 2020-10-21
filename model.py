import torch
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim

import math

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class TextClassifier(nn.Module):  # all modules should subclass nn.Module
    
    def __init__(self, vocab_size: int, n_classes: int, model_dim=256, n_heads=4, n_layers=4,
                 ff_dim=1024, dropout=0.1, activation='relu', padding_idx=0):
        super().__init__() # remember to call super
        
        self.padding_idx = padding_idx
        embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=model_dim, padding_idx=padding_idx)
        pos_enc = PositionalEncoding(d_model=model_dim)
        self.embeddings = nn.Sequential(embeddings, pos_enc)

        enc_layer = TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                                   dropout=dropout, activation=activation)
        self.encoder = TransformerEncoder(enc_layer, num_layers=n_layers)
        
        self.cls_proj = nn.Linear(model_dim, n_classes)
        
    def forward(self, texts, lengths, out='probs'):
        # [Batch x Length] --> [Batch x Length x HidDim]
        embs = self.embeddings(texts)
        
        # Ignore True positions in seq
        mask = (texts == self.padding_idx) # [B x L]
        
        # some modules accept batch as second dimension
        embs = embs.transpose(0, 1)            #[Length x Batch x HidDim]
        feats = self.encoder(embs, src_key_padding_mask=mask)
        feats = feats.transpose(0, 1)          #[Batch x Length x HidDim]
        
        #TODO: sentence representation
        max_feats, max_indices = feats.max(dim=1, keepdim=False)  #[Batch x HidDim]
        cls_logits = self.cls_proj(max_feats)       #[Batch x Classes]
        
        if out in (None, 'raw', 'logits'): 
            return cls_logits
        elif out == 'probs':
            return F.softmax(cls_logits, dim=1)
        elif out == 'log_probs':
            return F.log_softmax(cls_logits, dim=1)
        else:
            raise Exception(f'unknown arg {out}')

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
