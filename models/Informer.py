import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # add RevIN if need
        self.add_revin = configs.add_revin
        if self.add_revin: 
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_mid_embedding=False):
        # input dim: [batch_size, seq_len, channel]
        # add RevIN if need
        if self.add_revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # check whether return mid_embedding
        if return_mid_embedding:
            dec_out, mid_embedding = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, return_mid_embedding=True)
        else:
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        # RevIN denorm:
        if self.add_revin:
            dec_out = self.revin_layer(dec_out, 'denorm')

        # return results
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        elif return_mid_embedding:
            # return both mid_embedding and trend_part, two parts
            if self.add_revin:
                return dec_out[:, -self.pred_len:, :], mid_embedding[:, -self.pred_len:, :], self.revin_layer.mean, self.revin_layer.stdev
            else:
                return dec_out[:, -self.pred_len:, :], mid_embedding[:, -self.pred_len:, :]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
