# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # add RevIN if need
        self.add_revin = configs.add_revin
        if self.add_revin: 
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_mid_embedding=False):
        # input dim is: [batch_size, seq_len, channel]
        # add RevIN if need
        if self.add_revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                          trend=trend_init)
        if return_mid_embedding:
            seasonal_part, trend_part, mid_embedding = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                            trend=trend_init, return_mid_embedding=True)
        else:
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                            trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        
        # RevIN denorm:
        if self.add_revin:
            dec_out = self.revin_layer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        elif return_mid_embedding:
            # return both mid_embedding and trend_part, two parts
            if self.add_revin:
                return dec_out[:, -self.pred_len:, :], mid_embedding[:, -self.pred_len:, :], trend_part[:, -self.pred_len:, :], self.revin_layer.mean, self.revin_layer.stdev
            else:
                return dec_out[:, -self.pred_len:, :], mid_embedding[:, -self.pred_len:, :], trend_part[:, -self.pred_len:, :]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]