__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# https://github.com/luo3300612/Visualizer
# ! 注意：模型训练时不要调用get_local！！！
# ! 因为他会存储中间的注意力权重信息，导致这些内存一直没能释放，从而内存泄漏了！！！
# from visualizer import get_local

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, 
                 c_in:int, 
                 context_window:int, 
                 target_window:int, 
                 patch_len:int, 
                 stride:int, 
                 max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, 
                 d_model=128, 
                 n_heads=16, 
                 d_k:Optional[int]=None, 
                 d_v:Optional[int]=None,
                 d_ff:int=256, 
                 norm:str='BatchNorm', 
                 attn_dropout:float=0., 
                 dropout:float=0., 
                 act:str="gelu", 
                 key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, 
                 attn_mask:Optional[Tensor]=None, 
                 res_attention:bool=True, 
                 pre_norm:bool=False, 
                 store_attn:bool=False,
                 pe:str='zeros', 
                 learn_pe:bool=True, 
                 fc_dropout:float=0., 
                 head_dropout = 0, 
                 padding_patch = None,
                 pretrain_head:bool=False, 
                 head_type = 'flatten', 
                 individual = False, 
                 revin = True, 
                 affine = True, 
                 subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        # 这里默认是做RevIN的，且维度c_in为channel维
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        # PS：这里context_window就等于seq_len，target_window就等于pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        # 获得总的patch数
        patch_num = int((context_window - patch_len) / stride + 1)
        
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone
        # 核心的Encoder框架
        # （PS：embedding和位置编码也都会在TSTiEncoder内完成）
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        # 也即最后一个"展平 & 线性层"
        # 其将(d_model, patch_num)的输入战平后并映射至(1, pred_len)的输出
        self.head_nf = d_model * patch_num  # d_model和patch总数的乘积，为encoder的输出
        self.n_vars = c_in  # n_vars就设置为channel数
        self.pretrain_head = pretrain_head  # 默认为False
        self.head_type = head_type  # 默认为"flatten"
        self.individual = individual  # 默认为False

        if self.pretrain_head:
            # 如果要预训练的话，那么改用一个一维卷积的预测头
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            # 如果不预训练线性层的话，那么调用Flatten_Head
            # 它会完成：展平 + 线性映射 + dropout （不过这里的dropout默认设置为0）
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    # @profile
    def forward(self, z):                                                   # z: [bs x nvars x seq_len]
        # norm
        # 1、先做RevIN归一化
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        # 2、做patching分割
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        # unfold函数啊按照选定的尺寸与步长来切分矩阵，相当于滑动窗口操作，也即只有卷、没有积
        # 参数为（dim,size,step）：dim表明想要切分的维度，size表明切分块（滑动窗口）的大小，step表明切分的步长
        # 这里用于从一个分批输入的张量中提取滑动的局部块
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        # 3、经过encoder主干模型（PS：embedding和位置编码都会在backbone内完成）
        z = self.backbone(z)                                                # z: [bs x nvars x d_model x patch_num]
        # 4、展平 + 线性映射（+dropout）
        z = self.head(z)                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        # 5、再用RevIN做denorm回来
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        # 这里的dropout会使用fc_dropout
        return nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv1d(in_channels=head_nf, out_channels=vars, kernel_size=1)
                )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        # 最后一个"展平 & 线性层"
        # 其将(d_model, patch_num)的输入战平后并映射至(1, pred_len)的输出
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # 将d_model和patch_num维展平，并做线性映射到1*pred_len维
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            # 遍历各个channel
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            # 先展平、再线性映射，还外加一个dropout（不过默认为0）
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  # "i" means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        # 主干网络
        # 包括完成embedding和encoder部分
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        # 这里用一个线性层做value-embedding，将输入从patch_len维映射到d_model维
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        # 有很多位置编码备选项，这里默认使用"zeros"，也即初始化为[-0.02,0.02]区间内的均匀分布
        # * 同时，learn_pe为True，说明这里位置编码默认是可学习的
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        # 最关键的encoder部分
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    
    # @profile 
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # 1、value-embedding，将输入从patch_len映射到d_model
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        # channel independence
        u = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))      # u: [(bs * nvars) x patch_num x d_model]
        
        # 2、将embedding和位置编码相加，再经过dropout
        u = self.dropout(u + self.W_pos)                                         # u: [(bs * nvars) x patch_num x d_model]

        # Encoder
        # 3、进入encoder做处理
        z = self.encoder(u)                                                      # z: [(bs * nvars) x patch_num x d_model]
        
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        # 一共有n_layers层encoder
        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    # @profile
    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                # 上面的这个是原来的code
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # # ! 注意这里有一个小的改动，目的是为了可视化注意力权重！！！
                # output = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: 
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        
        # 如果未设置，那么d_k和d_v都默认为(d_model // n_heads)
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        # 第一个是Attention中的norm
        # 注意，PatchTST参考TST的设计，其中默认使用的是BatchNorm
        # 而不像In/Auto/FED-former等默认使用LayerNorm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        # 这里默认使用GELU作为MLP的中间层，dropout也夹在中间而非末尾
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        # 第二个是FFN中的Norm，同样也是默认为BatchNorm，也可能是LayerNorm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        # 此二者默认为False？？
        self.pre_norm = pre_norm
        self.store_attn = store_attn


    # @get_local('attn')
    # @profile
    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        # 1、在注意力前先做一个BatchNorm（也可能是LayerNorm？）
        # 不过self.pre_norm默认为False，也就是默认不做normalization？？
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        # 2、多头自注意力
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        # 3.1 Add：先做个dropout、然后和残差连接的src相加
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        # 3.2 Norm：然后也做一个BatchNorm（也可能是LayerNorm）
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        # 4、FFN
        # 这里FFN默认为使用GELU激活的MLP
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        # 5、也是Add + Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        # 用一个线性映射 + dropout映射回原来的维度
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    # @profile
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        # 先将线性变换对应应用在QKV矩阵上
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        # 多头点积注意力输出结果
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        # transpose + to_out 变换回原来的维度
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        
        # scale默认为：sqrt(head_dim)
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    # @profile
    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # 先计算Q和K的点积
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        # （可选）在softmax前，从之前层加入注意力分数
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        # 将注意力分数再进经过softmax
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        # 将注意力分数和v计算后得到最终输出
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

