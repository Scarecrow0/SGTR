"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
from torch import Tensor, nn


class Transformer(nn.Module):
    """
    Transformer structure for DETR
    """

    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg
        d_model = cfg.MODEL.DETR.TRANSFORMER.D_MODEL
        nhead = cfg.MODEL.DETR.TRANSFORMER.N_HEAD
        num_encoder_layers = cfg.MODEL.DETR.TRANSFORMER.NUM_ENC_LAYERS
        num_decoder_layers = cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS
        dim_feedforward = cfg.MODEL.DETR.TRANSFORMER.DIM_FFN
        dropout = cfg.MODEL.DETR.TRANSFORMER.DROPOUT_RATE
        activation = cfg.MODEL.DETR.TRANSFORMER.ACTIVATION
        normalize_before = cfg.MODEL.DETR.TRANSFORMER.PRE_NORM
        return_intermediate_dec = cfg.MODEL.DETR.TRANSFORMER.RETURN_INTERMEDIATE_DEC

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, return_intermediate=True
        )
        self.num_encoder_layers = num_encoder_layers
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def foreground_mask_gen(self, mask, ent_coords):
        N_, H_, W_ = mask.shape
        valid_h = torch.sum(torch.sum(~mask, 1) > 0, 1)
        valid_w = torch.sum(torch.sum(~mask, 2) > 0, 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=mask.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=mask.device),
        )
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid_normed = (grid.unsqueeze(0).expand(N_, -1, -1, -1)) / scale
        valid_grid_mask = grid_normed <= 1

        valid_grid_mask = ~torch.logical_and(
            valid_grid_mask[..., 0], valid_grid_mask[..., 1]
        )

        cnter = ent_coords[:, :, :2].unsqueeze(-2)
        scale = ent_coords[:, :, 2:].unsqueeze(-2)
        _, num_q, _ = ent_coords.shape
        mask_size = W_ * H_

        sub_res = grid_normed.reshape(N_, -1, 2).unsqueeze(-3).repeat(
            1, num_q, 1, 1
        ) - cnter.repeat(1, 1, mask_size, 1)

        beta = 0.6
        # ent_confidence = torch.softmax(ent_cls, -1)[..., :-1].max(-1)[0]
        logits = sub_res ** 2 / (scale.repeat(1, 1, mask_size, 1) ** 2 * beta) * -1
        g_map = logits.sum(-1).exp()  # batch_size, num_q, mask_size

        g_map_filtered = g_map.clone()
        g_map_filtered[valid_grid_mask.flatten(1).unsqueeze(1).repeat(1, num_q, 1)] = 0
        max_x = g_map_filtered.max(-1)[1] % W_ / W_
        max_y = torch.div(g_map_filtered.max(-1)[1], W_, rounding_mode='trunc') / H_
        max_coord = torch.cat((max_x.unsqueeze(-1), max_y.unsqueeze(-1)), dim=-1)
        max_enc_feat_idx = g_map_filtered.max(-1)[1]

        # g_map_conf = (
        #     g_map * ent_confidence.unsqueeze(-1).repeat(1, 1, mask_size)
        # ).mean(1)

        g_map = g_map.mean(1)  # batch_size, mask_size

        def minmax_scale(arr):
            return (arr - arr.min(1)[0].unsqueeze(-1)) / (
                    arr.max(1)[0] - arr.min(1)[0]
            ).unsqueeze(-1)

        g_map = minmax_scale(g_map)
        # g_map_conf = minmax_scale(g_map_conf)

        g_map[valid_grid_mask.flatten(1)] = 0
        # g_map_conf[valid_grid_mask.flatten(1)] = 0

        # visualize the heat map for debug
        # maps = ascii_heatmap(g_map[0].reshape( H_, W_))

        return g_map.reshape(-1, H_, W_).detach(), max_coord, max_enc_feat_idx

    def extract_precompute_box_features(self, mask, memory, precompute_prop):
        """[summary]

        Args:
            mask ([type]): [description]
            memory ([type]): Size x bs x dim
            precompute_prop ([type]): [description]
        """
        bs, H_, W_ = mask.shape
        device = mask.device
        selected_mems = []

        max_ent_len = self.cfg.MODEL.DETR.NUM_QUERIES
        # ent_coords = torch.stack([prp.box.tensor for prp in precompute_prop ]) # B, N, Dim
        for idx, prp in enumerate(precompute_prop):
            ent_coords = torch.unsqueeze(prp.box.tensor, 0).to(device)
            g_map, max_coord, max_enc_feat_idx = self.foreground_mask_gen(
                torch.unsqueeze(mask[idx], 0), ent_coords,
            )
            selected_mem = memory.transpose(1, 0)[idx][max_enc_feat_idx[0]]
            while len(selected_mem) < max_ent_len:
                selected_mem = torch.cat((selected_mem,
                                          selected_mem[-(max_ent_len - len(selected_mem)):]))
            selected_mems.append(selected_mem)

        return torch.stack(selected_mems).transpose(1, 0)

    def forward(self, src, mask, query_embed, pos_embed, enc_return_lvl=-1, precompute_prop=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # src_flatten = src.flatten(2).permute(2, 0, 1)
        # pos_embed_flatten = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask_flatten = mask.flatten(1)
        #
        # memory, mem_inter = self.encoder(src_flatten,
        #                                  src_key_padding_mask=mask_flatten,
        #                                  pos=pos_embed_flatten)
        # mem_inter = mem_inter.permute(0, 2, 3, 1).view(
        #     self.num_encoder_layers, bs, c, h, w
        # )
        #
        (mask_flatten,
         mem_inter, memory,
         pos_embed_flatten) = self.apply_encoder(mask, pos_embed, src)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if precompute_prop is not None:
            memory = memory + pos_embed_flatten
            pos_embed_flatten = None
            memory = self.extract_precompute_box_features(mask, memory, precompute_prop)
            mask_flatten = None

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask_flatten,
            pos=pos_embed_flatten,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), mem_inter[enc_return_lvl]

    def apply_encoder(self, mask, pos_embed, src):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src_flatten = src.flatten(2).permute(2, 0, 1)
        pos_embed_flatten = pos_embed.flatten(2).permute(2, 0, 1)
        mask_flatten = mask.flatten(1)

        memory, mem_inter = self.encoder(src_flatten,
                                         src_key_padding_mask=mask_flatten,
                                         pos=pos_embed_flatten)
        mem_inter = mem_inter.permute(0, 2, 3, 1).view(
            self.num_encoder_layers, bs, c, h, w
        )
        return mask_flatten, mem_inter, memory, pos_embed_flatten


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            reduce_mask: Optional[Tensor] = None,
    ):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            # if reduce_mask is not None:
            #     output *= (
            #         reduce_mask.flatten(1)
            #         .transpose(1, 0)
            #         .unsqueeze(-1)
            #         .repeat(1, 1, output.shape[-1])
            #     )

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return output, torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            adaptive_att_weight: Optional[Tensor] = None,  # pos embedding for query
            return_value_sum=False
    ):
        output = tgt

        intermediate = []
        inter_value_sum = []
        inter_att_weight = []

        for layer in self.layers:
            output_dict = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                adaptive_att_weight=adaptive_att_weight,
                return_value_sum=True,
            )
            output = output_dict['tgt']

            if self.return_intermediate:
                inter_att_weight.append(output_dict['att_weight'])
                inter_value_sum.append(output_dict['value_sum'])
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
            output_dict['tgt'] = output

        if self.return_intermediate:
            if return_value_sum:
                return torch.stack(intermediate), torch.stack(inter_att_weight), torch.stack(inter_value_sum)
            else:
                return torch.stack(intermediate)

        if return_value_sum:
            return output, output_dict['att_weight'], output_dict['value_sum']
        else:
            return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttentionAdaptWeight(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttentionAdaptWeight(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,  # pos_embedding for key
            query_pos: Optional[Tensor] = None,  # pos embedding for query
            adaptive_att_weight: Optional[Tensor] = None,  # pos embedding for query
            return_value_sum=False,

    ):
        # self attention on query
        q = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, q, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross attention between the query after self-attention and memory
        tgt2, att_weight, value_sum = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            adaptive_att_weight=adaptive_att_weight
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_value_sum:
            return {'tgt': tgt, "att_weight": att_weight, "value_sum": value_sum}
        return tgt

    def forward_pre(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            adaptive_att_weight: Optional[Tensor] = None,  # pos embedding for query
            return_value_sum=False,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, att_weight, value_sum = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            adaptive_att_weight=adaptive_att_weight
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_value_sum:
            return {'tgt': tgt, "att_weight": att_weight, "value_sum": value_sum}
        return tgt

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            adaptive_att_weight: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            return_value_sum=False,
    ):
        """

        Args:
            tgt:
            memory:
            tgt_mask:
            memory_mask:
            tgt_key_padding_mask:
            memory_key_padding_mask:
            pos:
            query_pos:

        Returns:

        """
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                adaptive_att_weight,
                return_value_sum=return_value_sum,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            adaptive_att_weight,
            return_value_sum=return_value_sum,
        )


class MultiheadAttentionAdaptWeight(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
    ):
        super(MultiheadAttentionAdaptWeight, self).__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim
        )

    def forward(
            self,
            query,
            key,
            value,
            adaptive_att_weight=None,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
    ):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward_adapt_weights(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                adaptive_att_weight=adaptive_att_weight,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward_adapt_weights(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                adaptive_att_weight=adaptive_att_weight
            )


from torch.overrides import has_torch_function, handle_torch_function
import torch.nn.functional as F
from torch.nn.functional import linear, softmax
from torch._jit_internal import Optional, Tuple


def multi_head_attention_forward_adapt_weights(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_v_sum: bool = True,
        attn_mask: Optional[Tensor] = None,
        adaptive_att_weight: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        adaptive_att_weight: adatively control the softmax attention weights, instead of modify the query and key value features
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - adaptive_att_weight: math:`(N, L, S)` where N is the batch size,
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (
            query,
            key,
            value,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            out_proj_weight,
            out_proj_bias,
        )
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
                tens_ops
        ):
            return handle_torch_function(
                multi_head_attention_forward_adapt_weights,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
            )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim: (embed_dim * 2)]
            )
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            )
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

    if adaptive_att_weight is not None:
        adaptive_att_weight = adaptive_att_weight.unsqueeze(1).repeat(1, num_heads, 1, 1).reshape(-1, tgt_len, src_len)
        assert adaptive_att_weight.shape == attn_output_weights.shape
        attn_output_weights += torch.log(adaptive_att_weight + 1e-7)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training
    )

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    value_sum = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(value_sum, out_proj_weight, out_proj_bias)

    ret = []
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        ret = [attn_output, attn_output_weights.sum(dim=1) / num_heads]
    else:
        ret = [attn_output, None]

    if need_v_sum:
        ret.append(value_sum)
    else:
        ret.append(None)
    return ret


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    # TODO: move to layers
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
