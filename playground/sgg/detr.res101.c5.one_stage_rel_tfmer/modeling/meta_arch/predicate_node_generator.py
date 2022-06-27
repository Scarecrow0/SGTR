import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.modeling.backbone.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)

def build_decoder(
        cfg, num_decoder_layers,
):
    d_model = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL
    nhead = cfg.MODEL.REL_DETR.TRANSFORMER.N_HEAD
    dim_feedforward = cfg.MODEL.REL_DETR.TRANSFORMER.DIM_FFN
    dropout = cfg.MODEL.REL_DETR.TRANSFORMER.DROPOUT_RATE
    activation = cfg.MODEL.REL_DETR.TRANSFORMER.ACTIVATION
    normalize_before = cfg.MODEL.REL_DETR.TRANSFORMER.PRE_NORM
    return_intermediate_dec = cfg.MODEL.REL_DETR.TRANSFORMER.RETURN_INTERMEDIATE_DEC

    decoder_layer = TransformerDecoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    decoder_norm = nn.LayerNorm(d_model)
    decoder = TransformerDecoder(
        decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=False,
    )

    return decoder



class HiddenInstance:
    def __init__(self, feature=None, box=None, logits=None):

        self.feature = feature
        self.box = box
        self.logits = logits

        if isinstance(feature, list):
            if len(feature) > 0:
                self.feature = torch.stack(self.feature)
            else:
                self.feature = None
        if isinstance(box, list):
            if len(box) > 0:
                self.box = torch.stack(self.box)
            else:
                self.box = None
        if isinstance(logits, list):
            if len(logits) > 0:
                self.logits = torch.stack(self.logits)
            else:
                self.logits = None


class PredicateNodeGenerator(nn.Module):
    def __init__(self, cfg, input_shape=None):
        super(PredicateNodeGenerator, self).__init__()
        self.cfg = cfg

        d_model = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL
        nhead = cfg.MODEL.REL_DETR.TRANSFORMER.N_HEAD
        num_encoder_layers = cfg.MODEL.REL_DETR.TRANSFORMER.NUM_ENC_LAYERS
        num_predicates_decoder_layers = cfg.MODEL.REL_DETR.TRANSFORMER.NUM_DEC_LAYERS
        dim_feedforward = cfg.MODEL.REL_DETR.TRANSFORMER.DIM_FFN
        dropout = cfg.MODEL.REL_DETR.TRANSFORMER.DROPOUT_RATE
        activation = cfg.MODEL.REL_DETR.TRANSFORMER.ACTIVATION
        normalize_before = cfg.MODEL.REL_DETR.TRANSFORMER.PRE_NORM
        return_intermediate_dec = cfg.MODEL.REL_DETR.TRANSFORMER.RETURN_INTERMEDIATE_DEC

        self.d_model = d_model
        self.nhead = nhead

        # set as None means share the encoder with the entities detr
        self.encoder = None
        if num_encoder_layers is not None:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, normalize_before
            )
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        self.predicate_decoder = None
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            self.num_decoder_layer = self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.NUM_FUSE_LAYER
        else:
            self.num_decoder_layer = num_predicates_decoder_layers

        if num_predicates_decoder_layers > 0:
            self.predicate_decoder = build_decoder(cfg, num_predicates_decoder_layers)
            self.predicate_sub_decoder_layers = self.predicate_decoder.layers
            self.rel_decoder_norm = nn.LayerNorm(d_model)

        self.queries_cache = {}
        self.num_rel_queries = cfg.MODEL.REL_DETR.NUM_QUERIES

        # Predicate Query
        self.rel_query_embed = nn.Embedding(self.num_rel_queries, d_model)
        self.dynamic_query_on = cfg.MODEL.REL_DETR.DYNAMIC_QUERY
        if self.dynamic_query_on:
            self.coord_points_embed = nn.Sequential(nn.Linear(4, self.d_model))
            num_classes = cfg.MODEL.DETR.NUM_CLASSES + 1
            self.logits_embed = nn.Linear(num_classes, d_model)

            self.rel_q_gen = build_decoder(cfg, 1)

            self.ent_pos_sine_proj = nn.Linear(d_model, d_model)
            self.split_query = nn.Sequential(
                nn.ReLU(True), nn.Linear(self.d_model, self.d_model * 3)
            )

        self.entities_aware_decoder = None
        self.ent_pred_fuse_lyrnorm = nn.LayerNorm(d_model)
        self.scale_factor = None  # for pooler do roi pooling with image size
        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:

            self.rel_query_embed_sub = nn.Embedding(self.num_rel_queries, d_model)
            self.rel_query_embed_obj = nn.Embedding(self.num_rel_queries, d_model)

            if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.CROSS_DECODER:
                self.ent_decoder_init(cfg, d_model)

            self.ent_rel_fuse_fc_obj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model)
            )
            self.ent_rel_fuse_fc_sub = copy.deepcopy(self.ent_rel_fuse_fc_obj)


    def ent_decoder_init(self, cfg, d_model):
        num_decoder_layers = cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.NUM_DEC_LAYERS
        if (
                cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INTERACTIVE_REL_DECODER.ENT_DEC_EACH_LVL
        ):
            num_decoder_layers = (
                self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.NUM_FUSE_LAYER
            )

        self.entities_aware_decoder = build_decoder(cfg, num_decoder_layers)

        self.rel_ent_crs_decoder_layers_obj = self.entities_aware_decoder.layers
        self.rel_ent_crs_decoder_layers_sub = copy.deepcopy(
            self.entities_aware_decoder.layers
        )
        self.rel_ent_crs_decoder_norm = nn.LayerNorm(d_model)

        self.update_query_by_rel_hs = (
            cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INTERACTIVE_REL_DECODER.UPDATE_QUERY_BY_REL_HS
        )
        self.ent_dec_each_lvl = (
            cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INTERACTIVE_REL_DECODER.ENT_DEC_EACH_LVL
        )


    def debug_print(self, info):
        if self.cfg.DEBUG:
            print(info)

    def dynamic_predicate_query_generation(self, query_embed, ent_hs, ent_coords, rel_q_gen=None):
        ent_coords_embd = self.ent_pos_sine_proj(
            gen_sineembed_for_position(ent_coords[..., :2])
        ).contiguous()
        ent_hs = ent_hs[-1].transpose(1, 0)
        ent_coords_embd = ent_coords_embd.transpose(1, 0)
        if rel_q_gen is None:
            rel_q_gen = self.rel_q_gen
        query_embed = rel_q_gen(query_embed, ent_hs + ent_coords_embd)
        query_embed = self.split_query(query_embed)

        return query_embed  # seq_len, bz, dim

    def set_box_scale_factor(self, scale_factor):
        "resize the box into the image relavent size"
        self.scale_factor = scale_factor



    def forward(
            self,
            src,
            mask,
            query_embed,
            src_pos_embed,
            rel_query_pos_embed,
            shared_encoder_memory=None,
            ent_hs=None,
            ent_coords=None,
            ent_cls=None,
    ):

        """

        Args:
            src: the backbone features
            mask: mask for backbone features sequence
            query_embed: relationship prediction query embedding (N_q, dim)
            src_pos_embed: position_embedding for the src backbone features (W*H, dim)
            query_pos_embed: position_embedding for relationship prediction query (N_q, dim)
            shared_encoder_memory: the output of entities encoder (bz, dim, W*H )
            ent_hs: entities transformer outputs (lys, bz, num_q, dim)

        Returns:

        """

        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1)

        mask_flatten = mask.flatten(1)
        device = query_embed.device
        assert shared_encoder_memory is not None

        # decoder input
        if "transformer.decoder" in self.cfg.MODEL.WEIGHTS_FIXED:
            shared_encoder_memory = shared_encoder_memory.detach()
            ent_hs = ent_hs.detach()

        bs, h, w, rel_memory = self.rel_encoder(
            src, src_pos_embed, shared_encoder_memory, mask_flatten
        )

        # initialize the rel mem features HWxNxC
        # foreground mask generation
        ent_hs_input = ent_hs[-1].permute(1, 0, 2)  # seq_len, bz, dim
        enc_featmap = rel_memory.permute(1, 2, 0).reshape(bs, -1, h, w)

        # initialize the triplets query
        (
            query_embed_obj_init,
            query_embed_sub_init,
            query_embed_rel_init,
        ) = self.query_tgt_initialization(ent_hs, ent_coords)

        (rel_tgt, ent_obj_tgt, ent_sub_tgt) = self.reset_tgt()

        # outputs placeholder & container
        intermediate = []
        inter_rel_hs = []
        inter_value_sum = []
        inter_att_weight = []

        ent_obj_box = None
        ent_sub_box = None

        decoder_out_dict = {}
        ent_sub_dec_outputs = {}
        predicate_sub_dec_output_dict = None

        # decoder layer indexing
        num_ent_aware_layer = self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.NUM_FUSE_LAYER
        start = self.num_decoder_layer - num_ent_aware_layer
        end = self.num_decoder_layer
        predicate_box = None
        for idx in range(self.num_decoder_layer):

            output_dict = {}

            # predicates sub-decoder
            rel_hs_out = None
            if self.predicate_decoder is not None:

                predicate_sub_dec_output_dict = self.predicate_sub_decoder_layers[idx](
                    rel_tgt,
                    rel_memory,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=mask_flatten,
                    pos=src_pos_embed,
                    query_pos=query_embed_rel_init,
                    adaptive_att_weight=None,
                    return_value_sum=True,
                )
                rel_tgt = self.rel_decoder_norm(predicate_sub_dec_output_dict["tgt"])

                if predicate_sub_dec_output_dict.get("att_weight") is not None:
                    bs, num_rel_q, _ = predicate_sub_dec_output_dict["att_weight"].shape
                    predicate_sub_dec_output_dict["att_weight"] = predicate_sub_dec_output_dict[
                        "att_weight"
                    ].reshape(bs, num_rel_q, h, w)
                    inter_att_weight.append(predicate_sub_dec_output_dict["att_weight"])

                if predicate_sub_dec_output_dict.get("value_sum") is not None:
                    inter_value_sum.append(predicate_sub_dec_output_dict["value_sum"])

                inter_rel_hs.append(rel_tgt)
                output_dict["rel_hs"] = rel_tgt

                rel_hs_out = inter_rel_hs[-1]

            if self.entities_aware_decoder is not None and idx >= start:
                # entity indicator sub-decoder
                ent_sub_dec_outputs = self.entities_sub_decoder(
                    ent_hs_input, query_embed_obj_init, query_embed_sub_init,
                    ent_obj_tgt, ent_sub_tgt, start, idx, rel_hs_out,
                )
                ent_obj_tgt = ent_sub_dec_outputs['obj_ent_hs']
                ent_sub_tgt = ent_sub_dec_outputs['sub_ent_hs']
                rel_tgt = ent_sub_dec_outputs["ent_aug_rel_hs"]

                output_dict.update(ent_sub_dec_outputs)
                # only return needed intermediate hs
                intermediate.append(output_dict)

        if predicate_sub_dec_output_dict is not None:
            if predicate_sub_dec_output_dict.get('att_weight') is not None:
                decoder_out_dict["rel_attention"] = predicate_sub_dec_output_dict["att_weight"]

        if ent_sub_dec_outputs is not None:
            if ent_sub_dec_outputs.get('ent_sub_output_dict') is not None:
                if ent_sub_dec_outputs['ent_sub_output_dict'].get('att_weight') is not None:
                    decoder_out_dict.update({
                        "ent_sub_attention": ent_sub_dec_outputs['ent_sub_output_dict']["att_weight"],
                        "ent_obj_attention": ent_sub_dec_outputs['ent_obj_output_dict']["att_weight"],
                    })

        rel_feat_all = []
        ent_aware_rel_hs_sub = []
        ent_aware_rel_hs_obj = []

        for outs in intermediate:
            if "ent_aug_rel_hs" in outs.keys():
                rel_feat_all.append(outs["ent_aug_rel_hs"])
            elif "rel_hs" in outs.keys():
                rel_feat_all.append(outs["rel_hs"])

            if "obj_ent_hs" in outs.keys():
                ent_aware_rel_hs_sub.append(outs["sub_ent_hs"])  # layer x [Nq, bz dim]
                ent_aware_rel_hs_obj.append(outs["obj_ent_hs"])

        assert len(rel_feat_all) > 0

        rel_rep = HiddenInstance(feature=rel_feat_all)
        rel_rep.feature.transpose_(1, 2)

        sub_ent_rep = HiddenInstance(feature=ent_aware_rel_hs_sub)
        obj_ent_rep = HiddenInstance(feature=ent_aware_rel_hs_obj)

        sub_ent_rep.feature.transpose_(1, 2)
        obj_ent_rep.feature.transpose_(1, 2)

        dynamic_query = None
        if self.queries_cache.get("dynamic_query") is not None:
            dynamic_query = self.queries_cache.get("dynamic_query")
            dynamic_query = HiddenInstance(feature=dynamic_query)
            dynamic_query.feature.transpose_(1, 2)

        return (
            rel_rep,
            (sub_ent_rep, obj_ent_rep, dynamic_query),
            decoder_out_dict,
        )

    def entities_sub_decoder(
            self,
            ent_hs_input,
            query_pos_embed_obj,
            query_pos_embed_sub,
            ent_obj_tgt,
            ent_sub_tgt,
            start,
            idx,
            rel_hs_out,
    ):
        rel_hs_out_obj_hs = []
        rel_hs_out_sub_hs = []

        if self.update_query_by_rel_hs and rel_hs_out is not None:
            ent_sub_tgt = ent_sub_tgt + rel_hs_out
            ent_obj_tgt = ent_obj_tgt + rel_hs_out


        _sub_ent_dec_layers = self.rel_ent_crs_decoder_layers_sub
        _obj_ent_dec_layers = self.rel_ent_crs_decoder_layers_obj

        if self.ent_dec_each_lvl:
            _sub_ent_dec_layers = [self.rel_ent_crs_decoder_layers_sub[idx - start]]
            _obj_ent_dec_layers = [self.rel_ent_crs_decoder_layers_obj[idx - start]]

        for layeri, (ent_dec_layer_sub, ent_dec_layer_obj) in enumerate(zip(
                _sub_ent_dec_layers, _obj_ent_dec_layers,
        )):
            # seq_len, bs, dim = rel_hs_out.shape
            # self.debug_print('ent_dec_layers id' + str(layeri))

            ent_sub_output_dict = ent_dec_layer_sub(
                ent_sub_tgt,
                ent_hs_input,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=query_pos_embed_sub,
                adaptive_att_weight=None,
                return_value_sum=True,
            )

            ent_obj_output_dict = ent_dec_layer_obj(
                ent_obj_tgt,
                ent_hs_input,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=query_pos_embed_obj,
                adaptive_att_weight=None,
                return_value_sum=True,
            )

            ent_obj_tgt = ent_obj_output_dict["tgt"]
            ent_sub_tgt = ent_sub_output_dict["tgt"]

            rel_hs_sub = self.rel_ent_crs_decoder_norm(ent_sub_tgt)
            rel_hs_obj = self.rel_ent_crs_decoder_norm(ent_obj_tgt)

            rel_hs_out_obj_hs.append(rel_hs_obj)
            rel_hs_out_sub_hs.append(rel_hs_sub)

        ent_sub_tgt = rel_hs_out_sub_hs[-1]
        ent_obj_tgt = rel_hs_out_obj_hs[-1]

        # merge the final representation for prediction
        ent_aug_rel_hs_out = (F.relu(self.ent_rel_fuse_fc_sub(ent_sub_tgt)
                                     + self.ent_rel_fuse_fc_obj(ent_obj_tgt)))
        if rel_hs_out is not None:
            ent_aug_rel_hs_out = rel_hs_out + ent_aug_rel_hs_out


        return {
            "ent_aug_rel_hs": ent_aug_rel_hs_out,
            "sub_ent_hs": ent_sub_tgt,
            "obj_ent_hs": ent_obj_tgt,
            "ent_sub_output_dict": ent_sub_output_dict,
            "ent_obj_output_dict": ent_obj_output_dict,
        }


    def query_tgt_initialization(self, ent_hs, ent_coords):
        """
        apply the dynamic query into the
        """
        # static query weights (N_q, dim) -> (N_q, bz, dim)
        self.queries_cache = {}
        bs = ent_hs.shape[1]
        query_embed_obj_init_w = self.rel_query_embed_obj.weight.unsqueeze(1).repeat(
            1, bs, 1
        )
        query_embed_sub_init_w = self.rel_query_embed_sub.weight.unsqueeze(1).repeat(
            1, bs, 1
        )
        query_embed_rel_init_w = self.rel_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # query pos embedding
        query_embed_sub_init = query_embed_sub_init_w
        query_embed_obj_init = query_embed_obj_init_w
        query_embed_rel_init = query_embed_rel_init_w

        self.queries_cache.update(
            {
                "query_embed_obj_init_w": query_embed_obj_init_w,
                "query_embed_sub_init_w": query_embed_sub_init_w,
                "query_embed_rel_init_w": query_embed_rel_init_w,
            }
        )

        if self.dynamic_query_on:
            dynamic_query = self.dynamic_predicate_query_generation(
                query_embed_rel_init_w, ent_hs, ent_coords
            )
            seq_len, bs, dim = dynamic_query.shape  # seq_len, bz, dim
            dynamic_query = dynamic_query.reshape(seq_len, bs, 3, dim // 3).transpose(
                3, 2
            )
            d_query_embed_sub_input_init = dynamic_query[..., 0,]
            d_query_embed_obj_input_init = dynamic_query[..., 1,]
            d_query_embed_rel_input_init = dynamic_query[..., 2,]
            dynamic_query = dynamic_query.permute(3, 0, 1, 2)

            self.queries_cache.update(
                {
                    "dynamic_query": dynamic_query,
                    "d_query_embed_sub_input_init": d_query_embed_sub_input_init,
                    "d_query_embed_obj_input_init": d_query_embed_obj_input_init,
                    "d_query_embed_rel_input_init": d_query_embed_rel_input_init,
                }
            )

        self.queries_cache.update(
            {
                "query_embed_obj_init": query_embed_obj_init,
                "query_embed_sub_init": query_embed_sub_init,
                "query_embed_rel_init": query_embed_rel_init,
            }
        )

        return query_embed_obj_init, query_embed_sub_init, query_embed_rel_init

    def reset_tgt(self):
        # keys & tgt:
        #   initialization by the dynamic query
        if self.dynamic_query_on:
            d_query_embed_sub_input_init = self.queries_cache[
                "d_query_embed_sub_input_init"
            ]
            d_query_embed_obj_input_init = self.queries_cache[
                "d_query_embed_obj_input_init"
            ]
            d_query_embed_rel_input_init = self.queries_cache[
                "d_query_embed_rel_input_init"
            ]

            ent_sub_tgt = d_query_embed_sub_input_init.clone()
            ent_obj_tgt = d_query_embed_obj_input_init.clone()
            rel_tgt = d_query_embed_rel_input_init.clone()
        else:
            query_embed_rel_init = self.queries_cache["query_embed_rel_init"]
            ent_sub_tgt = torch.zeros_like(query_embed_rel_init)
            ent_obj_tgt = torch.zeros_like(query_embed_rel_init)
            rel_tgt = torch.zeros_like(query_embed_rel_init)

        return rel_tgt, ent_obj_tgt, ent_sub_tgt


    def rel_encoder(self, src, src_pos_embed, shared_encoder_memory, mask_flatten):
        if self.encoder is not None:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = shared_encoder_memory.shape
            shared_encoder_memory = shared_encoder_memory.flatten(2).permute(2, 0, 1)
            rel_memory = self.encoder(
                shared_encoder_memory,
                src_key_padding_mask=mask_flatten,
                pos=src_pos_embed,
            )
        else:
            if len(src.shape) == 4 and len(shared_encoder_memory.shape):
                bs, c, h, w = src.shape
                # flatten NxCxHxW to HWxNxC for following decoder
                rel_memory = shared_encoder_memory.view(bs, c, h * w).permute(2, 0, 1)
            else:
                # not satisfy the reshape: directly use
                # must in shape (len, bz, dim)
                rel_memory = shared_encoder_memory
                bs, c, h_w = rel_memory.shape
        return bs, h, w, rel_memory



def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos
