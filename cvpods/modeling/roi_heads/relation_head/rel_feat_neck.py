import torch
from torch import nn

from cvpods.data.datasets.builtin_meta import get_dataset_statistics
from cvpods.structures.relationship import Relationships
from .utils_motifs import obj_edge_vectors, encode_box_info


class RelationshipFeatureNeck(nn.Module):
    def __init__(self, cfg, rel_feat_input_shape, ent_feat_input_shape):
        super(RelationshipFeatureNeck, self).__init__()
        self.cfg = cfg
        self.pairwise_rel_feat_extractor = PairwiseFeatureExtractor(
            cfg, rel_feat_input_shape, ent_feat_input_shape
        )

        self._output_size = self.pairwise_rel_feat_extractor.output_size

    @property
    def output_size(self):
        return self._output_size

    def forward(
            self, rel_proposals: Relationships, rel_features: list, ent_features: list
    ):
        """augment the relationship features with additional structure

        Args:
            rel_proposals (Relationships): [description]
            rel_features (list): [description]
            ent_features (list): [description]

        Returns:
            rel_features (list): [description]
            ent_features (list): [description]
            outputs (dict): additional intermidate output
        """

        #   pair-wise ent_features
        #   word embedding
        #   visual features
        rel_features, ent_features = self.pairwise_rel_feat_extractor(
            rel_proposals, rel_features, ent_features
        )

        #   pair wise feature attention(optional)

        outputs = None
        return rel_features, ent_features, outputs


class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, rel_feat_input_shape, ent_feat_input_shape):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        self.num_obj_classes = config.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_rel_classes = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        data_statistics = get_dataset_statistics(config)

        self.obj_classes = data_statistics["obj_classes"]
        self.rel_classes = data_statistics["rel_classes"]

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.ent_word_embed_dim = (
            self.cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE_REL_FEATURE.WORD_EMBEDDING_FEATURES_DIM
        )
        self.ent_input_dim = ent_feat_input_shape

        self.hidden_dim = (
            self.cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE_REL_FEATURE.HIDDEN_DIM
        )
        self.output_size = rel_feat_input_shape

        self.word_embed_feats_on = (
            self.cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE_REL_FEATURE.WORD_EMBEDDING_FEATURES
        )
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(
                self.obj_classes[1:],  # remove the background categories
                wv_dir=self.cfg.EXT_KNOWLEDGE.GLOVE_DIR,
                wv_dim=self.ent_word_embed_dim,
            )
            self.ent_pred_word_embed = nn.Embedding(
                self.num_obj_classes, self.ent_word_embed_dim
            )
            with torch.no_grad():
                self.ent_pred_word_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 256
        self.pos_embed = nn.Sequential(
                nn.Linear(9, 32),
                nn.BatchNorm1d(32, momentum=0.001),
                nn.Linear(32, self.geometry_feat_dim),
                nn.ReLU(inplace=True),
        )
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.ent4rel_hidden_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(
                self.ent_input_dim + self.ent_word_embed_dim + self.geometry_feat_dim,
                self.hidden_dim,
            ),
        )

        self.pairwise_obj_feat_updim_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        )

        self.pair_bbox_geo_embed = nn.Sequential(
            nn.Linear(32, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )

        self.ent_pair_fuse_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.union_fuse_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.output_size, self.hidden_dim)
        )

        self.output_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_size)
        )

    def forward(self, rel_proposals: list, rel_features: list, ent_features: list):
        """

        Args:
            rel_proposals: list(Relationships)
            rel_features: list(Tensor)
            ent_features: list(Tensor)

        Returns:

            rel_features: list(Tensor) the augmented relationship features

        """
        # using label or logits do the label space embeddings

        entit_proposals = [p.instances for p in rel_proposals]

        rel_batch_sizes = [len(p) for p in rel_proposals]
        ent_batch_sizes = [len(p) for p in entit_proposals]

        # box positive geometry embedding
        pos_embed = self.pos_embed(
            encode_box_info(
                [p.pred_boxes.tensor for p in entit_proposals],
                [p.image_size for p in entit_proposals],
            )
        )

        ent_features_cated = cat(ent_features)

        # entities word embedding
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            ent_pred_labels = torch.cat(
                [proposal.gt_classes for proposal in entit_proposals], dim=0
            )
        else:
            ent_pred_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                ent_word_embeding = self.ent_pred_word_embed(ent_pred_labels.long())
            else:
                ent_pred_dist = torch.cat(
                    [proposal.pred_score_dist for proposal in entit_proposals], dim=0,
                ).detach()

                ent_word_embeding = ent_pred_dist @ self.ent_pred_word_embed.weight

            ent_feats4pair = cat((ent_features_cated, ent_word_embeding, pos_embed), -1)
        else:
            ent_feats4pair = cat((ent_features_cated, pos_embed), -1)



        # mapping to hidden
        ent_feats4pair = self.ent4rel_hidden_fc(ent_feats4pair)

        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(ent_feats4pair)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.shape[0], 2, self.hidden_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)

        # split
        head_reps = head_rep.split(ent_batch_sizes, dim=0)
        tail_reps = tail_rep.split(ent_batch_sizes, dim=0)

        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        ent_pair_feat4rel = []
        pair_bboxs_info = []

        ent_pred_boxes = [p.pred_boxes.tensor for p in entit_proposals]
        rel_pair_idxs = [r.rel_pair_tensor for r in rel_proposals]

        for pair_idx, head_rep, tail_rep, obj_box in zip(
                rel_pair_idxs, head_reps, tail_reps, ent_pred_boxes
        ):
            ent_pair_feat4rel.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )

            pair_bboxs_info.append(
                get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]])
            )

        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        pair_bbox_geo_feats = self.pair_bbox_geo_embed(pair_bbox_geo_info)

        ent_pair_feat4rel = cat(ent_pair_feat4rel, dim=0)  # (num_rel, hidden_dim * 2)

        ent_pair_feat4rel = ent_pair_feat4rel * pair_bbox_geo_feats
        ent_pair_feat4rel = self.ent_pair_fuse_fc(ent_pair_feat4rel)  # (num_rel, hidden_dim * 2) -> (num_rel, out_dim)

        rel_features_fuse_union = self.union_fuse_fc(torch.cat(rel_features)) + ent_pair_feat4rel

        rel_features = torch.split(
            self.output_fc(rel_features_fuse_union),
            rel_batch_sizes
        )

        return rel_features, ent_features


def get_box_pair_info(box1, box2):
    """
    input:
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output:
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:, :4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)

    # intersection box
    intersextion_box = box1[:, :4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(
        intersextion_box[:, 2].contiguous().view(-1)
        < intersextion_box[:, 0].contiguous().view(-1)
    ).view(-1)
    case2 = torch.nonzero(
        intersextion_box[:, 3].contiguous().view(-1)
        < intersextion_box[:, 1].contiguous().view(-1)
    ).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch.cat(
        (
            get_box_info(box1, need_norm=False),
            get_box_info(box2, need_norm=False),
            union_info,
            intersextion_info,
        ),
        1,
    )


def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)
