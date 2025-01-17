from turtle import towards
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
import torch


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class Predictor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        # self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        feat_logits = self.backbone(input_dict)
        # print(feat_logits.shape)
        # print(input_dict["normal"].shape)

        pred, target = feat_logits, input_dict["normal"]

        l1_loss = torch.mean(torch.abs(target - pred))
        cos_loss = -torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        # print("l1",l1_loss.shape)
        # print("cos",cos_loss.shape)

        loss = l1_loss
        # print(loss)
        # print(loss.shape)
        if self.training:
            return dict(loss=loss)
        else:
            return dict(loss = loss, feat_logits=feat_logits, coord = input_dict["coord"], normal = input_dict["normal"])

        # # train
        # if self.training:
        #     loss = self.criteria(seg_logits, input_dict["normal"])
        #     return dict(loss=loss)
        # # eval
        # elif "segment" in input_dict.keys():
        #     loss = self.criteria(seg_logits, input_dict["normal"])
        #     return dict(loss=loss, seg_logits=seg_logits)
        # # test
        # else:
        #     return dict(seg_logits=seg_logits)

@MODELS.register_module()
class Predictor_6(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        # self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        feat_logits = self.backbone(input_dict)

        centroid = feat_logits[:,:3]
        towards = feat_logits[:,3:]

        centroid_gt = input_dict["normal"]
        towards_gt = input_dict["vector_attr_0"]


        # centroid = torch.nn.functional.normalize(centroid, dim=1)
        # centroid_gt = torch.nn.functional.normalize(centroid_gt, dim=1)
        
        # centroid_loss = torch.mean(torch.abs(centroid_gt - centroid)/(centroid_gt.norm(dim=1,keepdim=True) + 1e-6))
        centroid_loss = torch.mean(torch.abs(centroid_gt - centroid))
        # centroid_loss = centroid_loss / torch.mean(centroid_gt.norm(dim=1) + 1e-6)
        # towards[towards[:,2]<0] *= -1

        # v = torch.Tensor([1.0, 1.0, 1.0]).cuda()
        # dot_product = torch.matmul(towards, v)
        # cond = torch.sum(towards, dim=1)
        # towards[cond < 0] *=-1
        towards = towards/towards.norm(dim=1, keepdim=True)
        towards_loss = torch.mean(torch.abs(towards_gt - towards))
        towards_cos_loss = -torch.mean(torch.abs(torch.nn.functional.cosine_similarity(towards_gt, towards, dim=1)))

        # towards_loss = (-torch.mean(torch.nn.functional.cosine_similarity(towards_gt, towards, dim=-1)) + torch.mean(torch.abs(towards_gt - towards)))/2
        # loss = centroid_loss * 5 + towards_loss
        loss = 0.25*towards_cos_loss + 0.75*centroid_loss
        # loss = centroid_loss
        # loss = towards_cos_loss + centroid_loss

        if self.training:
            return dict(loss=loss)
        else:
            return dict(loss = loss, feat_logits=feat_logits, coord = input_dict["coord"], normal = input_dict["normal"])
        
@MODELS.register_module()
class Predictor_6_towards(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        # self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        feat_logits = self.backbone(input_dict)

        centroid = feat_logits[:,:3]
        towards = feat_logits[:,3:]

        centroid_gt = input_dict["normal"]
        towards_gt = input_dict["vector_attr_0"]


        # centroid = torch.nn.functional.normalize(centroid, dim=1)
        # centroid_gt = torch.nn.functional.normalize(centroid_gt, dim=1)
        
        # centroid_loss = torch.mean(torch.abs(centroid_gt - centroid)/(centroid_gt.norm(dim=1,keepdim=True) + 1e-6))
        centroid_loss = torch.mean(torch.abs(centroid_gt - centroid))
        centroid_loss = centroid_loss / torch.mean(centroid_gt.norm(dim=1) + 1e-6)
        # towards[towards[:,2]<0] *= -1

        # v = torch.Tensor([1.0, 1.0, 1.0]).cuda()
        # dot_product = torch.matmul(towards, v)
        cond = torch.sum(towards, dim=1)
        towards[cond < 0] *=-1
        towards_loss = torch.mean(torch.abs(towards_gt - towards))

        # towards_loss = (-torch.mean(torch.nn.functional.cosine_similarity(towards_gt, towards, dim=-1)) + torch.mean(torch.abs(towards_gt - towards)))/2
        loss = centroid_loss + towards_loss

        if self.training:
            return dict(loss=loss)
        else:
            return dict(loss = loss, feat_logits=feat_logits, coord = input_dict["coord"], normal = input_dict["normal"])

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
