# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument,mixup_one_target
import torch.nn.functional as F
import torch.distributed as dist

Projection_dim = 16

class Direct(nn.Module):
    def __init__(self):
        super(Direct,self).__init__()
        
    def forward(self, x):
        return x

class ABCNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.projection = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.backbone.num_features,Projection_dim),
        )
        # self.projection = Direct()
        
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@IMB_ALGORITHMS.register('bacon')
class BaCon(ImbAlgorithmBase):
    """
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - abc_p_cutoff (`float`):
                threshold for the auxilariy classifier
            - abc_loss_ratio (`float`):
                loss ration for auxiliary classifier
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(abc_p_cutoff=args.abc_p_cutoff, abc_loss_ratio=args.abc_loss_ratio)

        super(BaCon, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        
        # TODO: better ways
        self.base_model = self.model
        self.model = ABCNet(self.model, num_classes=self.num_classes)
        self.ema_model = ABCNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

        # placeholder
        self.history_preds = None
        self.datapoint_bank = None

        num_ulb = len(self.dataset_dict['train_ulb'])
        self.uncertainty_selected = torch.zeros(num_ulb)
        self.uncertainty_ema_map = torch.zeros(num_ulb, args.num_classes)
        self.uncertainty_ema_step = 1.0

        self.ulb_dest_len = args.ulb_dest_len
        self.lb_dest_len  = args.lb_dest_len
        self.selected_label = torch.ones((self.lb_dest_len+self.ulb_dest_len,), dtype=torch.long, ) * -1 
        self.selected_label = self.selected_label.to('cuda')
        self.cls_freq = torch.ones((args.num_classes,)).to('cuda')
        self.feat_list = torch.ones((self.lb_dest_len+self.ulb_dest_len,Projection_dim)).to('cuda')
        self.class_feat_center = torch.ones((self.num_classes,Projection_dim)).to('cuda')

    def imb_init(self, abc_p_cutoff=0.95, abc_loss_ratio=1.0):
        self.abc_p_cutoff = abc_p_cutoff
        self.abc_loss_ratio = abc_loss_ratio

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    ## trainstep for fixmatch
    def train_step(self, *args, **kwargs):
        x_lb, y_lb, x_ulb_w, x_ulb_s, idx_lb, idx_ulb = super().train_step(*args, **kwargs)

        num_lb = y_lb.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.base_model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.base_model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.base_model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.base_model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w.softmax(-1).clone().detach(), softmax_x_ulb=False)
        # generate unlabeled targets using pseudo label hook
        pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=logits_x_ulb_w.softmax(-1).clone().detach(),
                                        use_hard_label=self.use_hard_label,
                                        T=self.T,
                                        softmax=False)
        unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask)

        # parform abc-head calculation and do chunk
        feats = torch.cat((feats_x_lb,feats_x_ulb_w,feats_x_ulb_s))
        abc_out = self.model.module.aux_classifier(feats)
        abc_logits_x_lb = abc_out[:num_lb]
        abc_logits_x_ulb_w, abc_logits_x_ulb_s = abc_out[num_lb:].chunk(2)

        # update class count
        abc_max_probs, abc_max_idx = torch.max(abc_logits_x_ulb_w,dim=-1)
        select = abc_max_probs.ge(0.95)
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[self.lb_dest_len+idx_ulb[select == 1]] = abc_max_idx[select == 1]
            self.selected_label[idx_lb] = y_lb
        for i in range(self.num_classes):
            self.cls_freq[i] = torch.sum(self.selected_label==i)
            
        with torch.no_grad():
            ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)

        # compute abc loss using logits_aux from dict
        abc_loss, mask_ulb = self.compute_abc_loss(
            logits_x_lb=abc_logits_x_lb,
            y_lb=y_lb,
            logits_x_ulb_w=abc_logits_x_ulb_w,
            logits_x_ulb_s=abc_logits_x_ulb_s
            )

        select_lb = (torch.max(abc_logits_x_lb.softmax(-1),dim=-1)[0]).ge(0.98)
        select_ulb = (torch.max(abc_logits_x_ulb_w.softmax(-1),dim=-1)[0]).ge(0.98)
        select_all = torch.cat((select_lb,select_ulb),dim=0)

        feats_contra = self.model.module.projection(feats)
        # feats_contra = F.normalize(feats_contra,dim=1)
        proj_lb = feats_contra[:num_lb]
        proj_ulb_w, proj_ulb_s = feats_contra[num_lb:].chunk(2)

        contra_loss = torch.tensor(0).to(abc_loss.device)
        if self.it>150000:
            y_ulb = torch.max(abc_logits_x_ulb_w.softmax(-1),dim=-1)[1]
            contra_loss = self.contrastive_loss(
                anchors = self.class_feat_center,
                feats = torch.cat((proj_lb,proj_ulb_w),dim=0),
                y_lb = y_lb,
                top_ulb = abc_logits_x_ulb_w.topk(3,dim=-1)[1],
                select = select_all,
            )
            contra_loss = contra_loss * 1
        # if self.it % 100 == 0:
        #     self.print_fn('[fsj_debug] contra loss set as 0')

        total_loss = self.abc_loss_ratio * abc_loss + contra_loss + sup_loss + unsup_loss
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(abc_loss=abc_loss.item(), 
                                        contra_loss=contra_loss.item(),
                                        sup_loss=sup_loss.item(),
                                        unsup_loss=unsup_loss.item(),
                                        # kl_loss=kl_loss.item(),
                                        total_loss=total_loss.item(), 
                                        util_ratio=mask_ulb.float().mean().item(),
                                        select_for_contra=select_all.sum().item())
        
        # update feature space
        self.feat_list[idx_lb[select_lb==1]] = proj_lb[select_lb==1].clone().detach()
        self.feat_list[(idx_ulb+self.lb_dest_len)[select_ulb==1]] = proj_ulb_w[select_ulb==1].clone().detach()
        for i in range(self.num_classes):
            self.class_feat_center[i] = torch.mean(self.feat_list[self.selected_label==i],0)
            # self.class_feat_center[i] = torch.mean(self.feat_list[self.select_all==i],0)

        return out_dict, log_dict

    ## trainstep for remixmatch
    # def train_step(self, *args, **kwargs):
    #     x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, idx_lb, idx_ulb, x_ulb_s_0_rot, rot_v = super().train_step(*args, **kwargs)

    #     num_lb = y_lb.shape[0]
    #     # inference and calculate sup/unsup losses
    #     with self.amp_cm():
    #         with torch.no_grad():
    #             self.bn_controller.freeze_bn(self.model)

    #             outs_x_ulb_w = self.model(x_ulb_w)
    #             logits_x_ulb_w = outs_x_ulb_w['logits']
    #             feats_x_ulb_w = outs_x_ulb_w['feat']
    #             self.bn_controller.unfreeze_bn(self.model)

    #             prob_x_ulb = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=self.compute_prob(logits_x_ulb_w))
    #             sharpen_prob_x_ulb = prob_x_ulb ** (1 / self.T)
    #             sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

    #         self.bn_controller.freeze_bn(self.model)
    #         outs_x_lb = self.model(x_lb)
    #         outs_x_ulb_s_0 = self.model(x_ulb_s_0)
    #         outs_x_ulb_s_1 = self.model(x_ulb_s_1)
    #         self.bn_controller.unfreeze_bn(self.model)

    #         feat_dict = {'x_lb':outs_x_lb['feat'], 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':[outs_x_ulb_s_0['feat'], outs_x_ulb_s_1['feat']]}

    #         # mix up
    #         # with torch.no_grad():
    #         input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
    #         if self.mixup_manifold:
    #             inputs = torch.cat([outs_x_lb['feat'], outs_x_ulb_s_0['feat'], outs_x_ulb_s_1['feat'],  outs_x_ulb_w['feat']], dim=0)
    #         else:
    #             inputs = torch.cat([x_lb, x_ulb_s_0, x_ulb_s_1, x_ulb_w])
    #         mixed_x, mixed_y, _ = mixup_one_target(inputs, input_labels, self.mixup_alpha, is_bias=True)
    #         mixed_x = list(torch.split(mixed_x, num_lb))

    #         # calculate BN only for the first batch
    #         if self.mixup_manifold:
    #             logits = [self.model(mixed_x[0], only_fc=self.mixup_manifold)]
    #             # calculate BN for only the first batch
    #             self.bn_controller.freeze_bn(self.model)
    #             for ipt in mixed_x[1:]:
    #                 logits.append(self.model(ipt, only_fc=self.mixup_manifold))
    #             self.bn_controller.unfreeze_bn(self.model)
    #         else:
    #             logits = [self.model(mixed_x[0])['logits']]
    #             # calculate BN for only the first batch
    #             self.bn_controller.freeze_bn(self.model)
    #             for ipt in mixed_x[1:]:
    #                 logits.append(self.model(ipt)['logits'])
    #             self.bn_controller.unfreeze_bn(self.model)
    #         u1_logits = outs_x_ulb_s_0['logits']

    #         # put interleaved samples back
    #         # logits = interleave(logits, num_lb)
    #         logits_x = logits[0]
    #         logits_u = torch.cat(logits[1:], dim=0)

    #         # sup loss
    #         sup_loss = self.ce_loss(logits_x, mixed_y[:num_lb], reduction='mean')
    #         # unsup_loss
    #         unsup_loss = self.consistency_loss(logits_u, mixed_y[num_lb:])
            
    #         # loss U1
    #         u1_loss = self.consistency_loss(u1_logits, sharpen_prob_x_ulb)

    #         # ramp for w_match
    #         unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
    #         total_loss = sup_loss + self.lambda_kl * unsup_warmup * u1_loss + self.lambda_u * unsup_warmup * unsup_loss

    #         # calculate rot loss with w_rot
    #         if self.use_rot:
    #             self.bn_controller.freeze_bn(self.model)
    #             logits_rot = self.model(x_ulb_s_0_rot, use_rot=True)['logits_rot']
    #             self.bn_controller.unfreeze_bn(self.model)
    #             rot_loss = F.cross_entropy(logits_rot, rot_v, reduction='mean')
    #             rot_loss = rot_loss.mean()
    #             total_loss += self.lambda_rot * rot_loss

    #     out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
    #     log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
    #                                      unsup_loss=unsup_loss.item(), 
    #                                      total_loss=total_loss.item())

    #     # get features
    #     num_lb = y_lb.shape[0]
    #     feats_x_lb = out_dict['feat']['x_lb']
    #     feats_x_ulb_w = out_dict['feat']['x_ulb_w']
    #     feats_x_ulb_s = out_dict['feat']['x_ulb_s']
    #     if isinstance(feats_x_ulb_s, list):
    #         feats_x_ulb_s = feats_x_ulb_s[0]
        
    #     # parform abc-head calculation and do chunk
    #     feats = torch.cat((feats_x_lb,feats_x_ulb_w,feats_x_ulb_s))
    #     abc_out = self.model.aux_classifier(feats)
    #     abc_logits_x_lb = abc_out[:num_lb]
    #     abc_logits_x_ulb_w, abc_logits_x_ulb_s = abc_out[num_lb:].chunk(2)

    #     # update class count
    #     abc_max_probs, abc_max_idx = torch.max(abc_logits_x_ulb_w,dim=-1)
    #     select = abc_max_probs.ge(0.95)
    #     if idx_ulb[select == 1].nelement() != 0:
    #         self.selected_label[self.lb_dest_len+idx_ulb[select == 1]] = abc_max_idx[select == 1]
    #         self.selected_label[idx_lb] = y_lb
    #     for i in range(self.num_classes):
    #         self.cls_freq[i] = torch.sum(self.selected_label==i)
            
    #     with torch.no_grad():
    #         ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)

    #     # compute abc loss using logits_aux from dict
    #     abc_loss, mask_ulb = self.compute_abc_loss(
    #         logits_x_lb=abc_logits_x_lb,
    #         y_lb=y_lb,
    #         logits_x_ulb_w=abc_logits_x_ulb_w,
    #         logits_x_ulb_s=abc_logits_x_ulb_s
    #         )

    #     select_lb = (torch.max(abc_logits_x_lb.softmax(-1),dim=-1)[0]).ge(0.98)
    #     select_ulb = (torch.max(abc_logits_x_ulb_w.softmax(-1),dim=-1)[0]).ge(0.98)
    #     select_all = torch.cat((select_lb,select_ulb),dim=0)

    #     feats_contra = self.model.projection(feats)
    #     # feats_contra = F.normalize(feats_contra,dim=1)
    #     proj_lb = feats_contra[:num_lb]
    #     proj_ulb_w, proj_ulb_s = feats_contra[num_lb:].chunk(2)

    #     contra_loss = torch.tensor(0).to(abc_loss.device)
    #     if self.it>10000:
    #         y_ulb = torch.max(abc_logits_x_ulb_w.softmax(-1),dim=-1)[1]
    #         contra_loss = self.contrastive_loss(
    #             anchors = self.class_feat_center,
    #             feats = torch.cat((proj_lb,proj_ulb_w),dim=0),
    #             y_lb = y_lb,
    #             top_ulb = abc_logits_x_ulb_w.topk(3,dim=-1)[1],
    #             select = select_all,
    #         )
    #         contra_loss = contra_loss * 1

    #     out_dict['loss'] += (self.abc_loss_ratio * abc_loss + contra_loss)
    #     log_dict['train/abc_loss'] = abc_loss.item()
    #     log_dict['train/contra_loss'] = contra_loss.item()
    #     log_dict['train/select_all'] = select_all.sum().item()
        
    #     # update feature space
    #     self.feat_list[idx_lb[select_lb==1]] = proj_lb[select_lb==1].clone().detach()
    #     self.feat_list[(idx_ulb+self.lb_dest_len)[select_ulb==1]] = proj_ulb_w[select_ulb==1].clone().detach()
    #     for i in range(self.num_classes):
    #         self.class_feat_center[i] = torch.mean(self.feat_list[self.selected_label==i],0)

    #     return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)

    def contrastive_loss(self, anchors, feats, y_lb, top_ulb, select):
        # print(0.1+0.01*torch.sqrt(self.cls_freq/torch.max(self.cls_freq)))
        contra_loss = 0
        y = torch.cat((y_lb,top_ulb[:,0]),dim=0)
        for i in range(self.num_classes):
            temp = top_ulb - i
            idx = torch.nonzero(temp==0)[:,0]
            neg_idx = torch.ones((top_ulb.shape[0],)).to(y_lb.device)
            neg_idx[idx] = 0
            neg_idx = torch.cat((y_lb[:]!=i,neg_idx),dim=0).to(torch.long)
            neg_samples = feats[neg_idx==1]
            pos = torch.exp(torch.cosine_similarity(feats[y==i],anchors[y][y==i],dim=-1)/(0.1-(1-self.it/300000)**2*0.0005*torch.sqrt(self.cls_freq[i]/torch.max(self.cls_freq))))
            # pos = torch.exp(torch.cosine_similarity(feats[y==i],anchors[y][y==i],dim=-1)/0.1)
            neg = torch.exp(torch.cosine_similarity(feats[y==i].unsqueeze(1).repeat(1,neg_samples.shape[0],1),neg_samples.unsqueeze(0).repeat(feats[y==i].shape[0],1,1),dim=-1)/0.1)
            loss = pos/(pos+64*neg.mean()+1e-8)
            contra_loss += (-1 * torch.log(loss) * select[y==i]).sum()
        return contra_loss/(select.sum()+1e-8)
    
    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()
    
    def compute_abc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        abc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none')*mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(self.abc_p_cutoff).to(logits_x_ulb_w.dtype)
            ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)
            mask_ulb_2 = self.bernouli_mask(ulb_class_dist[y_ulb])
            mask_ulb = mask_ulb_1 * mask_ulb_2
    
        abc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            abc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
        
        abc_loss = abc_lb_loss + abc_ulb_loss
        return abc_loss, mask_ulb


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--abc_p_cutoff', float, 0.95),
            SSL_Argument('--abc_loss_ratio', float, 1.0),
        ]        
