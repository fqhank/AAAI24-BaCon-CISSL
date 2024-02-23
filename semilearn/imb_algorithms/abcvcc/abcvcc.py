# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
import torch.nn.functional as F
import torch.distributed as dist

class ABCNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
    
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

@IMB_ALGORITHMS.register('abcmcdrop')
class ABC(ImbAlgorithmBase):
    """
        ABC algorithm (https://arxiv.org/abs/2110.10368).

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

        super(ABC, self).__init__(args, net_builder, tb_log, logger, **kwargs)

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
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1 
        self.selected_label = self.selected_label.to('cuda')
        self.cls_freq = torch.zeros((10,)).to('cuda')

    def imb_init(self, abc_p_cutoff=0.95, abc_loss_ratio=1.0):
        self.abc_p_cutoff = abc_p_cutoff
        self.abc_loss_ratio = abc_loss_ratio

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    # def train_step(self, *args, **kwargs):
    #     out_dict, log_dict, x, idx_ulb = super().train_step(*args, **kwargs)

    #     # get features
    #     feats_x_lb = out_dict['feat']['x_lb']
    #     feats_x_ulb_w = out_dict['feat']['x_ulb_w']
    #     feats_x_ulb_s = out_dict['feat']['x_ulb_s']
    #     if isinstance(feats_x_ulb_s, list):
    #         feats_x_ulb_s = feats_x_ulb_s[0]

    #     num_lb = feats_x_lb.shape[0]
        
    #     # get logits
    #     # logits_x_lb = self.model.module.aux_classifier(feats_x_lb)
    #     # logits_x_ulb_s = self.model.module.aux_classifier(feats_x_ulb_s)
    #     with torch.no_grad():
    #         ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)
    #     # with torch.no_grad():
    #     #     logits_x_ulb_w = self.model.module.aux_classifier(feats_x_ulb_w)

    #     feats = torch.cat((feats_x_lb,feats_x_ulb_s,feats_x_ulb_w))
    #     abc_out = self.model.module.aux_classifier(feats)
    #     logits_x_lb = abc_out[:num_lb]
    #     logits_x_ulb_s, logits_x_ulb_w = abc_out[num_lb:].chunk(2)
    #     logits_x_ulb_w = logits_x_ulb_w.clone().detach()

    #     max_probs, max_idx = torch.max(logits_x_ulb_w,dim=-1)
    #     select = max_probs.ge(0.95)
    #     # update
    #     if idx_ulb[select == 1].nelement() != 0:
    #         self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
    #     for i in range(10):
    #         self.cls_freq[i] = torch.sum(self.selected_label==i)
    #     # print(self.cls_freq)

    #     cali_gt_label = self.calc_uncertainty(
    #             algorithm=self, x=x, ulb_x_idx=idx_ulb, feats=feats, logits=abc_out, cls_freq=self.cls_freq)

    #     _, recon_gt_ulb_w = cali_gt_label[num_lb:].chunk(2)
    #     recon_gt_ulb_w = self.update_uncertainty_map(idx_ulb, recon_gt_ulb_w)

    #     # compute abc loss using logits_aux from dict
    #     if self.epoch<20000:
    #         recon_gt_ulb_w = logits_x_ulb_w
    #     abc_loss = self.compute_abc_loss(
    #         logits_x_lb=logits_x_lb,
    #         y_lb=kwargs['y_lb'],
    #         # logits_x_ulb_w=logits_x_ulb_w,
    #         logits_x_ulb_w=recon_gt_ulb_w,
    #         logits_x_ulb_s=logits_x_ulb_s
    #         )
    #     out_dict['loss'] += self.abc_loss_ratio * abc_loss 
    #     log_dict['train/abc_loss'] = abc_loss.item()

    #     return out_dict, log_dict

    def train_step(self, *args, **kwargs):
        x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb = super().train_step(*args, **kwargs)

        print('vcc')
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

        # compute abc-head, and count
        feats = torch.cat((feats_x_lb,feats_x_ulb_s,feats_x_ulb_w))
        abc_out = self.model.module.aux_classifier(feats)
        abc_logits_x_lb = abc_out[:num_lb]
        abc_logits_x_ulb_s, abc_logits_x_ulb_w = abc_out[num_lb:].chunk(2)
        abc_logits_x_ulb_w = abc_logits_x_ulb_w.clone().detach()

        abc_max_probs, abc_max_idx = torch.max(torch.softmax(abc_logits_x_ulb_w,dim=-1),dim=-1)
        select = abc_max_probs.ge(0.95)
        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = abc_max_idx[select == 1]
        for i in range(self.num_classes):
            self.cls_freq[i] = torch.sum(self.selected_label==i)

        cali_gt_label = self.calc_uncertainty(
                algorithm=self, x=torch.cat((x_lb, x_ulb_s, x_ulb_w)), ulb_x_idx=idx_ulb, feats=feats, logits=abc_out, cls_freq=self.cls_freq)
        _, recon_gt_ulb_w = cali_gt_label[num_lb:].chunk(2)
        recon_gt_ulb_w = self.update_uncertainty_map(idx_ulb, recon_gt_ulb_w)

        probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach()) # probs for backbone head
        if self.it>20000:
            probs_x_ulb_w = recon_gt_ulb_w.clone().detach()
            abc_logits_x_ulb_w = recon_gt_ulb_w.clone().detach()
            
        # compute mask
        mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

        # generate unlabeled targets using pseudo label hook
        pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

        unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())

        with torch.no_grad():
            ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)

        # compute abc loss using logits_aux from dict
        abc_loss = self.compute_abc_loss(
            logits_x_lb=abc_logits_x_lb,
            y_lb=y_lb,
            logits_x_ulb_w=abc_logits_x_ulb_w,
            logits_x_ulb_s=abc_logits_x_ulb_s
            )
        out_dict['loss'] += self.abc_loss_ratio * abc_loss 
        log_dict['train/abc_loss'] = abc_loss.item()

        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)

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
        abc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

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
        return abc_loss


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--abc_p_cutoff', float, 0.95),
            SSL_Argument('--abc_loss_ratio', float, 1.0),
        ]        

    def update_uncertainty_map(self, idx_ulb, recon_gt_ulb_w):
        if dist.get_world_size() > 1:
            dist_idx_ulb = idx_ulb.new_zeros(self.uncertainty_selected.shape[0])
            dist_upd_val = recon_gt_ulb_w.new_zeros(self.uncertainty_selected.shape[0], recon_gt_ulb_w.shape[1])
            dist_idx_ulb[idx_ulb], dist_upd_val[idx_ulb] = 1, recon_gt_ulb_w
            dist.all_reduce(dist_idx_ulb, op=dist.ReduceOp.SUM)
            dist.all_reduce(dist_upd_val, op=dist.ReduceOp.SUM)
            dist.barrier()
            dist_upd_val = dist_upd_val / (dist_idx_ulb[..., None] + 1e-7)
            recon_gt_ulb_w = dist_upd_val[idx_ulb]
            dist.barrier()

        self.uncertainty_ema_map = self.uncertainty_ema_map.to(self.gpu)
        update_weight = torch.ones_like(recon_gt_ulb_w)
        update_weight[(self.uncertainty_selected[idx_ulb.cpu()] == 1)] = self.uncertainty_ema_step
        self.uncertainty_selected[idx_ulb] = 1
        updated_value = update_weight * recon_gt_ulb_w + (1 - update_weight) * self.uncertainty_ema_map[idx_ulb].cuda()
        self.uncertainty_ema_map[idx_ulb] = updated_value
        return updated_value

    def calc_uncertainty(self, **kwargs):
        kwargs['batch_size'] = kwargs['feats'].shape[0]
        kwargs['num_classes'] = self.num_classes
        # uncertainty_method = self.args.vcc_uncertainty_method
        uncertainty_method = 'consistency'
        uncertainty_method = getattr(self, f'calc_uncertainty_{uncertainty_method}')
        return uncertainty_method(**kwargs)[0]

    def calc_uncertainty_consistency(self, algorithm, x, ulb_x_idx, feats, logits, cls_freq, batch_size, num_classes):
        # assert self.args.vcc_lab_loss_weight == 0
        ulb_num, lb_num = ulb_x_idx.shape[0], batch_size - 2 * ulb_x_idx.shape[0]
        lb_x, (ulb_x_w, ulb_x_s) = x[:lb_num], x[lb_num:].chunk(2)
        total_ulb_num = len(algorithm.dataset_dict['train_ulb'])
        if self.datapoint_bank is None:
            self.datapoint_bank = [[] for _ in range(num_classes)]

        all_confidence = logits.softmax(1).detach()
        preds = all_confidence[lb_num:lb_num + ulb_num]

        # Temporal Consistency
        if self.history_preds is None:
            self.history_preds = ulb_x_s.new_ones((total_ulb_num, num_classes)) / num_classes
        self.history_preds = self.history_preds.to(ulb_x_s.device)

        prev_preds = self.history_preds[ulb_x_idx]
        # if (abs(self.history_preds.sum(1) - 1.0) > 1e-5).any():
        #     joblib.dump(self.history_preds.cpu().detach().numpy(), 'debug.pth')
        #     print('Save debug')
        # assert abs(prev_preds[0].sum() - 1.0) < 1e-5
        temporal_kl_div = torch.kl_div((preds + 1e-7).log(), prev_preds).sum(1)
        upd_preds = ulb_x_s.new_zeros((total_ulb_num, num_classes))
        upd_cnt = ulb_x_s.new_zeros((total_ulb_num,))
        upd_preds[ulb_x_idx], upd_cnt[ulb_x_idx] = preds, 1
        if algorithm.args.distributed:
            dist.all_reduce(upd_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(upd_cnt, op=dist.ReduceOp.SUM)
            dist.barrier()
        upd_mask = (upd_cnt != 0)
        upd_preds[upd_mask] /= upd_cnt[upd_mask][:, None]
        self.history_preds[upd_mask] = upd_preds[upd_mask]

        # Instance Consistency
        self.dropout_keep_p, self.sampling_times = 0.7, 8
        _, ic_logits = self.calc_uncertainty_mcdropout(feats[lb_num:lb_num + ulb_num], ulb_num, num_classes,cls_freq)
        ic_pred = ic_logits.softmax(1).reshape(self.sampling_times, ulb_num, num_classes)
        ic_pred = ic_pred.mean(0)
        entropy = -(ic_pred * (ic_pred + 1e-7).log()).sum(1)

        # View Consistency: EMA v.s. ori
        algorithm.ema.apply_shadow()
        ema_feats = self.model.module.backbone(ulb_x_w, only_feat=True)
        # ema_logits = self.base_net(feats[lb_num:lb_num + ulb_num], only_fc=True)
        ema_logits = self.model.module.aux_classifier(feats[lb_num:lb_num + ulb_num])
        ema_preds = ema_logits.softmax(1)
        algorithm.ema.restore()
        # ori_logits = self.base_net(ema_feats, only_fc=True)
        ori_logits = self.model.module.aux_classifier(ema_feats)
        ori_preds = ori_logits.softmax(1)
        view_kl_div = torch.kl_div((ori_preds + 1e-7).log(), ema_preds).sum(1)

        # Ori confidence
        confidence = all_confidence[lb_num:lb_num + ulb_num]

        if algorithm.args.distributed:
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1
        gmm_feats = torch.cat(
            [confidence.max(1)[0][None], temporal_kl_div[None], entropy[None], view_kl_div[None]], 0
        ).transpose(0, 1)
        pseudo_labels = logits[lb_num:lb_num + ulb_num].argmax(1)
        dist_gmm_feats = gmm_feats.new_zeros(ulb_num * world_size, gmm_feats.shape[1])
        dist_pseudo_labels = pseudo_labels.new_zeros(ulb_num * world_size)
        dist_gmm_feats[ulb_num * rank: ulb_num * (rank + 1)] = gmm_feats
        dist_pseudo_labels[ulb_num * rank: ulb_num * (rank + 1)] = pseudo_labels
        if algorithm.args.distributed:
            dist.all_reduce(dist_gmm_feats, op=dist.ReduceOp.SUM)
            dist.all_reduce(dist_pseudo_labels, op=dist.ReduceOp.SUM)
            dist.barrier()
        # datapoint_bank_size = algorithm.args.vcc_datapoint_bank_size
        datapoint_bank_size = 100
        for i, label in enumerate(dist_pseudo_labels):
            self.datapoint_bank[label].append(dist_gmm_feats[i].cpu().tolist())
            self.datapoint_bank[label] = self.datapoint_bank[label][-datapoint_bank_size:]

        cali_conf = all_confidence

        def compute_score(data, max_norm, min_norm):
            max_norm, min_norm = max_norm[None], min_norm[None]
            data = (data - min_norm) / (max_norm - min_norm + 1e-5)
            data[:, 0] = 1 - data[:, 0]
            return (data ** 2).sum(1)

        for label in set(pseudo_labels.tolist()):
            if len(self.datapoint_bank[label]) < 50:
                continue
            mask = (pseudo_labels == label)
            cls_data = np.array(self.datapoint_bank[label])
            max_norm, min_norm = cls_data.max(0), cls_data.min(0)
            max_conf, min_conf = max_norm[0], min_norm[0]
            cls_score = compute_score(cls_data, max_norm, min_norm)
            max_score, min_score = cls_score.max(), cls_score.min()
            batch_score = compute_score(gmm_feats[mask].cpu().detach().numpy(), max_norm, min_norm)
            batch_cali_conf = ((max_score - batch_score) / (max_score - min_score + 1e-7)
                               * (max_conf - min_conf) + min_conf)
            batch_cali_conf = cali_conf.new_tensor(batch_cali_conf)
            ori_confidence = confidence[mask]
            ori_others_conf = 1 - ori_confidence[:, label]
            cur_others_conf = 1 - batch_cali_conf
            cali_conf[lb_num:ulb_num+lb_num][mask] *= (
                    cur_others_conf / (ori_others_conf + 1e-7))[..., None]
            cali_conf[lb_num:ulb_num+lb_num][mask, label] = batch_cali_conf

        return cali_conf, None

    def calc_uncertainty_mcdropout(self, feats, batch_size, num_classes,cls_freq, **kwargs):
        feats = torch.cat([feats for _ in range(self.sampling_times)], 0)
        with torch.no_grad():
            feats = torch.dropout(feats, p=1 - self.dropout_keep_p, train=True)
            # pred = self.base_net.fc(feats)
            pred = self.model.module.aux_classifier(feats)
        # pred = pred.softmax(dim=-1)
        ### SaR refinement ###
        # beta = (1-0.999)/(1-0.999**(cls_freq.squeeze(0)+1))
        # pred = pred * beta.to(pred.device)
        # pred = pred / pred.sum(-1).unsqueeze(-1).repeat(1,num_classes)
        #####################
        result = pred.argmax(1)
        result = F.one_hot(result, num_classes)
        result = result.reshape(self.sampling_times, batch_size, num_classes)
        result = result.permute(1, 0, 2)
        result = result.sum(1).float() / self.sampling_times
        return result, pred