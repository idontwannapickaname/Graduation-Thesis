# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.train_loader_size = None
        self.iter = 0

    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        num_choose = 100
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []

            count = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                x = x.to(self.device)

                processX = self.net.vitProcess(x)

                features = self.net.local_vitmodel.patch_embed(processX)
                cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
                features = torch.cat((cls_token, features), dim=1)
                features = features + self.net.local_vitmodel.pos_embed

                # forward pass till -3
                for block in self.net.local_vitmodel.blocks[:-3]:
                    features = block(features)

                features = self.net.Forever_freezed_blocks(features)

                features = self.net.local_vitmodel.norm(features)

                class_token = features[:, 0, :]

                fc_features_list.append(self.net.fcArr[self.current_task](class_token))

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T)
            self.net.distributions.append(MultivariateNormal(mu, sigma))

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        self.net.Freezed_local_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.local_vitmodel.blocks[-3:])))
        self.net.Freezed_global_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.global_vitmodel.blocks[-3:])))

        for block in self.net.Freezed_local_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block in self.net.Freezed_global_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def begin_task(self, dataset, threshold=0) -> None:
        if not hasattr(dataset.train_loader.dataset, 'targets'):
            # fallback: scan one batch
            unique_classes = set()
            for data in dataset.train_loader:
                unique_classes.update(data[1].tolist())
                break  # one batch is enough to get class indices
            
        if not hasattr(self, 'class_to_task'):
            self.class_to_task = {}

        # Infer task classes directly from the train loader's targets
        targets = dataset.train_loader.dataset.targets
        if isinstance(targets, list):
            unique_classes = set(targets)
        else:
            # tensor or numpy array
            unique_classes = set(targets.tolist())

        for class_idx in unique_classes:
            self.class_to_task[class_idx] = self.current_task

        train_loader = dataset.train_loader
        if self.current_task > 0:
            num_choose = 50
            with torch.no_grad():
                train_iter = iter(train_loader)

                pbar = tqdm(train_iter, total=num_choose,
                            desc=f"Choose params for task {self.current_task + 1}",
                            disable=False, mininterval=0.5)

                count = 0
                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0]
                    x = x.to(self.device)

                    processX = self.net.vitProcess(x)

                    features = self.net.local_vitmodel.patch_embed(processX)
                    cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
                    features = torch.cat((cls_token, features), dim=1)
                    features = features + self.net.local_vitmodel.pos_embed

                    # forward pass till -3
                    for block in self.net.local_vitmodel.blocks[:-3]:
                        features = block(features)

                    features = self.net.Forever_freezed_blocks(features)

                    features = self.net.local_vitmodel.norm(features)

                    class_token = features[:, 0, :]
                    distances = [0] * len(self.net.fcArr)
                    for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
                        fc_feature = fc(class_token)
                        delta = fc_feature - dist.mean
                        inv_cov = torch.linalg.inv(dist.covariance_matrix)
                        mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
                        distances[t] += mahalanobis.mean()

                    count += 1
                    bar_log = {'distances': [round((x / count).item(), 2) for x in distances]}
                    pbar.set_postfix(bar_log, refresh=False)
                    pbar.update()
                pbar.close()

                min_idx = torch.argmin(torch.tensor(distances)).item()
                self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

        self.opt = self.get_optimizer()

    def myPrediction(self,x, tau=-10.0, N=2):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.hybrid_rematch(x, tau, N)
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()

        self.opt.zero_grad()
        if len(self.net.fcArr) > 1:
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            loss_mi = l2_distance(global_features, Freezed_global_features) #Directly calculate the L2 distance between features is more efficient than calculate MI between prediction, and it's also effective
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_kd + loss_hsic + loss_mi
            loss_vis = [loss_ce.item(), loss_kd.item(), loss_hsic.item(), loss_mi.item()]
        else:
            outputs, global_features, local_features = self.net(inputs)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()


        self.opt.step()

        return loss_vis

    def cal_expert_dist(self,x):
        processX = self.net.vitProcess(x)

        features = self.net.local_vitmodel.patch_embed(processX)
        cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.local_vitmodel.pos_embed

        # forward pass till -3
        for block in self.net.local_vitmodel.blocks[:-3]:
            features = block(features)

        features = self.net.Forever_freezed_blocks(features)

        features = self.net.local_vitmodel.norm(features)

        class_token = features[:, 0, :]
        distances = [0] * len(self.net.fcArr)
        for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
            distances[t] += mahalanobis.mean().item()
        return distances

    def hybrid_rematch(self, x, tau=-10.0, N=2, M=100, gamma=0.1):
        """
        Hybrid Re-matching inference for LEAR.
        Replaces the simple argmin(distances) call in evaluation.

        x      : input batch, already on self.device
        tau    : CRM confidence threshold (GEN entropy). Re-tune on val set.
        N      : number of top candidate experts for CRM (keep at 2)
        M      : top-M probabilities used in GEN entropy
        gamma  : GEN entropy hyperparameter
        """
        with torch.no_grad():

            # ── Step 1: ESM initial matching (same as cal_expert_dist) ──────────
            distances = self.cal_expert_dist(x)          # list of floats, len=num_experts
            ranked = np.argsort(distances)               # ascending: ranked[0] = best match
            t_f = int(ranked[0])                         # initial task identity

            # ── Step 2: forward with initially matched expert ───────────────────
            logits_f = self.net.myprediction(x, t_f)    # shape [B, num_classes]
            y_hat = logits_f.argmax(dim=-1)              # [B]

            # ── Step 3: Direct Re-matching (DRM) ────────────────────────────────
            # Map each predicted class back to its task identity
            t_s_direct_per_sample = torch.tensor(
                [self.class_to_task.get(c.item(), t_f) for c in y_hat],
                device=self.device
            )                                            # [B]

            # Check if any sample's predicted class belongs to a different task
            needs_drm = (t_s_direct_per_sample != t_f)  # [B] bool mask

            if needs_drm.any():
                # For simplicity: if majority of batch points to a new task, re-run
                # (batches in LEAR eval are typically per-domain; single expert per call)
                counts = Counter(t_s_direct_per_sample[needs_drm].tolist())
                t_s_drm = counts.most_common(1)[0][0]   # most voted task

                logits_drm = self.net.myprediction(x, t_s_drm)
                y_hat_drm  = logits_drm.argmax(dim=-1)

                # For samples that triggered DRM, update prediction and logits
                logits_f[needs_drm] = logits_drm[needs_drm]
                y_hat[needs_drm]    = y_hat_drm[needs_drm]

                # DRM resolved these samples — mark them done
                drm_resolved = needs_drm                # [B] bool

            else:
                drm_resolved = torch.zeros(x.shape[0], dtype=torch.bool,
                                        device=self.device)

            # ── Step 4: Confidence-based Re-matching (CRM) ──────────────────────
            # Only run CRM on samples NOT already fixed by DRM
            crm_candidates = ~drm_resolved              # [B] bool

            if crm_candidates.any():
                E = gen_entropy(logits_f, M=M, gamma=gamma)  # [B]

                # Samples with low confidence (E <= tau) are suspected mismatches
                low_conf = crm_candidates & (E <= tau)       # [B] bool

                if low_conf.any():
                    x_lc = x[low_conf]                       # subset of low-conf samples

                    # Get top-N expert candidates from ESM ranking
                    Gamma = ranked[:N].tolist()              # e.g. [t_f, second_best]

                    # Compare GEN entropy for each candidate expert
                    best_E      = torch.full((low_conf.sum(),), -float('inf'),
                                            device=self.device)
                    best_logits = logits_f[low_conf].clone()

                    for c in Gamma:
                        logits_c = self.net.myprediction(x_lc, c)   # [B_lc, C]
                        E_c      = gen_entropy(logits_c, M=M,
                                            gamma=gamma)          # [B_lc]

                        # Update best where this expert is more confident
                        improved = E_c > best_E                      # [B_lc] bool
                        best_logits[improved] = logits_c[improved]
                        best_E[improved]      = E_c[improved]

                    # Write CRM results back
                    lc_indices = low_conf.nonzero(as_tuple=True)[0]
                    y_hat[lc_indices]    = best_logits.argmax(dim=-1)

        return y_hat

def kl_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld

def gen_entropy(logits, M=100, gamma=0.1):
    """
    Post-hoc Generalized ENtropy (GEN) from HRM-PET Eq.3.
    logits: tensor of shape [B, num_classes]
    Returns entropy score per sample, shape [B].
    Higher = more confident (less likely mismatched).
    """
    probs = torch.softmax(logits, dim=-1)                        # [B, C]
    top_probs, _ = torch.topk(probs, min(M, probs.shape[-1]),
                               dim=-1)                            # [B, M]
    entropy = -(top_probs ** gamma * (1 - top_probs ** gamma))   # [B, M]
    return entropy.sum(dim=-1)                                    # [B]