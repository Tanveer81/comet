# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

# Controls debug messages to be printed in the console with the function log.
DEBUG = False

# Utility function to control debug messages.
def log(s, q=False):
    if DEBUG:
        print(s)
        if q == True:
            quit()


class COMET(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(COMET, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.wt = nn.Parameter(data=torch.ones(16), requires_grad=True).cuda()
        
        # At te initial stage we would create transformer architecture dynamically.
        self.initial = True

    def set_forward(self, x, joints=None, is_feature=False):
        z_support, z_query = self.parse_feature(x, joints, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        log(f"prototype dimention {z_proto.shape}")
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        log(f"z_query dimention {z_query.shape}")
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        log(f"scores {scores.shape}")

        return scores

    def set_forward_loss(self, x, joints):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x, joints)

        return self.loss_fn(scores, y_query)

    def parse_feature(self, x, joints, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            log(f"z_all {z_all.shape}")
            z_avg = self.globalpool(z_all).view(z_all.size(0), z_all.size(1))
            
            # This is the size of embedding vector of each concept.
            concept_embedding_size = z_all.size(1)
            
            log(f"z_avg {z_avg.shape}")
            joints = joints.contiguous().view(self.n_way * (self.n_support + self.n_query), *joints.size()[2:])
            log(f"joints {joints.shape}")
            img_len = x.size()[-1]
            log(f"img_len {img_len}")
            feat_len = z_all.size()[-1]
            log(f"feat_len {feat_len}")
            joints[:, :, :2] = joints[:, :, :2] / img_len * feat_len
            joints = joints.round().int()
            batch_num = joints.size(0)
            log(f"batch_num {batch_num}")
            joints_num = joints.size(1)
            log(f"joints_num {joints_num}")

            feat_list = []
            for i in range(batch_num):
                feat = []
                for j in range(joints_num):
                    if joints[i, j, 2] == 1 and joints[i, j, 0] >= 0 and joints[i, j, 1] >= 0 and \
                            joints[i, j, 0] < feat_len and joints[i, j, 1] < feat_len:
                        feat.append(z_all[i, :, joints[i, j, 0], joints[i, j, 1]])
                    else:
                        feat.append(z_avg[i, :])
                feat.append(z_avg[i, :])
                feat = torch.cat(feat, dim=0) 
                feat_list.append(feat.view(1, -1))
            log(f"feat_list[0] {feat_list[0].shape}")

            # Number of extracted concepts
            number_of_concepts = int(feat_list[0].shape[1] / concept_embedding_size)
            log(f"concept_embedding_size {concept_embedding_size}")
            log(f"number_of_concepts {number_of_concepts}")
            
            # In the initial stage transformer encoder is created dynamically based on number of concepts.
            if self.initial:
                # Concept embedding size is the number of feature for each concept in the sequence.
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=concept_embedding_size, nhead=8).cuda()
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6).cuda()
                # This will block the transformer to be initialized further.
                self.initial = False

            z_all = torch.cat(feat_list, dim=0)
            log(f"z_all1 {z_all.shape}")
            
            # Only used transformer for automatic feature extraction
            if is_feature:
                z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
            else:
                # The first dimention = batch_size. The embedding was reshaped to be used by transformer.
                z_all = z_all.view(self.n_way * (self.n_support + self.n_query), number_of_concepts, concept_embedding_size)
                log(f"z_all2 {z_all.shape}")
                # Transformer encoder expects the batch_size to be in the second dimention
                z_all = z_all.permute(1, 0, 2)
                # Get new embedding vectors from transformer encoder
                z_all = self.transformer_encoder(z_all)
                # Bring batch_size to the first dimention again
                z_all = z_all.permute(1, 0, 2)
                # Reshaped to the original dimentions again.
                z_all = torch.reshape(z_all, (self.n_way, self.n_support + self.n_query, -1))
                log(f"z_all3 {z_all.shape}")
                
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        log(f"z_support {z_support.shape}")
        log(f"z_query {z_query.shape}")

        return z_support, z_query

    def correct(self, x, joints):
        scores = self.set_forward(x, joints)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, tf_writer):
        print_freq = 10

        avg_loss = 0
        for i, (x, y, joints) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, joints)
            log(f"loss {loss}")
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                tf_writer.add_scalar('loss/train', avg_loss / float(i + 1), epoch)

            log(f"avg_loss {avg_loss}", True)

    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y, joints) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x, joints)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (
        iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean


def euclidean_dist(x, y):
    # x: N x D  z_query,
    # y: M x D  z_proto
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
