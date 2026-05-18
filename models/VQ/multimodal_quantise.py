# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Model codes for CodeBind
# @Reference     : https://github.com/lucidrains/vector-quantize-pytorch/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
import numpy as np
import random
from torch import einsum
from einops import repeat, rearrange
from icecream import ic 
import pdb

vq_args = {
    "embed_num": 1024,
    "embed_dim": 8,
    "distance": 'cos',
    "anchor": 'random',
    "ema_update": True,  # use ema instead of codebook loss to update codebook
    "norm_code": True,
    "kmeans_init": True,

    "shuffle_scale": 0,

    "sim_loss": False,
    "contras_loss": False,
    "uni_loss": False,
}

mvq_args = {
    "distance": 'cos',
    "anchor": 'random',
    "ema_update": True,  # use ema instead of codebook loss to update codebook
    "norm_code": True,
    "kmeans_init": True,

    "shuffle_scale": 0,

    "sim_loss": False,
    "contras_loss": False,
    "uni_loss": False,

    "embed_num": {
        "common": 1024,
        "text": 256,
        "vision": 256,
        "depth": 256,
        "audio": 256,
        "thermal": 256,  
        "imu": 256
    },
    # text: m * common_dim
    # image: m * (common_dim + specific_dim)
    "embed_dim": {
        "common": 8,
        "text": 8,
        "vision": 8,
        "depth": 8,
        "audio": 8,
        "thermal":8,  
        "imu":8
    }
}


def batched_sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    remain_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(remain_count, p / remainder)
        sample[i] = s
        remain_count -= s
        remainder -= p

    if sample.sum() < total_count:
        sample[-1] += total_count - sample.sum()
    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def all_gather_variables(x):
    rank = distributed.get_rank()
    all_x = []

    for i in range(distributed.get_world_size()):
        t = x if i == rank else x.new_empty(x.shape)
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    num_samples, device = local_samples.shape[0], local_samples.device
    if num_samples >= samples_per_rank[rank]:
        indices = torch.randperm(num_samples, device = device)[:samples_per_rank[rank]]
    else:
        indices = torch.randint(0, num_samples, (samples_per_rank[rank],), device = device)
    local_samples = local_samples[indices]
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return out


class MultimodalVectorQuantizer(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            print("MultimodalVectorQuantizer: using default settings.")
            args = mvq_args
        else:
            default_args = mvq_args
            default_args.update(cfg)
            args = default_args
        # pdb.set_trace()
        self.codevector_num = args.get('codevector_num')
        self.codevector_dim = args.get('codevector_dim')
        self.input_vector_dim = args.get('input_vector_dim')
        self.seg_type = args.get('seg_type')
        self.share_codebook = args.get('share_codebook', False)
        self.seperate_codebook = args.get('seperate_codebook', False) 

        self.mvq_dict = {}
        # common VQ shared among all modality
        self.mvq_dict['common'] = VectorQuantizer(
            embed_num=self.codevector_num.get('common'),
            embed_dim=self.codevector_dim.get('common'),
            args=args,
            )
        # specific VQ for each modality
        all_modality_list = ["vision", "text", "audio", "thermal", "depth", "imu", "tactile", "eeg"]
        modality_list = args.get('modality_pair') if args.get('modality_pair') else all_modality_list
        print(f"build MVQ for {modality_list}")
        for k in modality_list:
            if self.codevector_dim.get(k) == 0 or self.input_vector_dim.get(k) == 0:
                continue
            self.mvq_dict[k] = VectorQuantizer(
                embed_num=self.codevector_num.get(k), 
                embed_dim=self.codevector_dim.get(k),
                args=args,
                )
        self.mvq_dict = nn.ModuleDict(self.mvq_dict)
    

    def split_vector(self, z, modality_name):
        if z.dim() > 2:
            dim_last = z.shape[-1]
            z = z.view(-1, dim_last)
        bs = z.shape[0]
        codevector_common_dim = self.codevector_dim.get('common')
        codevector_specific_dim = self.codevector_dim.get(modality_name)

        if self.seg_type == 1:  # first, divide tensor into common and specific subvectors, then split each subvector into small chunks for VQ
            input_vector_common_dim = self.input_vector_dim.get('common')
            input_vector_specific_dim = self.input_vector_dim.get(modality_name)
            z_common, z_specific = torch.split(z, [input_vector_common_dim, input_vector_specific_dim], dim=1)
            z_common = torch.reshape(z_common, (bs, -1, codevector_common_dim))
            z_specific = torch.reshape(z_specific, (bs, -1, codevector_specific_dim)) if codevector_specific_dim else z_specific
        
        if self.seg_type == 2:  # first, split tensor into equally sized chunks, then divide each chunk into common and specific subvectors
            # assert self.input_vector_dim.get('common') / codevector_common_dim == self.input_vector_dim.get('modality_name') / codevector_specific_dim
            z_full = torch.reshape(z, (bs, -1, codevector_common_dim + codevector_specific_dim))
            z_common, z_specific = torch.split(z_full, [codevector_common_dim, codevector_specific_dim], dim=2)
        
        return z_common, z_specific
    

    def merge_vector(self, z_common, z_specific):
        if self.seg_type == 1:
            z_common = torch.flatten(z_common, start_dim=1)
            z_specific = torch.flatten(z_specific, start_dim=1)
            z_full = torch.cat((z_common, z_specific), dim=-1)
        if self.seg_type == 2:
            z_full = torch.cat((z_common, z_specific), dim=2)
            z_full = torch.flatten(z_full, start_dim=1)
        return z_full

    
    def forward(self, z, modality_name, mode='val'):
        """ mode in [
            "train",  # VQ inference, update codebook and get loss
            "val",    # VQ inference
            "loss"    # VQ inference and get loss
            ]
        """
        # pdb.set_trace()
        # split intput vector z into common and sepefic subvectors
        input_vector_specific_dim = self.input_vector_dim.get(modality_name)

        pesudo_bs = [i for i in z.shape[:-1]]  # [bs] or [bs, d1, d2, ...]
        ori_shape = pesudo_bs + [-1]  # bs, d1, d2, ..., -1

        z_common, z_specific = self.split_vector(z, modality_name)
        res_dict = {}

        # vq
        # common
        if self.seperate_codebook:
            # in seperate codebook mode, common codebook is only applied to text, and specific codebook is applied to other modalities
            common_codebook_name = 'common' if modality_name == 'text' else modality_name
        else:
            common_codebook_name = 'common' 
        res_common_dict = self.mvq_dict[common_codebook_name](z_common, pesudo_bs) 

        # res_common_dict = self.mvq_dict['common'](z_common, pesudo_bs, update_codebook=False)
        # specific
        specific_codebook_name = 'common' if self.share_codebook else modality_name
        res_specific_dict = self.mvq_dict[specific_codebook_name](z_specific, pesudo_bs) if input_vector_specific_dim else {'q': z_specific}  # Note: input_vector_specific_dim can be 0

        q_common = res_common_dict.get('q')
        q_specific = res_specific_dict.get('q')
        q = self.merge_vector(q_common, q_specific)

        q_common = F.normalize(q_common.view(ori_shape), p=2, dim=-1)
        q_specific = F.normalize(q_specific.view(ori_shape), p=2, dim=-1) if input_vector_specific_dim else q_specific
        q = F.normalize(q.view(ori_shape), p=2, dim=-1)

        # q_common = q_common.view(ori_shape)
        # q_specific = q_specific.view(ori_shape) if input_vector_specific_dim else q_specific
        # q = q.view(ori_shape)

        res_dict.update({'common': q_common})      # q_common.view(bs, -1)
        res_dict.update({'specific': q_specific})  # q_specific.view(bs, -1)
        res_dict.update({'concat': q})

        # assemble the subvectors together
        # TODO input_vector_specific_dim = 0 时，补全代码逻辑的完整性
        if mode == 'train':
            loss_vq_list = ['loss_codebook', 'loss_contra', 'loss_uni']
            for loss_term in loss_vq_list:
                out_loss_c = res_common_dict.get(loss_term, 0)
                out_loss_s = res_specific_dict.get(loss_term, 0)
                res_dict.update({loss_term: out_loss_c + out_loss_s})
            res_dict.update({'res_common': res_common_dict})
            res_dict.update({'res_specific': res_specific_dict})
        elif mode == 'val':
            pass
        return res_dict
            
    
# contrastive loss
def contrastive_loss(dist, embed_num):
    
    # 对于每个code来说，拉大接近的feature与远离的feature之间的距离
    # sort_distance, indices = dist.sort(dim=0)
    # dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/embed_num)):,:].mean(dim=0, keepdim=True)
    # dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
    # dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
    # contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
    # 对于每个feature来说，拉大接近的code与远离的code之间的距离
    sort_distance, indices = dist.sort(dim=1)
    dis_pos = sort_distance[:, -int(embed_num/10):].mean(dim=1, keepdim=True)
    dis_neg = sort_distance[:, :int(sort_distance.size(1)*1/2)]
    dis = torch.cat([dis_pos, dis_neg], dim=1) / 1
    if torch.any(dis < 0):
        dis = F.softmax(dis, dim=1)
    else:
        dis = F.softmax(torch.sqrt(dis), dim=1) 
    contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
    return contra_loss



def similarity_loss(normed_codebook, embed_num):
    cos_sim = 0.5 + 0.5 * torch.matmul(normed_codebook, normed_codebook.t())

    # cosine_similarity = torch.matmul(normed_z_flattened, normed_codebook.t())  # {a} . {b}
    # cos_sim = {a} . {b} / |a|*|b|, 值域为[-1， 1]。1为相同，-1则相反

    # 去掉对角线上自相似度==1的元素
    sim_loss = (torch.sum(cos_sim) - embed_num) / (embed_num * (embed_num - 1))  # sim_loss值在0.5 左右

    # cos_sim 按列求和，含义为：该entry 与其他entry的相似度之和
    # cos_sim 按列求max，含义为：该entry 与其他entry的相似度最大的值

    """
    # import pdb
    # pdb.set_trace()
    diag = torch.diag(cos_sim)  # 取对角线元素，输出为 1*N
    sim_diag = torch.diag_embed(diag)   # 由 diag 恢复为N*N
    sim_masked = cos_sim - sim_diag     # cos_sim 对角线置 0
    # sim_sum_each_row = torch.sum(sim_masked, dim=1)  # 每一行内的元素相加
    # sim_row_max_val = torch.max(sim_sum_each_row)        # 相似度之和的最大值   
    # sim_row_max_idx = torch.argmax(sim_sum_each_row)     # 相似度之和的最大值的索引   
    sim_max_each_row, sim_maxidx_each_row = torch.max(sim_masked, dim=1)
    sim_max = torch.max(sim_max_each_row)  # torch.max(sim_max_each_row) == torch.max(sim_masked)

    sim_max = torch.max(sim_masked)
    # loss_dict.update({'sim_max': sim_max})
    # loss_dict.update({'sim_row_max_val': sim_row_max_val})
    """

    return sim_loss

def uniform_loss(normed_codebook):
    # l2 distances among codes
    uni_loss = torch.pdist(normed_codebook,p=2).pow(2).mul(-1).exp().mean().log()
    return uni_loss

class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = F.normalize(torch.randn(num_tokens, codebook_dim), dim=1)
        # weight = torch.randn(num_tokens, codebook_dim)
        # weight = torch.Tensor(num_tokens, codebook_dim).uniform_(-1.0 / num_tokens, 1.0 / num_tokens)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class VectorQuantizer(nn.Module):
    def __init__(self, embed_num, embed_dim, beta=0.25, args=None):
        super().__init__()
        if args is None:
            print("args is None, using default settings.")
            args = vq_args

        self.embed_num = embed_num
        self.embed_dim = embed_dim # self.embed_dim_share + self.embed_dim_specific
        self.beta = beta  # commitment cost
        self.distance = args.get('distance')
        self.anchor = args.get('anchor')
        self.ema_update = args.get('ema_update', False)
        self.contras_loss = args.get('contras_loss', False)
        self.sim_loss = args.get('sim_loss', False)
        self.uni_loss = args.get('uni_loss', False)
        self.norm_code = args.get('norm_code', False)
        self.codebook_stat = args.get('codebook_stat', False)
        self.sync_codebook = True if distributed.is_initialized() else False

        # --
        self.decay = 0.99
        self.shuffle_scale = args.get('shuffle_scale', 0)

        # self.pool = FeaturePool(self.embed_num, self.embed_dim)
        if self.ema_update:
            self.embedding = EmbeddingEMA(self.embed_num, self.embed_dim)
        else:
            self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.embed_num, 1.0 / self.embed_num)
        self.register_buffer("embed_prob", torch.zeros(self.embed_num))

        self.kmeans_init = args.get('kmeans_init', False)
        self.register_buffer('initted', torch.Tensor([not self.kmeans_init]))

        print(f"VectorQuantiser: codebook_size = {self.embed_num} * {self.embed_dim}")
        print(f"codebook kmeans init = {self.kmeans_init}")
        print(f"codebook reinit mode = {self.anchor}")
        print(f"ema update = {self.ema_update}")
        print(f"features and codes normalization = {self.norm_code}")
        print("self.training =", self.training)
        print("---> VectorQuantizer: init success.")


    def upsample(self, x, scale_factor=2):
        # N,C,H_in,W_in --> N,C,H_in*scale_factor,W_in*scale_factor
        y = F.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        # shape (∗,C,H×r,W×r) to (∗,C×r^2,H,W)
        y = F.pixel_unshuffle(y, downscale_factor=scale_factor) 
        # bs, c, h, w = x.size()
        # y = y.view([bs, c, h, scale_factor, w, scale_factor])
        # y = torch.permute(y, (0, 1, 3, 5, 2, 4))
        # y = y.reshape([bs, c*scale_factor**2, h, w])
        return y
    

    def downsample(self, x, scale_factor=2):
        # shape (∗, C×r^2, H, W) to (∗, C, H×r, W×r) 
        y = F.pixel_shuffle(x, upscale_factor=scale_factor) 
        y = F.avg_pool2d(y, kernel_size=scale_factor)
        return y

        
    def preprocess(self, z, is_4d_tensor):
        if self.shuffle_scale:
            z = self.upsample(z, scale_factor=self.shuffle_scale)
        if is_4d_tensor:  # len(z.shape) == 4:  # channel_first to channel_last
            z = rearrange(z, 'b c h w -> b h w c').contiguous()  # z.shape: torch.Size([40, 1024])
        return z
    

    def postprocess(self, z, is_4d_tensor):
        if is_4d_tensor:
            # reshape back to match original input shape
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        if self.shuffle_scale:
            z = self.downsample(z, scale_factor=self.shuffle_scale)
        return z
    

    def update_codebook(self, z_flattened, dist, bin_count):
        # calculate the average usage of code entries
        # add: each element of 'avg_probs' is scaled by alpha before being used.
        if self.sync_codebook:
            distributed.all_reduce(bin_count)
        avg_probs = bin_count / torch.sum(bin_count)
        self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
        # running average updates
        if self.anchor in ['closest', 'random', 'probrandom', 'disturbance']:
            # closest sampling
            # 取最接近codebook的self.embed_num个flattened feature vector 
            if self.anchor == 'closest':
                encoding_indices_dim0 = torch.argmax(dist, dim=0)
                random_feat = z_flattened.detach()[encoding_indices_dim0]
                if self.sync_codebook:
                    distributed.all_reduce(random_feat)
                    random_feat /= distributed.get_world_size()
                if self.norm_code:
                    random_feat = F.normalize(random_feat, dim=-1)
                # print("encoding_indices.shape =", encoding_indices.shape, "indices.shape =", indices.shape)
                      
            # feature pool based random sampling
            # 将以往的feature保存在feature pool中，随机从中给出feature
            elif self.anchor == 'random':
                if self.sync_codebook:
                    random_feat = sample_from_featurepool_distributed(z_flattened.detach(), self.embed_num)
                else:
                    random_feat = sample_from_featurepool(z_flattened.detach(), self.embed_num)
                # random_feat = self.pool.query(z_flattened.detach())  

            # probabilitical based random sampling
            elif self.anchor == 'probrandom':
                norm_distance = F.softmax(dist.t(), dim=1)  #  消耗显存
                # 将norm_distance作为采样的权重，取num_samples个数，返回的是索引。
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)  
                random_feat = z_flattened.detach()[prob]
                if self.sync_codebook:
                    distributed.all_reduce(random_feat)
                    random_feat /= distributed.get_world_size()
                if self.norm_code:
                    random_feat = F.normalize(random_feat, dim=-1)

            # TODO
            elif self.anchor == 'disturbance':  
                # random_feat = self.pool.query(z_flattened.detach())
                if self.sync_codebook:
                    random_feat = sample_from_featurepool_distributed(z_flattened.detach(), self.embed_num)
                else:
                    random_feat = sample_from_featurepool(z_flattened.detach(), self.embed_num)
                # For each bit of feature, add a random disturbance between [-1, 1]
                # 对feature的每一位，增加一个[-1, 1] 之间的随机扰动
                random_feat += (torch.rand((self.embed_num, self.embed_dim)) * 2 - 1).to(random_feat.device)
                if self.norm_code:
                     random_feat = F.normalize(random_feat, dim=-1)       

            # decay parameter based on the average usage
            decay = torch.exp(-(self.embed_prob*self.embed_num*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
            # decay = 0.99
            # 初始值都比较小（0.002），逐渐增大（0.4）
            # 说明随着训练逐渐收敛，codebook被匹配的向量逐渐固化下来，并且部分codebook的匹配较少。
            # print("torch.max(decay)=", torch.max(decay))  
            self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        elif self.anchor in ['none', 'mutation']:
            if self.anchor == 'none':  # 
                pass
            elif self.anchor == 'mutation':  # 基因突变
                # TODO 若采用突变的方式，则需要维护两个codebook， 仅对需要更新的codebook进行突变， 且不再进行decay
                # --- 单点突变
                # - 随机选择一个位置，增加一个随机的扰动

                # --- 整体突变
                # - 对entry的每一位，增加一个[-1, 1] 之间的随机扰动
                # random_feat = (torch.rand((z_flattened.size(0), self.embed_dim)) * 2 - 1)/ z_flattened.size(0)
                # torch.rand((self.embed_num, self.embed_dim))* 2 - 1

                # --- 结构突变
                # - 随机交换2个单点位置
                # - 随机位置截断，然后交换顺序

                # --- 交叉突变
                # - 随机从其他entry中，选择一部分进行替换
                
                raise NotImplementedError
        else:
            raise NotImplementedError


    def kmeans(self, samples, num_iters = 10, use_cosine_sim = True, sample_fn = batched_sample_vectors):
        dim, dtype = samples.shape[-1], samples.dtype
        means = sample_fn(samples, self.embed_num)
        
        for _ in range(num_iters):
            if use_cosine_sim:
                if not self.norm_code:
                    dists = torch.einsum('bd,dn->bn', F.normalize(samples, dim=-1), rearrange(F.normalize(means, dim=-1), 'n d -> d n'))
                else:
                    dists = torch.einsum('bd,dn->bn', samples, rearrange(means, 'n d -> d n'))
            else:
                dists = - torch.sum(samples ** 2, dim=1, keepdim=True) - \
                    torch.sum(means ** 2, dim=1) + \
                    2 * torch.einsum('bd, dn-> bn', samples, rearrange(means, 'n d-> d n'))

            buckets = torch.argmax(dists, dim = -1)
            new_means = buckets.new_zeros(self.embed_num, dim, dtype = dtype)
            new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
            bins = torch.bincount(buckets, minlength=self.embed_num) 
            if self.sync_codebook:
                distributed.all_reduce(bins)
                distributed.all_reduce(new_means)
            bins_min_clamped = bins.masked_fill(bins==0, 1)
            new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
            
            if self.norm_code:
                new_means = F.normalize(new_means, dim=-1)

            means = torch.where(
                rearrange(bins==0, '... -> ... 1'),
                means,
                new_means
            )

        return means, bins

    def init_embed_(self, z, pesudo_bs):

        z_init = z.reshape((pesudo_bs[0], -1, z.shape[1], z.shape[2]))[:, 0, ...]
        embed, cluster_size = self.kmeans(z_init.reshape(-1, self.embed_dim), 
                                          sample_fn = sample_vectors_distributed if self.sync_codebook else batched_sample_vectors,
                                          )
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embedding.weight.data.copy_(embed)
        if self.ema_update:
            self.embedding.embed_avg.data.copy_(embed_sum)
            self.embedding.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        
    def forward(self, z, pesudo_bs=None, update_codebook=True):
        embed_dim = self.embed_dim
        embed_num = self.embed_num
        pesudo_bs = [i for i in z.shape[:-1]] if pesudo_bs is None else pesudo_bs
        res_dict = {}
        # BUG: 存在bug可能，即当对shape 为[bs, token_num, m, c]的input 进行VQ时，需 reshape 为 [bs * token_num * m, c]。当前的preprocess的操作有误
        if self.norm_code:
            z = F.normalize(z, dim=-1)
        is_4d_tensor = len(z.shape) == 4  
        z = self.preprocess(z, is_4d_tensor)

        # flatten
        # Split the feature vector into multiple segments 
        z_flattened = z.reshape(-1, embed_dim)

        if not self.initted:
            self.init_embed_(z, pesudo_bs)
        embed_weight = self.embedding.weight

        # all_embed_weight = all_gather_variables(embed_weight)
        # print(all_embed_weight[0].equal(all_embed_weight[1]))

        # calculate the distance
        if self.distance == 'l2':
            if not self.norm_code:
                z_flattened = F.normalize(z_flattened, dim=1) 
            normed_codebook = F.normalize(embed_weight, dim=1)         # [embed_num, embed_dim]
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            dist = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(embed_weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(embed_weight, 'n d-> d n'))
       
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j
            if not self.norm_code: 
                z_flattened = F.normalize(z_flattened, dim=1)      # [bs * orig_feature_dim/embed_dim, embed_dim]
            normed_codebook = F.normalize(embed_weight, dim=1)         # [embed_num, embed_dim]
            # 以下这步einsum的显存占用，与embed_num成正比，与embed_dim^2成反比。因而embed_num x embed_dim = 8192 x 2时，显存占用很高。          
            dist = torch.einsum('bd,dn->bn', z_flattened.detach(), rearrange(normed_codebook, 'n d -> d n'))  # [bs * orig_feature_dim/embed_dim, embed_num]
        
        encoding_indices = torch.argmax(dist, dim=1)  # [bs * orig_feature_dim/embed_dim, ]

        # quantise and unflatten 
        z_q = embed_weight[encoding_indices, :] # num, dim
        z_q = z_q.view(z.shape)   # [bs, m, embed_dim] or [bs*token_num, m, embed_dim]

        
        bin_count = torch.bincount(encoding_indices, minlength=embed_num)  # bincount 来获得不同codebook的出现频率

        # bin count for each pesudo_bs
        if self.codebook_stat:
            bin_count_batch = torch.zeros(pesudo_bs + [embed_num, z.shape[1]], device=encoding_indices.device) # [bs, token_num, embed_num, m]
            src = torch.ones_like(encoding_indices.view(pesudo_bs + [-1])[..., None, :], dtype=torch.float32) # [bs, token_num, 1, m]
            bin_count_batch = bin_count_batch.scatter_add(-2, encoding_indices.view(pesudo_bs + [-1])[..., None, :], src) # [bs, token_num, embed_num, m]
            bin_count_batch = bin_count_batch.sum(-1) # [bs, token_num, embed_num]
            # bin_count = bin_count_batch.view(-1, embed_num).sum(0) # bincount overall batch and token
        
        res_dict.update({'dist': dist.view(pesudo_bs + [-1, dist.shape[-1]])})
        res_dict.update({'indices': encoding_indices.view(pesudo_bs + [-1])})
        if self.codebook_stat:
            res_dict.update({'bin_count': bin_count_batch})
            res_dict.update({'normed_codebook': normed_codebook})

        if self.training and update_codebook:
            # codebook ema update
            if self.ema_update:
                encodings = F.one_hot(encoding_indices, embed_num).type(z.dtype)     
                #EMA cluster size
                encodings_sum = encodings.sum(0)  # == bin_count.type(encodings.dtype)
                if self.sync_codebook:
                    distributed.all_reduce(encodings_sum)        
                self.embedding.cluster_size_ema_update(encodings_sum)
                #EMA embedding average
                embed_sum = encodings.transpose(0,1) @ z_flattened
                if self.sync_codebook:
                    distributed.all_reduce(embed_sum)            
                self.embedding.embed_avg_ema_update(embed_sum)
                #normalize embed_avg and update weight
                self.embedding.weight_update(embed_num)

            # online clustered reinitialisation for unoptimized points
            self.update_codebook(z_flattened, dist, bin_count)

        # compute loss for embedding
        loss_commitment = torch.mean((z_q.detach()-z)**2)      # commitment loss
        if self.ema_update:
            loss_vq = self.beta * loss_commitment
            res_dict.update({'loss_codebook': loss_vq})
        else:
            loss_codebook = torch.mean((z_q - z.detach()) ** 2)    # codebook loss
            loss_vq = self.beta * loss_commitment + loss_codebook
            res_dict.update({'loss_codebook': loss_vq})

        # contrastive loss
        if self.contras_loss:
            contra_loss = contrastive_loss(dist, embed_num)
            # loss += contra_loss
            res_dict.update({'loss_contra': contra_loss}) # contra_loss

        # codebook的每个entry之间尽可能正交
        # 计算两两的相似度
        if self.sim_loss:
            sim_loss = similarity_loss(normed_codebook, embed_num)
            # loss += sim_loss
            res_dict.update({'loss_sim': sim_loss})

        if self.uni_loss:
            uni_loss = uniform_loss(normed_codebook)
            # loss += uni_loss
            res_dict.update({'loss_uni': uni_loss})


        # res_dict.update({'loss': loss})

        # preserve gradients
        if update_codebook:
            z_q = z + (z_q - z).detach()
        z_q = self.postprocess(z_q, is_4d_tensor)
        res_dict.update({'q': z_q})
        return res_dict


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

def sample_from_featurepool(features, pool_size):
    nums_features = 0
    feature_dim = features.shape[-1]
    new_samples = (torch.rand((pool_size, feature_dim)) * 2 - 1)/ pool_size
    new_samples = new_samples.to(features.device)

    if nums_features < pool_size:
        if features.size(0) > pool_size: # if the batch size is large enough, directly update the whole codebook
            random_feat_id = torch.randint(0, features.size(0), (int(pool_size),))
            new_samples = features[random_feat_id]
            nums_features = pool_size
        else:
            # if the mini-batch is not large nuough, just store it for the next update
            num = nums_features + features.size(0)
            new_samples[nums_features:num] = features
            nums_features = num
    else:
        if features.size(0) > int(pool_size):
            random_feat_id = torch.randint(0, features.size(0), (int(pool_size),))
            new_samples = features[random_feat_id]
        else:
            random_id = torch.randperm(pool_size)
            new_samples[random_id[:features.size(0)]] = features

    return new_samples
    
def sample_from_featurepool_distributed(local_features, pool_size):

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_features, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(pool_size, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_features = sample_from_featurepool(local_features, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_features, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return out