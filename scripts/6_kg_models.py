# -*- coding: utf-8 -*-
"""
6_kg_models.py  v4.1

15 个 KG embedding 模型, 统一接口供 7_train_kg.py 调用.
所有模型继承 KGEModel, 实现 .score(h, r, t) → (B,) 标量.

★ v4.1 相对 v4.0:
  - 增加 MODEL_REG = MODEL_REGISTRY 别名 (兼容旧代码)
  - get_entity_embedding / get_relation_embedding 全部返回 numpy.ndarray
    (调用方不要再 .detach().cpu().numpy())

模型清单:
  Euclidean:  TransE, DistMult, ComplEx, RotatE, QuatE, TuckER, MuRE
  Hyperbolic: MuRP, MuRS, RotH, RefH, AttH, ConE
  Mixed:      M2GNN, GIE
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 公共: 双曲操作 (Poincare ball, 简化版)
# ============================================================
EPS = 1e-15
MAX_NORM = 1.0 - 1e-5


def expmap0(v, c):
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    gamma = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    res = gamma * v
    return project(res, c)


def logmap0(y, c):
    sqrt_c = c.sqrt()
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return (1.0 / sqrt_c) * (torch.atanh((sqrt_c * y_norm).clamp(-MAX_NORM, MAX_NORM)) / y_norm) * y


def project(x, c):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    maxn = (1.0 - 1e-3) / c.sqrt()
    cond = (norm > maxn).float()
    proj = x * (maxn / norm)
    return cond * proj + (1 - cond) * x


def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return project(num / den.clamp_min(EPS), c)


def hyp_distance(x, y, c):
    sqrt_c = c.sqrt()
    diff = mobius_add(-x, y, c)
    return 2.0 / sqrt_c * torch.atanh((sqrt_c * diff.norm(dim=-1)).clamp(-MAX_NORM, MAX_NORM))


def givens_rotation(r, x):
    cos_h = r.view(*r.shape[:-1], -1, 2)
    cos = cos_h[..., 0]
    sin = cos_h[..., 1]
    norm = (cos * cos + sin * sin).clamp_min(EPS).sqrt()
    cos = cos / norm
    sin = sin / norm
    x_h = x.view(*x.shape[:-1], -1, 2)
    x0 = x_h[..., 0]; x1 = x_h[..., 1]
    out0 = cos * x0 - sin * x1
    out1 = sin * x0 + cos * x1
    out = torch.stack([out0, out1], dim=-1).reshape(*x.shape)
    return out


def givens_reflection(r, x):
    cos_h = r.view(*r.shape[:-1], -1, 2)
    cos = cos_h[..., 0]; sin = cos_h[..., 1]
    norm = (cos * cos + sin * sin).clamp_min(EPS).sqrt()
    cos = cos / norm; sin = sin / norm
    x_h = x.view(*x.shape[:-1], -1, 2)
    x0 = x_h[..., 0]; x1 = x_h[..., 1]
    out0 = cos * x0 + sin * x1
    out1 = sin * x0 - cos * x1
    out = torch.stack([out0, out1], dim=-1).reshape(*x.shape)
    return out


# ============================================================
# 基类
# ============================================================
class KGEModel(nn.Module):
    """所有模型基类. 子类需实现 score(h_idx, r_idx, t_idx) → (B,)
    get_entity_embedding/get_relation_embedding 必须返回 numpy.ndarray."""
    def __init__(self, n_e, n_r, dim):
        super().__init__()
        self.n_e = n_e
        self.n_r = n_r
        self.dim = dim

    def score(self, h, r, t):
        raise NotImplementedError

    def get_entity_embedding(self):
        raise NotImplementedError

    def get_relation_embedding(self):
        raise NotImplementedError

    def forward(self, h, r, t):
        return self.score(h, r, t)


# ============================================================
# Euclidean 模型
# ============================================================
class TransE(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E = nn.Embedding(n_e, dim); nn.init.xavier_uniform_(self.E.weight)
        self.R = nn.Embedding(n_r, dim); nn.init.xavier_uniform_(self.R.weight)
    def score(self, h, r, t):
        return -(self.E(h) + self.R(r) - self.E(t)).norm(p=2, dim=-1)
    def get_entity_embedding(self):    return self.E.weight.detach().cpu().numpy()
    def get_relation_embedding(self):  return self.R.weight.detach().cpu().numpy()


class DistMult(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E = nn.Embedding(n_e, dim); nn.init.xavier_uniform_(self.E.weight)
        self.R = nn.Embedding(n_r, dim); nn.init.xavier_uniform_(self.R.weight)
    def score(self, h, r, t):
        return (self.E(h) * self.R(r) * self.E(t)).sum(-1)
    def get_entity_embedding(self):    return self.E.weight.detach().cpu().numpy()
    def get_relation_embedding(self):  return self.R.weight.detach().cpu().numpy()


class ComplEx(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E_re = nn.Embedding(n_e, dim); self.E_im = nn.Embedding(n_e, dim)
        self.R_re = nn.Embedding(n_r, dim); self.R_im = nn.Embedding(n_r, dim)
        for emb in (self.E_re, self.E_im, self.R_re, self.R_im):
            nn.init.xavier_uniform_(emb.weight)
    def score(self, h, r, t):
        h_re, h_im = self.E_re(h), self.E_im(h)
        r_re, r_im = self.R_re(r), self.R_im(r)
        t_re, t_im = self.E_re(t), self.E_im(t)
        return ((h_re * r_re - h_im * r_im) * t_re
                + (h_re * r_im + h_im * r_re) * t_im).sum(-1)
    def get_entity_embedding(self):
        return torch.cat([self.E_re.weight, self.E_im.weight], dim=-1).detach().cpu().numpy()
    def get_relation_embedding(self):
        return torch.cat([self.R_re.weight, self.R_im.weight], dim=-1).detach().cpu().numpy()


class RotatE(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E_re = nn.Embedding(n_e, dim); self.E_im = nn.Embedding(n_e, dim)
        self.R_phase = nn.Embedding(n_r, dim)
        for emb in (self.E_re, self.E_im):
            nn.init.xavier_uniform_(emb.weight)
        nn.init.uniform_(self.R_phase.weight, -math.pi, math.pi)
    def score(self, h, r, t):
        h_re, h_im = self.E_re(h), self.E_im(h)
        t_re, t_im = self.E_re(t), self.E_im(t)
        ph = self.R_phase(r)
        rr_re = ph.cos(); rr_im = ph.sin()
        d_re = h_re * rr_re - h_im * rr_im - t_re
        d_im = h_re * rr_im + h_im * rr_re - t_im
        return -(d_re * d_re + d_im * d_im).sum(-1).sqrt()
    def get_entity_embedding(self):
        return torch.cat([self.E_re.weight, self.E_im.weight], dim=-1).detach().cpu().numpy()
    def get_relation_embedding(self):
        return self.R_phase.weight.detach().cpu().numpy()


class QuatE(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E_a = nn.Embedding(n_e, dim); self.E_b = nn.Embedding(n_e, dim)
        self.E_c = nn.Embedding(n_e, dim); self.E_d = nn.Embedding(n_e, dim)
        self.R_a = nn.Embedding(n_r, dim); self.R_b = nn.Embedding(n_r, dim)
        self.R_c = nn.Embedding(n_r, dim); self.R_d = nn.Embedding(n_r, dim)
        for emb in (self.E_a, self.E_b, self.E_c, self.E_d,
                    self.R_a, self.R_b, self.R_c, self.R_d):
            nn.init.xavier_uniform_(emb.weight)
    def _normalize_q(self, a, b, c, d):
        norm = (a*a + b*b + c*c + d*d).clamp_min(EPS).sqrt()
        return a/norm, b/norm, c/norm, d/norm
    def score(self, h, r, t):
        ha, hb, hc, hd = self.E_a(h), self.E_b(h), self.E_c(h), self.E_d(h)
        ra, rb, rc, rd = self._normalize_q(self.R_a(r), self.R_b(r),
                                           self.R_c(r), self.R_d(r))
        ta, tb, tc, td = self.E_a(t), self.E_b(t), self.E_c(t), self.E_d(t)
        a = ha*ra - hb*rb - hc*rc - hd*rd
        b = ha*rb + hb*ra + hc*rd - hd*rc
        c = ha*rc - hb*rd + hc*ra + hd*rb
        d = ha*rd + hb*rc - hc*rb + hd*ra
        return (a*ta + b*tb + c*tc + d*td).sum(-1)
    def get_entity_embedding(self):
        return torch.cat([self.E_a.weight, self.E_b.weight,
                          self.E_c.weight, self.E_d.weight], dim=-1).detach().cpu().numpy()
    def get_relation_embedding(self):
        return torch.cat([self.R_a.weight, self.R_b.weight,
                          self.R_c.weight, self.R_d.weight], dim=-1).detach().cpu().numpy()


class TuckER(KGEModel):
    def __init__(self, n_e, n_r, dim, dim_r=None):
        super().__init__(n_e, n_r, dim)
        dim_r = dim_r or dim
        self.E = nn.Embedding(n_e, dim);   nn.init.xavier_uniform_(self.E.weight)
        self.R = nn.Embedding(n_r, dim_r); nn.init.xavier_uniform_(self.R.weight)
        self.W = nn.Parameter(torch.empty(dim, dim_r, dim))
        nn.init.xavier_normal_(self.W)
        self.bn0 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.3)
    def score(self, h, r, t):
        eh = self.E(h)
        eh = self.bn0(eh) if eh.size(0) > 1 else eh
        eh = self.drop(eh)
        rr = self.R(r)
        W_mat = torch.einsum("ijk,bj->bik", self.W, rr)
        out = torch.einsum("bd,bdk->bk", eh, W_mat)
        et = self.E(t)
        return (out * et).sum(-1)
    def get_entity_embedding(self):    return self.E.weight.detach().cpu().numpy()
    def get_relation_embedding(self):  return self.R.weight.detach().cpu().numpy()


class MuRE(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E = nn.Embedding(n_e, dim); nn.init.xavier_uniform_(self.E.weight)
        self.bh = nn.Embedding(n_e, 1);  nn.init.zeros_(self.bh.weight)
        self.bt = nn.Embedding(n_e, 1);  nn.init.zeros_(self.bt.weight)
        self.Rd = nn.Embedding(n_r, dim); nn.init.ones_(self.Rd.weight)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
    def score(self, h, r, t):
        eh = self.E(h) * self.Rd(r)
        et = self.E(t)
        d = (eh - et + self.Rt(r))
        return -(d * d).sum(-1) + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_entity_embedding(self):    return self.E.weight.detach().cpu().numpy()
    def get_relation_embedding(self):
        return torch.cat([self.Rd.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


# ============================================================
# Hyperbolic 模型
# ============================================================
class _HypBase(KGEModel):
    def __init__(self, n_e, n_r, dim, init_c=1.0):
        super().__init__(n_e, n_r, dim)
        self.E = nn.Embedding(n_e, dim); nn.init.xavier_uniform_(self.E.weight)
        self.bh = nn.Embedding(n_e, 1);  nn.init.zeros_(self.bh.weight)
        self.bt = nn.Embedding(n_e, 1);  nn.init.zeros_(self.bt.weight)
        self.c  = nn.Parameter(torch.tensor([init_c]))
    def get_entity_embedding(self): return self.E.weight.detach().cpu().numpy()
    def _c(self): return F.softplus(self.c)


class MuRP(_HypBase):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.Rd = nn.Embedding(n_r, dim); nn.init.ones_(self.Rd.weight)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
    def score(self, h, r, t):
        c = self._c()
        eh = expmap0(self.E(h) * self.Rd(r), c)
        et = expmap0(self.E(t), c)
        rt = expmap0(self.Rt(r), c)
        eh_t = mobius_add(eh, rt, c)
        d = hyp_distance(eh_t, et, c)
        return -d * d + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_relation_embedding(self):
        return torch.cat([self.Rd.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


class MuRS(_HypBase):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim, init_c=0.5)
        self.Rd = nn.Embedding(n_r, dim); nn.init.ones_(self.Rd.weight)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
    def score(self, h, r, t):
        eh = self.E(h) * self.Rd(r) + self.Rt(r)
        et = self.E(t)
        eh_n = F.normalize(eh, dim=-1)
        et_n = F.normalize(et, dim=-1)
        cos_d = (eh_n * et_n).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        d = torch.acos(cos_d)
        return -d * d + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_relation_embedding(self):
        return torch.cat([self.Rd.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


class RotH(_HypBase):
    def __init__(self, n_e, n_r, dim):
        if dim % 2 == 1: dim += 1
        super().__init__(n_e, n_r, dim)
        self.Rg = nn.Embedding(n_r, dim); nn.init.uniform_(self.Rg.weight, -1, 1)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
    def score(self, h, r, t):
        c = self._c()
        eh_e = self.E(h)
        eh_rot = givens_rotation(self.Rg(r), eh_e)
        eh = expmap0(eh_rot, c)
        et = expmap0(self.E(t), c)
        rt = expmap0(self.Rt(r), c)
        eh_t = mobius_add(eh, rt, c)
        d = hyp_distance(eh_t, et, c)
        return -d * d + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_relation_embedding(self):
        return torch.cat([self.Rg.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


class RefH(_HypBase):
    def __init__(self, n_e, n_r, dim):
        if dim % 2 == 1: dim += 1
        super().__init__(n_e, n_r, dim)
        self.Rf = nn.Embedding(n_r, dim); nn.init.uniform_(self.Rf.weight, -1, 1)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
    def score(self, h, r, t):
        c = self._c()
        eh_e = self.E(h)
        eh_ref = givens_reflection(self.Rf(r), eh_e)
        eh = expmap0(eh_ref, c)
        et = expmap0(self.E(t), c)
        rt = expmap0(self.Rt(r), c)
        eh_t = mobius_add(eh, rt, c)
        d = hyp_distance(eh_t, et, c)
        return -d * d + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_relation_embedding(self):
        return torch.cat([self.Rf.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


class AttH(_HypBase):
    def __init__(self, n_e, n_r, dim):
        if dim % 2 == 1: dim += 1
        super().__init__(n_e, n_r, dim)
        self.Rg = nn.Embedding(n_r, dim); nn.init.uniform_(self.Rg.weight, -1, 1)
        self.Rf = nn.Embedding(n_r, dim); nn.init.uniform_(self.Rf.weight, -1, 1)
        self.Rt = nn.Embedding(n_r, dim); nn.init.zeros_(self.Rt.weight)
        self.Ar = nn.Embedding(n_r, dim); nn.init.normal_(self.Ar.weight, std=0.1)
    def score(self, h, r, t):
        c = self._c()
        eh_e = self.E(h)
        rot = givens_rotation(self.Rg(r), eh_e)
        ref = givens_reflection(self.Rf(r), eh_e)
        a_rot = (rot * self.Ar(r)).sum(-1, keepdim=True)
        a_ref = (ref * self.Ar(r)).sum(-1, keepdim=True)
        a = torch.softmax(torch.cat([a_rot, a_ref], dim=-1), dim=-1)
        eh_mix = a[:, 0:1] * rot + a[:, 1:2] * ref
        eh = expmap0(eh_mix, c)
        et = expmap0(self.E(t), c)
        rt = expmap0(self.Rt(r), c)
        eh_t = mobius_add(eh, rt, c)
        d = hyp_distance(eh_t, et, c)
        return -d * d + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)
    def get_relation_embedding(self):
        return torch.cat([self.Rg.weight, self.Rf.weight, self.Rt.weight], dim=-1).detach().cpu().numpy()


class ConE(_HypBase):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim, init_c=1.0)
        self.K = nn.Embedding(n_r, dim); nn.init.uniform_(self.K.weight, 0, 1)
        self.Rd = nn.Embedding(n_r, dim); nn.init.ones_(self.Rd.weight)
    def score(self, h, r, t):
        c = self._c()
        eh = expmap0(self.E(h) * self.Rd(r), c)
        et = expmap0(self.E(t), c)
        d = hyp_distance(eh, et, c)
        ap = F.softplus(self.K(r)).sum(-1) * 0.01
        return -(d - ap) ** 2
    def get_relation_embedding(self):
        return torch.cat([self.K.weight, self.Rd.weight], dim=-1).detach().cpu().numpy()


# ============================================================
# Mixed-curvature 模型 (M2GNN, GIE)
# ============================================================
class M2GNN(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.dim_e = dim // 3
        self.dim_h = dim // 3
        self.dim_s = dim - 2 * (dim // 3)
        self.E_e = nn.Embedding(n_e, self.dim_e); nn.init.xavier_uniform_(self.E_e.weight)
        self.E_h = nn.Embedding(n_e, self.dim_h); nn.init.xavier_uniform_(self.E_h.weight)
        self.E_s = nn.Embedding(n_e, self.dim_s); nn.init.xavier_uniform_(self.E_s.weight)
        self.R_e = nn.Embedding(n_r, self.dim_e); nn.init.xavier_uniform_(self.R_e.weight)
        self.R_h = nn.Embedding(n_r, self.dim_h); nn.init.xavier_uniform_(self.R_h.weight)
        self.R_s = nn.Embedding(n_r, self.dim_s); nn.init.xavier_uniform_(self.R_s.weight)
        self.c = nn.Parameter(torch.tensor([1.0]))
    def score(self, h, r, t):
        s_e = -(self.E_e(h) + self.R_e(r) - self.E_e(t)).norm(p=2, dim=-1)
        c = F.softplus(self.c)
        eh = expmap0(self.E_h(h), c); et = expmap0(self.E_h(t), c)
        rt = expmap0(self.R_h(r), c)
        s_h = -hyp_distance(mobius_add(eh, rt, c), et, c)
        eh_s = F.normalize(self.E_s(h) + self.R_s(r), dim=-1)
        et_s = F.normalize(self.E_s(t), dim=-1)
        s_s = (eh_s * et_s).sum(-1)
        return s_e + s_h + s_s
    def get_entity_embedding(self):
        return torch.cat([self.E_e.weight, self.E_h.weight, self.E_s.weight], dim=-1).detach().cpu().numpy()
    def get_relation_embedding(self):
        return torch.cat([self.R_e.weight, self.R_h.weight, self.R_s.weight], dim=-1).detach().cpu().numpy()


class GIE(KGEModel):
    def __init__(self, n_e, n_r, dim):
        super().__init__(n_e, n_r, dim)
        self.E = nn.Embedding(n_e, dim); nn.init.xavier_uniform_(self.E.weight)
        self.R = nn.Embedding(n_r, dim); nn.init.xavier_uniform_(self.R.weight)
        self.A = nn.Embedding(n_r, 3);   nn.init.normal_(self.A.weight, std=0.1)
        self.c = nn.Parameter(torch.tensor([1.0]))
    def score(self, h, r, t):
        eh, et = self.E(h), self.E(t)
        rr = self.R(r)
        s_e = -(eh + rr - et).norm(p=2, dim=-1)
        c = F.softplus(self.c)
        eh_h = expmap0(eh, c); et_h = expmap0(et, c); rr_h = expmap0(rr, c)
        s_h = -hyp_distance(mobius_add(eh_h, rr_h, c), et_h, c)
        eh_s = F.normalize(eh + rr, dim=-1)
        et_s = F.normalize(et, dim=-1)
        s_s = (eh_s * et_s).sum(-1)
        a = torch.softmax(self.A(r), dim=-1)
        return a[:, 0] * s_e + a[:, 1] * s_h + a[:, 2] * s_s
    def get_entity_embedding(self):    return self.E.weight.detach().cpu().numpy()
    def get_relation_embedding(self):  return self.R.weight.detach().cpu().numpy()


# ============================================================
# 模型注册
# ============================================================
MODEL_REGISTRY = {
    "transe":   TransE,
    "distmult": DistMult,
    "complex":  ComplEx,
    "rotate":   RotatE,
    "quate":    QuatE,
    "tucker":   TuckER,
    "mure":     MuRE,
    "murp":     MuRP,
    "murs":     MuRS,
    "roth":     RotH,
    "refh":     RefH,
    "atth":     AttH,
    "cone":     ConE,
    "m2gnn":    M2GNN,
    "gie":      GIE,
}

# 兼容旧版 (7_train_kg 早期写法)
MODEL_REG = MODEL_REGISTRY


def get_model(name, n_e, n_r, dim):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}; 可选 {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](n_e, n_r, dim)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("可用模型:")
    for k, v in MODEL_REGISTRY.items():
        print(f"  {k:<12} -> {v.__name__}")
    print("\n各模型 forward 自测 (n_e=100, n_r=8, dim=32, batch=16):")
    n_e, n_r, dim, B = 100, 8, 32, 16
    h = torch.randint(0, n_e, (B,))
    r = torch.randint(0, n_r, (B,))
    t = torch.randint(0, n_e, (B,))
    n_pass, n_fail = 0, 0
    for k in MODEL_REGISTRY:
        try:
            m = get_model(k, n_e, n_r, dim)
            m.train()
            s = m.score(h, r, t)
            assert s.shape == (B,), f"shape 不对: {s.shape}"
            # 同时检查 emb 接口返回 numpy
            emb = m.get_entity_embedding()
            assert hasattr(emb, "shape") and not hasattr(emb, "detach"), \
                "get_entity_embedding 必须返回 numpy.ndarray"
            print(f"  {k:<12} ✓  score.mean={s.mean().item():+.4f}  emb.shape={emb.shape}")
            n_pass += 1
        except Exception as e:
            print(f"  {k:<12} ✗  {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n  通过: {n_pass}/{len(MODEL_REGISTRY)}  失败: {n_fail}")
