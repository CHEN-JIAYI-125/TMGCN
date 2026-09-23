#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reply Structure Knowledge Distillation 核心模块。

本文件只负责「回复结构」这一条通道，不改动 Speaker / Dialogue / Temporal
超图与对比学习。三个组件对应论文式流水线：

1. ``ReplyPredictor``（Student）
   输入冻结句向量 ``h_i``，对每个当前话语 ``u_t`` 预测它回复历史话语
   ``u_j (j < t)`` 以及特殊选项 ROOT 的概率 ``P_S(j|t)``。
   输出形状 ``[N, N+1]``：第 0 列是 ROOT，第 ``j+1`` 列是节点 ``j``。

2. ``ReplyKDLoss``
   用温度 ``T`` 的 KL 散度把 Student 分布拉向 LLM Teacher 分布：
   ``L_reply_KD = T^2 * KL(P_T || P_S)``。

3. ``build_soft_reply_graph``
   将 ``P_S``（或 ``P_T``）阈值化成带权回复边，替换原来的
   ``reply_candidate_edge_index``（全历史、等权）供 Reply RGCN 使用。

关闭 ``reply_distillation`` 时本模块不会被实例化，主模型前向与 baseline
完全一致。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_causal_reply_mask(
    logits: torch.Tensor,
    extra_valid: Optional[torch.Tensor] = None,
    use_index_tril: bool = False,
) -> torch.Tensor:
    """对 ``[N, N+1]`` 回复打分施加因果掩码。

    规则：
    - 第 0 列 ROOT 永远合法；
    - 禁止自己回复自己；
    - 若提供 ``extra_valid``（推荐：原 ``reply_candidate_edge_index`` 转成的
      ``[N, N]`` 矩阵），只保留「同一 conversation 内、turn j < t」的历史；
    - ``use_index_tril=True`` 才用节点下标 ``j < t`` 做下三角。目标图节点
      顺序是 first-occurrence，不是对话时间，**默认关闭**。

    非法位置填 ``-inf``，供后续 masked softmax 使用。
    """
    n = logits.size(0)
    device = logits.device
    valid = torch.ones((n, n + 1), dtype=torch.bool, device=device)
    valid[:, 0] = True
    if n > 0:
        valid[:, 1:] = ~torch.eye(n, dtype=torch.bool, device=device)
    if use_index_tril:
        hist_valid = torch.tril(
            torch.ones((n, n), dtype=torch.bool, device=device),
            diagonal=-1,
        )
        valid[:, 1:] = valid[:, 1:] & hist_valid
    if extra_valid is not None:
        extra = extra_valid.to(device=device, dtype=torch.bool)
        if extra.dim() == 2 and extra.size(1) == n:
            extra_full = torch.zeros_like(valid)
            extra_full[:, 0] = True
            extra_full[:, 1:] = extra
            extra = extra_full
        valid = valid & extra
    return logits.masked_fill(~valid, float("-inf"))


def masked_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """带 -inf 掩码的 softmax。某行若全为 -inf（理论上不应发生，ROOT 总开着），
    则退回均匀分布，避免 NaN。"""
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        n_col = logits.size(dim)
        return torch.full_like(logits, 1.0 / max(1, n_col))
    # 对「整行都非法」的极端情况做保护
    all_masked = ~finite.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(all_masked, 0.0)
    probs = torch.softmax(safe, dim=dim)
    if bool(all_masked.any()):
        fill = torch.full_like(probs, 1.0 / probs.size(dim))
        probs = torch.where(all_masked, fill, probs)
    return probs


def candidate_edges_to_valid_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """把回复候选边 ``(j -> t)`` 转成 ``[N, N]`` 合法历史掩码。

    原 TMGCN 的 ``reply_candidate_edge_index`` 已经是「同一 conversation 内
    全部 j < t」，用它作为 Student 的 support，既保持对话内因果，又避免
    目标图上跨对话的节点被错误连边。
    """
    if device is None and edge_index.numel() > 0:
        device = edge_index.device
    valid = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    if edge_index is None or edge_index.numel() == 0:
        return valid
    src = edge_index[0].long()  # 历史节点 j
    dst = edge_index[1].long()  # 当前节点 t
    ok = (
        (src >= 0)
        & (dst >= 0)
        & (src < num_nodes)
        & (dst < num_nodes)
        & (src != dst)
    )
    if bool(ok.any()):
        valid[dst[ok], src[ok]] = True
    return valid


def build_soft_reply_graph(
    reply_probs: torch.Tensor,
    threshold: float = 0.05,
    valid_hist: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """由回复概率构造带权 Soft Reply Graph。

    Parameters
    ----------
    reply_probs:
        ``[N, N+1]``，``P(j|t)``。第 0 列是 ROOT（不是图节点，不会变成边）。
        ``reply_probs[t, j+1]`` = 当前话语 t 回复历史节点 j 的概率。
    threshold:
        只保留 ``P_S(j|t) > threshold`` 的边。默认 0.05。
    valid_hist:
        可选 ``[N, N]`` 布尔掩码；若给出，只在合法历史位置建边
        （通常就是原 reply-candidate 支持集）。

    Returns
    -------
    edge_index : LongTensor [2, E]
        有向边 ``j -> t``（历史指向当前，与原 candidate 图方向一致）。
    edge_weight : FloatTensor [E]
        边权就是 ``P(j|t)``，供 Reply RGCN 做加权消息传递。

    例：原 candidate 图 ``u1,u2,u3 -> u4`` 权全为 1；
    蒸馏后可能变成 ``0.08 / 0.12 / 0.80``。
    """
    if reply_probs.dim() != 2 or reply_probs.size(1) != reply_probs.size(0) + 1:
        raise ValueError(
            "reply_probs 必须是 [N, N+1]，当前形状为 %s" % (tuple(reply_probs.shape),)
        )
    n = reply_probs.size(0)
    device = reply_probs.device
    # 丢掉 ROOT 列，得到当前话语对历史节点的概率 [N, N]
    hist = reply_probs[:, 1:]
    keep = hist > float(threshold)
    if valid_hist is not None:
        keep = keep & valid_hist.to(device=device, dtype=torch.bool)
    # 再保险：禁止自环
    keep = keep & ~torch.eye(n, dtype=torch.bool, device=device)
    # 训练初期 Student 近均匀时，单条历史的概率可能全部低于阈值，
    # 回复通道会变成空图。对「有合法历史但一条边都没留下」的节点，
    # 至少保留概率最大的那条 antecedent，保证 Reply RGCN 仍能传播。
    if valid_hist is not None:
        hist_ok = valid_hist.to(device=device, dtype=torch.bool)
        has_hist = hist_ok.any(dim=-1)
        has_edge = keep.any(dim=-1)
        need = has_hist & ~has_edge
        if bool(need.any()):
            scores = hist.masked_fill(~hist_ok, float("-inf"))
            j_star = scores.argmax(dim=-1)
            t_star = need.nonzero(as_tuple=True)[0]
            keep[t_star, j_star[t_star]] = True

    if not bool(keep.any()):
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=reply_probs.dtype, device=device),
        )

    t_idx, j_idx = keep.nonzero(as_tuple=True)
    edge_index = torch.stack([j_idx, t_idx], dim=0)
    edge_weight = hist[t_idx, j_idx].to(dtype=reply_probs.dtype)
    return edge_index, edge_weight


def merge_teacher_fallback_edges(
    teacher_index: torch.Tensor,
    teacher_weight: torch.Tensor,
    candidate_index: torch.Tensor,
    teacher_row_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ablation D：Teacher 当图。没有 teacher 的目标节点回退到等权候选边。

    ``teacher_row_valid[t]=True`` 表示节点 t 至少有一条 teacher 监督；
    对这些节点只用 teacher 边。其余节点把原 ``reply_candidate`` 边以权 1 补上，
    避免测试集缺 teacher 文件时回复通道塌成空图。
    """
    device = candidate_index.device if candidate_index.numel() else teacher_index.device
    if candidate_index.numel() == 0:
        return teacher_index, teacher_weight
    src, dst = candidate_index[0], candidate_index[1]
    need = ~teacher_row_valid[dst]
    if not bool(need.any()):
        return teacher_index, teacher_weight
    fb_index = torch.stack([src[need], dst[need]], dim=0)
    fb_weight = torch.ones(int(need.sum().item()), dtype=teacher_weight.dtype, device=device)
    if teacher_index.numel() == 0:
        return fb_index, fb_weight
    return (
        torch.cat([teacher_index, fb_index], dim=1),
        torch.cat([teacher_weight, fb_weight], dim=0),
    )


class ReplyPredictor(nn.Module):
    """Student Reply Predictor：从句向量预测 soft 回复关系。

    对候选对 ``(t, j)`` 打分：

        score(t, j) = MLP([h_t ; h_j ; h_t * h_j ; h_t - h_j])

    ROOT 选项用可学习原型 ``r``，形式与历史对完全对称：

        score(t, ROOT) = MLP([h_t ; r ; h_t * r ; h_t - r])

    然后在「ROOT ∪ {j | j 是 t 的合法历史}」上做 masked softmax，得到
    ``P_S(·|t)``，形状 ``[N, N+1]``。

    只对候选边（以及每个节点的 ROOT）算 MLP，复杂度 O(E·D) 而不是 O(N²·D)，
    适合目标级图（数百节点、候选边=对话内前缀对）。
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = int(hidden_size)
        # 可学习 ROOT 原型，与历史话语向量同一空间
        self.root_emb = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.root_emb, mean=0.0, std=0.02)
        # 4 段拼接：h_t, h_j, 逐维积, 差
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, 1),
        )

    def _pair_score(self, h_t: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """``h_t, h_j: [E, D] -> [E]`` 标量打分。"""
        feat = torch.cat([h_t, h_j, h_t * h_j, h_t - h_j], dim=-1)
        return self.scorer(feat).squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        candidate_edge_index: Optional[torch.Tensor] = None,
        extra_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据句向量预测回复分布。

        Parameters
        ----------
        h:
            ``[N, D]`` 话语表示。训练时用冻结句向量（input_proj 之后、
            dropout 之前），与 Teacher 一样只看内容、不看 GNN 后的状态，
            避免和 Reply RGCN 循环依赖。
        candidate_edge_index:
            ``[2, E]``，``j -> t`` 的回复候选。None 则只保留严格下三角
            （``j < t``），用于节点已按时间排序的图。
        extra_valid:
            可选额外掩码，见 ``apply_causal_reply_mask``。

        Returns
        -------
        probs : [N, N+1]
            ``P_S(j|t)``，每行和为 1。
        logits : [N, N+1]
            掩码后的打分（非法位置为 -inf），供 KD 做温度 softmax。
        valid_hist : [N, N]
            历史节点合法掩码，建 Soft Graph 时复用。
        """
        n, _ = h.shape
        device = h.device
        logits = h.new_full((n, n + 1), float("-inf"))

        # ---- ROOT：每个当前话语都打一个「不回复任何人」的分数 ----
        root = self.root_emb.unsqueeze(0).expand(n, -1)
        logits[:, 0] = self._pair_score(h, root)

        # ---- 历史对：只在候选边 (j -> t) 上打分 ----
        # 候选边来自原 TMGCN：同一 conversation 内 turn j < t，已经是因果的。
        # 目标图节点 id 不是时间序，因此这里不用节点下标下三角。
        # 注意：即使候选边为空（全是单句对话），也绝不能退回 tril，
        # 否则会在目标图上把跨对话节点错误连起来。
        if candidate_edge_index is not None:
            valid_hist = candidate_edges_to_valid_matrix(candidate_edge_index, n, device)
            if candidate_edge_index.numel() > 0:
                src = candidate_edge_index[0].long()
                dst = candidate_edge_index[1].long()
                ok = (src >= 0) & (dst >= 0) & (src < n) & (dst < n) & (src != dst)
                src, dst = src[ok], dst[ok]
                if src.numel() > 0:
                    pair = self._pair_score(h[dst], h[src])
                    logits[dst, src + 1] = pair
            use_tril = False
        else:
            valid_hist = torch.tril(
                torch.ones((n, n), dtype=torch.bool, device=device), diagonal=-1
            )
            if n > 1:
                t_idx, j_idx = valid_hist.nonzero(as_tuple=True)
                logits[t_idx, j_idx + 1] = self._pair_score(h[t_idx], h[j_idx])
            use_tril = True

        logits = apply_causal_reply_mask(
            logits,
            extra_valid=valid_hist if extra_valid is None else extra_valid,
            use_index_tril=use_tril,
        )

        probs = masked_softmax(logits, dim=-1)
        return probs, logits, valid_hist


class ReplyKDLoss(nn.Module):
    """LLM Teacher → Student 的回复结构蒸馏损失。

    Teacher 给出 soft 分布 ``P_T``（含 ROOT），Student 给出 ``P_S``。

        L_reply_KD = T^2 * KL(P_T || P_S)

    实现细节：
    - 只在 Teacher 标注过的行、以及 Teacher 的 support（ROOT + 该话语的
      历史）上计算，Student 先在同一 support 上重新 softmax，保证两边
      是同一单纯形上的分布。
    - 温度 ``T`` 作用在 Student logits 上；Teacher 已是 soft 分数，再做
      ``p^{1/T}`` 平滑后归一化（Hinton KD 的标准做法）。
    - ``reduction='batchmean'``：对有效话语条数求平均。
    """

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        student_logits : [N, N+1]  掩码后的 Student 打分
        teacher_probs  : [N, N+1]  Teacher 概率（未标位置可为 0）
        teacher_mask   : [N, N+1]  True=该位置属于 Teacher support
        """
        if student_logits.numel() == 0:
            return student_logits.new_zeros(())
        mask = teacher_mask.to(dtype=torch.bool, device=student_logits.device)
        valid_rows = mask.any(dim=-1)
        if not bool(valid_rows.any()):
            return student_logits.new_zeros(())

        tau = max(self.temperature, 1e-6)
        s_logits = student_logits[valid_rows] / tau
        row_mask = mask[valid_rows]
        # Student 只在 Teacher 给过的选项上归一化
        s_logits = s_logits.masked_fill(~row_mask, float("-inf"))
        log_p_s = F.log_softmax(s_logits, dim=-1)

        p_t = teacher_probs[valid_rows].clamp_min(0.0) * row_mask.to(student_logits.dtype)
        # 温度平滑 Teacher：p^{1/T} / Z
        p_t = p_t.clamp_min(1e-8).pow(1.0 / tau)
        p_t = p_t * row_mask.to(p_t.dtype)
        z = p_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        p_t = p_t / z

        # F.kl_div(log_student, teacher) = KL(teacher || student)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
        return (tau ** 2) * kl
