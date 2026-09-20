#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 LANCE_DEBUG 调试产物解析成人类可读报告。

数据来源:
  debug_rank*.jsonl       计数/段结构/loss/时间/显存 (完整)
  debug_5step_<ts>.log    stdout 中的完整解码文本 (decoded text 巨行, jsonl 只存前256个id)

用法（容器内）:
    python3 examples/lance/config/train_local/debug_report.py
可选: --detail-rank 0
输出: debug_report_<ts>.txt (同目录)
"""
import argparse
import ast
import glob
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS_RE = re.compile(r"debug_5step_(\d{8})_(\d{6})\.log")
DOC_SPLIT = "<|im_start|>system"
SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:"


def load_last_run(path):
    records = []
    for line in open(path, encoding="utf-8"):
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    start_idx = 0
    for i, r in enumerate(records):
        if r.get("event") == "forward_inputs" and r.get("call") == 1:
            start_idx = i
    return records[start_idx:]


def group_docs(segments):
    docs = []
    for s in segments:
        if not docs or docs[-1]["sid"] != s["sample_id"]:
            docs.append({"sid": s["sample_id"], "segments": []})
        docs[-1]["segments"].append(s)
    return docs


def doc_text_len(doc):
    return sum(s["len"] for s in doc["segments"] if s["modality"] == "text")


def doc_vae_len(doc):
    return sum(s["len"] for s in doc["segments"] if s["modality"] == "vae")


def parse_decoded_texts(log_path, rank=0):
    """从 stdout 日志提取每次 forward 的完整解码文本（rank0 的 decoded text 巨行）。"""
    texts = []
    pat = re.compile(r"^\[LANCE-DBG\] r{}\s+decoded text: (.*)$".format(rank), re.M)
    raw = open(log_path, encoding="utf-8", errors="replace").read()
    for m in pat.finditer(raw):
        try:
            texts.append(ast.literal_eval(m.group(1)))
        except Exception:
            texts.append(m.group(1))
    return texts


def split_docs(decoded):
    """按 '<|im_start|>system' 切分成文档块。"""
    parts = decoded.split(DOC_SPLIT)
    return [p for p in parts if p.strip()]


def parse_caption(doc_text):
    """从文档块中提取 user caption 与 assistant 槽内容。"""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", doc_text, re.S)
    caption = m.group(1) if m else None
    a = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", doc_text, re.S)
    assistant = a.group(1) if a else ""
    return caption, assistant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-rank", type=int, default=0)
    args = ap.parse_args()

    jsonl_files = sorted(glob.glob(str(HERE / "debug_rank*.jsonl")))
    logs = sorted(HERE.glob("debug_5step_*.log"))
    if not jsonl_files:
        print("no debug_rank*.jsonl found")
        sys.exit(1)
    log_path = logs[-1] if logs else None
    ts = "latest"
    if log_path:
        m = TS_RE.search(log_path.name)
        if m:
            ts = "{}_{}".format(m.group(1), m.group(2))

    rank_data = {}
    for f in jsonl_files:
        rk = int(re.search(r"debug_rank(\d+)\.jsonl", f).group(1))
        rank_data[rk] = load_last_run(f)

    detail_rank = args.detail_rank
    recs = rank_data[detail_rank]
    forwards = sorted([r for r in recs if r["event"] == "forward_inputs"], key=lambda r: r["call"])
    steps = sorted([r for r in recs if r["event"] == "step"], key=lambda r: r["iteration"])

    decoded_texts = parse_decoded_texts(log_path, detail_rank) if log_path else []

    out = []
    w = out.append
    bar = "=" * 96
    w(bar)
    w("LANCE 调试报告（人类可读版）  来源: {} + debug_rank*.jsonl | 详情 rank = rank{} | 共 {} 步".format(
        log_path.name if log_path else "(无日志)", detail_rank, len(steps)))
    w(bar)

    # ---------- 0. 图例 ----------
    w("")
    w("─" * 96)
    w("0. 怎么读这份报告（先看这里）")
    w("─" * 96)
    w("""
[模型输入到底长什么样]
  8 个 NPU rank 并行，每个 rank 每步吃 1 个 packed batch（约 44,000~50,000 token 的长序列）。
  一个 packed batch 里塞了约 70 个互相独立的"文档"（= 70 条 t2i 训练样本，顺序打包）。
  每个文档 = 一段文本条件 + 一段图像 token，拼在同一个序列里：

    文本部分（画图指令/条件）                图像部分（学习目标）
    ┌───────────────────────────┐          ┌──────────────────────────────┐
    │ <|im_start|>system         │          │ 原图 → Wan2.2 VAE (f8 压缩)   │
    │  Describe the image ...    │          │  → 48 通道 latent             │
    │ <|im_start|>user           │          │  → 2x2 patch 化               │
    │  A close-up of a ...       │ ──条件──► │  → 每个token 192 维           │
    │ <|im_start|>assistant      │          │ 训练时按随机 timestep 加噪     │
    │  <|vision_start|>[图像]<|vision_end|> │ 损失 = 预测 velocity(noise-clean)│
    └───────────────────────────┘          └──────────────────────────────┘
      走 understanding 专家(冻结自Qwen的半边)    走 generation 专家(复制初始化的半边)
      因果注意力（只往前看）                     双向注意力
    像素换算: 1 个 latent token = 16x16 = 256 像素（VAE 压缩 8 倍 x patch 2x2）
    所有文档共用同一条 system 提示:
      "{}"

[注意力掩码规则（谁看得见谁）]
  1) 文本段看自己: 因果（只看前面的 token）
  2) 图像段看: 自己（双向）+ 前面的所有文本段（完整读到 caption —— 文生图的 conditioning 通路）
  3) 图像段是 noise 段: 永远不作为 KV 被其他段看到（防止目标泄漏）
  4) 文档之间完全互相不可见（packed 序列硬隔离）

[loss]
  本 run 数据全是 t2i: 没有任何文本监督（ce tokens = 0），
  唯一损失 = 图像 token 上的 flow-matching MSE。
""".format(SYSTEM_PROMPT_T2I))

    # ---------- 1. 第 1 步: 全量输入内容 ----------
    w("─" * 96)
    w("1. 输入是什么 —— 第 1 步 rank{} 的全部 {} 个文档".format(
        detail_rank, len(group_docs(forwards[0]["segments"])) if forwards else 0))
    w("─" * 96)
    f1 = forwards[0]
    docs = group_docs(f1["segments"])
    w("packed 序列总长 {} token | 文本 {} tok | 图像(latent) {} tok | MoT 路由: understanding {} / generation {}".format(
        f1["seq"], f1["text"], f1["vae"], f1["mot_und"], f1["mot_gen"]))
    dtexts = split_docs(decoded_texts[0]) if decoded_texts else []
    if len(dtexts) != len(docs):
        w("!! 警告: 日志解码文本切出 {} 个文档, 段结构有 {} 个, 以下按序号对齐取 min".format(len(dtexts), len(docs)))
    n = min(len(dtexts), len(docs))
    for di in range(n):
        doc = docs[di]
        caption, assistant = parse_caption(dtexts[di])
        vae_len = doc_vae_len(doc)
        tlen = doc_text_len(doc)
        w("")
        w("◆ 文档 #{:<3d}  {}  (文本 {} tok + 图像 {} tok)".format(di + 1, doc["sid"], tlen, vae_len))
        if caption is None:
            w("    user 描述: <解析失败, 原文片段: {!r}>".format(dtexts[di][:60]))
        elif caption.strip():
            w("    user 描述(画图指令): {}".format(caption.strip()))
        else:
            w("    user 描述: (空) —— 10% caption 条件 dropout 命中, 等价无条件生成分支(CFG 训练)")
        if "<|vision_start|>" in assistant:
            w("    assistant '回答' = 图像: {} 个 latent token ≈ {:,} 像素 (≈{}x{} 等效, 原图宽高比保留)".format(
                vae_len, vae_len * 256, int(round(vae_len ** 0.5)) * 16, int(round(vae_len ** 0.5)) * 16))
    if not decoded_texts:
        w("(日志中没有 decoded text 行 —— 需要 LANCE_DEBUG_TOKENIZER; 仅展示段结构)")
        for di, doc in enumerate(docs, 1):
            w("◆ 文档 #{:<3d} {} 文本{} 图像{}".format(di, doc["sid"], doc_text_len(doc), doc_vae_len(doc)))

    # ---------- 2. 第 2~N 步概览 ----------
    w("")
    w("─" * 96)
    w("2. 第 2~{} 步输入概览（每步换一个 packed batch；统计 + 抽样 3 条 caption）".format(len(forwards)))
    w("─" * 96)
    for i, f in enumerate(forwards):
        docs = group_docs(f["segments"])
        line = "forward#{}: 序列 {} tok | {} 个文档 | 文本 {} / 图像 {}".format(
            f["call"], f["seq"], len(docs), f["text"], f["vae"])
        if i < len(decoded_texts):
            dts = split_docs(decoded_texts[i])
            caps = [parse_caption(t)[0] or "" for t in dts]
            n_drop = sum(1 for c in caps if not c.strip())
            lens = [len(c) for c in caps if c.strip()]
            line += " | caption dropout: {}/{}".format(n_drop, len(caps))
            if lens:
                line += " | caption 长度(min/中位/max): {}/{}/{} 字符".format(
                    min(lens), int(statistics.median(lens)), max(lens))
        w(line)
        if i < len(decoded_texts):
            shown = 0
            for t in split_docs(decoded_texts[i]):
                c, _ = parse_caption(t)
                if c and c.strip() and shown < 3:
                    w("    例: {}".format(c.strip()[:90]))
                    shown += 1
        w("")

    # ---------- 3. 掩码结构 ----------
    docs1 = group_docs(forwards[0]["segments"])
    w("─" * 96)
    w("3. 注意力掩码结构（第 1 步 rank{} 实测 + 规则图示）".format(detail_rank))
    w("─" * 96)
    w("""
单文档掩码示意（每个文档都是 [文本段][图像段][段尾token] 三段式）:

                  ┌──────────┐   ┌──────────┐   ┌───────┐
                  │ 文本段    │   │ 图像段    │   │ 段尾   │
                  │ (caption) │   │ (noise)  │   │ token │
   ┌──────────┐   │          │   │          │   │       │
   │ 文本段    │   │ 因果 ▼   │   │    ×     │   │   ×   │  文本只看自己(因果)
   ├──────────┤   ├──────────┤   ├──────────┤   ├───────┤
   │ 图像段    │   │ 完整 →   │   │ 双向 ↔    │   │   ×   │  图像完整读 caption + 自己双向
   ├──────────┤   ├──────────┤   ├──────────┤   ├───────┤  (这就是"看文字画图"的通路)
   │ 段尾token │   │ 完整 →   │   │    ×     │   │ 自身 ▼ │  段尾看得到文本, 看不到图像
   └──────────┘   └──────────┘   └──────────┘   └───────┘  (noise 段永不作为 KV 外泄)
   其他 69 个文档    ×             ×              ×           文档间硬隔离

第 1 步 rank{} 实测: {} 个文档 x 每文档 5 个注意力块 = {} 块
  (因果块 140 = 每文档 2: 文本自看 + 段尾自看; 全连接块 210 = 每文档 3: 图像自看 + 图像→文本 + 段尾→文本)
掩码在实现上不是 44K x 44K 的稠密矩阵, 而是这 350 个矩形块的稀疏描述(LancePackedSequence.block_schedule)。
""".format(detail_rank, len(docs1), len(docs1) * 5))

    # ---------- 4. 时间 ----------
    w("─" * 96)
    w("4. 每步耗时")
    w("─" * 96)
    w("  step | 耗时(s) | 备注")
    elapsed = [s["elapsed_s"] for s in steps]
    for s in steps:
        note = "首步含编译/预热" if s["iteration"] == 1 else ""
        w("  {:>4d} | {:>7.2f} | {}".format(s["iteration"], s["elapsed_s"], note))
    if len(elapsed) > 1:
        steady = statistics.mean(elapsed[1:])
        w("  稳态步时: {:.2f}s ≈ {:.0f} 步/小时 (rank{}; 注意 debug 插桩含 tokenizer 解码, 约有 2 倍开销)".format(
            steady, 3600 / steady, detail_rank))
    w("  对照: 无插桩正式 run 实测约 8.8s/步 ≈ 410 步/小时")

    # ---------- 5. 显存 ----------
    w("─" * 96)
    w("5. 每步显存（rank{} 实测）".format(detail_rank))
    w("─" * 96)
    w("  step | 已分配GB | 峰值已分配GB | 已预留GB | 峰值已预留GB | 本卡HBM空闲/总量GB")
    for s in steps:
        m = s.get("memory") or {}
        w("  {:>4d} | {:>7.2f} | {:>9.2f} | {:>7.2f} | {:>9.2f} | {:>6.2f} / {:.2f}".format(
            s["iteration"],
            m.get("allocated_GB", float("nan")), m.get("max_allocated_GB", float("nan")),
            m.get("reserved_GB", float("nan")), m.get("max_reserved_GB", float("nan")),
            m.get("hbm_free_GB", float("nan")), m.get("hbm_total_GB", float("nan"))))
    w("  说明: allocated/reserved 是 PyTorch NPU 侧本 rank 的占用(8 rank 各占一卡, 各约 12G);")
    w("        HBM 是整卡 64G 视角(910B3 单卡 64G, 空闲值来自 rank{} 所在卡)".format(detail_rank))

    # ---------- 6. loss ----------
    w("─" * 96)
    w("6. 每步 loss / grad_norm / token 数")
    w("─" * 96)
    w("  step | loss(=mse) | ce | mse_tokens(8卡全局) | rank{}图像token | grad_norm".format(detail_rank))
    for s in steps:
        it = s["iteration"]
        vae_local = forwards[it - 1]["vae"] if it - 1 < len(forwards) else 0
        w("  {:>4d} | {:>10.4f} | 0.0 | {:>12.0f} | {:>8d} | {:>9.2f}".format(
            it, s["loss"], s["mse_tokens"], vae_local, s.get("grad_norm") or 0.0))
    w("  loss 只算在图像 token 上(flow-matching MSE: 预测 velocity = noise - clean);")
    w("  ce = 0 因为纯 t2i 数据没有文本答案监督; lr 在 warmup 中(前 ~7 步从 0 爬升到 1e-4)")

    # ---------- 7. 各 rank 对照 ----------
    w("─" * 96)
    w("7. 第 1 步各 rank 输入构成对照（每个 rank 拿不同的 packed batch）")
    w("─" * 96)
    w("  rank | 序列token | 文本token | 图像token | 文档数")
    for rk in sorted(rank_data):
        frs = [r for r in rank_data[rk] if r["event"] == "forward_inputs"]
        if not frs:
            continue
        f = frs[0]
        w("  {:>4d} | {:>8d} | {:>8d} | {:>8d} | {:>4d}".format(
            rk, f["seq"], f["text"], f["vae"], len(group_docs(f["segments"]))))

    # ---------- 8. 原始数据指引 ----------
    w("─" * 96)
    w("8. 原始数据在哪（想自己挖细节时）")
    w("─" * 96)
    w("""
  debug_5step_<ts>.log    完整 stdout:
    [LANCE-DBG] r0 forward#N ...   每步 token 构成
    [LANCE-DBG] r0   text token ids: [...]   前 64 个文本 token id
    [LANCE-DBG] r0   decoded text: '...'     完整解码文本(一个巨行, 本报告第 1/2 节的来源)
    [LANCE-DBG] r0   seg {...}               段结构(前 12 段)
    [LANCE-DBG] r0   attention blocks: N (causal X, full Y)
    [LANCE-DBG] r0 loss: total=... ce=... mse=...
    [LANCE-DBG] r0 step N t=...s loss=... mem alloc/max ... GB
  debug_rank{0..7}.jsonl  逐 rank 结构化记录(每步 3 类事件: forward_inputs/losses/step)
    注意: jsonl 里 text_ids 只有前 256 个 id(有意截断), 完整文本要看 log 的 decoded text 行
  字段速查: text_indexes=文本token位置, vae_indexes=latent位置, mse_indexes=算loss位置,
            MoT und/gen=两个专家各自处理的 token 数
""")

    out_path = HERE / "debug_report_{}.txt".format(ts)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("report written: {}  ({} lines)".format(out_path, len(out)))


if __name__ == "__main__":
    main()
