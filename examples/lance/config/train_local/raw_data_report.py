#!/usr/bin/env python3
"""Raw-PT 数据管线人类可读报告生成器（仿 debug_report 风格）。

用法（容器内）:
  cd /mnt/models/CODE/MindSpeed-MM
  NON_MEGATRON=true python3 examples/lance/config/train_local/raw_data_report.py

输出: examples/lance/config/train_local/raw_data_report_<ts>.txt
数据源: preencoded-raw 样本 + packed-raw batch + 各 manifest + encode/pack 日志
"""
import glob
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

REPO = "/mnt/models/CODE/MindSpeed-MM"
sys.path.insert(0, REPO)
os.environ.setdefault("NON_MEGATRON", "true")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PREENC = "/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-preencoded-raw"
PACKED = "/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed-raw"
QWEN = "/mnt/models/MODELS/Qwen3-0.6B"
GEN = "/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-GEN"
ENCODE_LOG = "/mnt/models/DATA_INIT/MULTI/T2I/encode_raw2_20260917.log"
PACK_LOG = "/mnt/models/DATA_INIT/MULTI/T2I/pack_raw2_20260917.log"

DETAIL_DOCS = 10          # 每任务逐条解码的文档数
STATS_SAMPLE_STRIDE = 1   # 统计扫描步长（1=全量）


def hms(sec):
    sec = int(sec)
    return "%dm%02ds" % (sec // 60, sec % 60)


def fmt_pct(x):
    return "%.2f%%" % (100.0 * x)


def main():
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
    ids_special = {
        name: tok.convert_tokens_to_ids(name)
        for name in ("<|im_start|>", "<|im_end|>", "<|vision_start|>",
                     "<|vision_end|>", "<|video_pad|>")
    }
    out = []
    W = out.append

    W("=" * 88)
    W(" Lance raw-PT 数据管线报告（Mobile-O-Pre-Train / Qwen3-0.6B / t2i+i2t 双任务）")
    W(" 生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    W(" 文档格式: 文本段=<|im_start|>text<|im_end|>；视觉段=<|vision_start|>payload<|vision_end|>")
    W("           （独立 segment；I2T CE 从目标文本段的 <|im_start|> 开始）")
    W("=" * 88)

    # ---------------- [1] 数据源与管线配置 ----------------
    W("")
    W("[1] 数据源与管线配置")
    import pyarrow.parquet as pq
    pq_files = sorted(glob.glob(GEN + "/*.parquet"))
    rows = 0
    schema_cols = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        rows += pf.metadata.num_rows
        if not schema_cols:
            schema_cols = [fl.name for fl in pf.schema_arrow]
    W("  parquet 源        : %s（%d 文件，%d 行，列=%s）" % (GEN, len(pq_files), rows, schema_cols))
    m0 = json.load(open(PREENC + "/rank-00000/manifest.json"))
    m0c = m0["config"]
    W("  LLM/tokenizer     : %s（effective_vocab=%s）" % (m0["qwen_path"], m0.get("effective_vocab_size")))
    W("  冻结 ViT          : %s" % m0["vit_path"])
    W("  冻结 VAE          : %s" % m0["vae_path"])
    W("  几何              : variant=%s latent_patch=%s max_latent=%s max_frames=%s | LLM hidden=%s head_dim=%s vit_out=%s" % (
        m0["variant"], m0c["latent_patch_size"], m0c["max_latent_size"], m0c["max_num_frames"],
        m0c["hidden_size"], m0c["head_dim"], m0c["vit_out_hidden_size"]))
    W("  任务计数          : %s（base_task_counts=%s）" % (m0.get("task_counts"), m0.get("base_task_counts")))
    W("  caption dropout   : %s（t2i 条件丢弃，CFG 无条件分支，sample_id+seed 稳定哈希）" % m0["text_cond_dropout_prob"])
    W("  emit_tasks        : %s（一行图像-文本对 → 同时产 t2i 与 i2t 样本）" % ",".join(m0.get("emit_tasks") or []))
    W("  text_format       : %s" % m0.get("text_format"))
    W("  打包              : --task-weights t2i=1,i2t=1（精确整数比例循环洗牌），44K/50K token 预算")

    # ---------------- [2] 编码阶段汇总 ----------------
    W("")
    W("[2] 编码阶段汇总（8 rank manifest）")
    total = Counter()
    enc_dur = None
    try:
        txt = open(ENCODE_LOG, errors="ignore").read()
        stamps = re.findall(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", txt, re.M)
        if len(stamps) >= 2:
            t0 = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
            enc_dur = (t1 - t0).total_seconds()
    except OSError:
        pass
    W("  rank | status    | 写入 | 失败")
    fails = 0
    for mf in sorted(glob.glob(PREENC + "/rank-*/manifest.json")):
        r = json.load(open(mf))
        rank = mf.split("/")[-2].replace("rank-", "")
        fails += r.get("failed") or 0
        W("  %4s | %-9s | %4d | %d" % (rank, r["status"], r["written"], r.get("failed") or 0))
        total[r["status"]] += r["written"]
    W("  合计: %d 样本（2646 t2i + 2646 i2t），失败 %d%s" % (
        sum(total.values()), fails,
        "，wall time %s（8×910B3）" % hms(enc_dur) if enc_dur else ""))

    # ---------------- 样本扫描（统计 + 抽样） ----------------
    t2i_files = sorted(glob.glob(PREENC + "/rank-*/t2i/sample-*.pt"))
    i2t_files = sorted(glob.glob(PREENC + "/rank-*/i2t/sample-*.pt"))
    stats = {
        "t2i": {"n": 0, "drop": 0, "doc_tok": [], "cap_tok": [], "lat_tok": [], "lat_res": Counter()},
        "i2t": {"n": 0, "doc_tok": [], "ans_tok": [], "vit_tok": [], "vit_res": Counter()},
    }
    detail = {"t2i": [], "i2t": []}
    im_s, im_e = ids_special["<|im_start|>"], ids_special["<|im_end|>"]
    vs_i, ve_i, vp_i = (ids_special["<|vision_start|>"], ids_special["<|vision_end|>"],
                        ids_special["<|video_pad|>"])

    def scan(task, files):
        for idx, f in enumerate(files):
            if idx % STATS_SAMPLE_STRIDE:
                continue
            s = torch.load(f, map_location="cpu", weights_only=False)
            ids = s.token_ids.tolist()
            n_pad = ids.count(vp_i)
            rec = stats[task]
            rec["n"] += 1
            rec["doc_tok"].append(len(ids))
            if task == "t2i":
                cap = ids[1:ids.index(im_e)] if ids and ids[0] == im_s else []
                rec["cap_tok"].append(len(cap))
                rec["lat_tok"].append(n_pad)
                side = int(round((n_pad) ** 0.5)) * 16
                rec["lat_res"]["~%dx%d" % (side, side)] += 1
                if len(cap) == 0:
                    rec["drop"] += 1
                if len(detail[task]) < DETAIL_DOCS:
                    detail[task].append((os.path.basename(f), s, len(cap), n_pad))
            else:
                ans = s.ce_labels.tolist()
                rec["ans_tok"].append(len(ans))
                rec["vit_tok"].append(n_pad)
                side = int(round((n_pad) ** 0.5)) * 28
                rec["vit_res"]["~%dx%d" % (side, side)] += 1
                if len(detail[task]) < DETAIL_DOCS:
                    detail[task].append((os.path.basename(f), s, len(ans), n_pad))

    scan("t2i", t2i_files)
    scan("i2t", i2t_files)

    def avg(v):
        return sum(v) / len(v) if v else 0.0

    # ---------------- [3] t2i 文档逐条解码 ----------------
    W("")
    W("[3] t2i 文档逐条解码（caption → VAE latent，MSE 监督）前 %d 条 + dropout 命中示例" % DETAIL_DOCS)
    shown_drop = 0
    for name, s, cap_len, n_pad in detail["t2i"]:
        ids = s.token_ids.tolist()
        text = tok.decode(ids[: ids.index(im_e) + 1]) if cap_len else "（无文本 segment）"
        mark = "[DROP: caption 已丢弃 → CFG 无条件分支]" if cap_len == 0 else ""
        if cap_len == 0:
            shown_drop += 1
        W("  --- %s %s" % (name, mark))
        W("      条件文本(%d tok): %s" % (cap_len, text if cap_len else "（空）"))
        W("      视觉目标: %d latent token ≈ 等效分辨率 ~%d×%d px（token=边/16）| MSE 监督 %d tok" % (
            n_pad, int(round(n_pad ** 0.5)) * 16, int(round(n_pad ** 0.5)) * 16, s.mse_indexes.numel()))
        segs = " | ".join("(%d,%s,%s)" % (g.length, g.attention_mode, g.modality) for g in s.segments)
        W("      段结构: %s" % segs)
    if shown_drop == 0:
        W("  （前 %d 条未出现 dropout；全量命中率见 [5]）" % DETAIL_DOCS)

    # ---------------- [4] i2t 文档逐条解码 ----------------
    W("")
    W("[4] i2t 文档逐条解码（ViT 特征 → caption，CE 监督）前 %d 条" % DETAIL_DOCS)
    for name, s, ans_len, n_pad in detail["i2t"]:
        ids = s.token_ids.tolist()
        ce_text = tok.decode(s.ce_labels.tolist())
        W("  --- %s" % name)
        W("      CE 标签(%d tok): %s" % (ans_len, ce_text))
        W("      视觉条件: %d ViT token ≈ 输入分辨率 ~%d×%d px（token=边/28, 616 编码）" % (
            n_pad, int(round(n_pad ** 0.5)) * 28, int(round(n_pad ** 0.5)) * 28))
        W("      CE 定位: 起点=%d（目标文本 <|im_start|> 处预测首 token）| vit_embeddings 宽=%s" % (
            s.ce_indexes[0].item(), tuple(s.vit_embeddings.shape)))
        segs = " | ".join("(%d,%s,%s)" % (g.length, g.attention_mode, g.modality) for g in s.segments)
        W("      段结构: %s" % segs)
        ok = ce_text.endswith("<|im_end|>") and s.ce_labels[-1].item() == im_e
        W("      验证: CE 标签=caption+<|im_end|> → %s" % ("通过" if ok else "失败!"))

    # ---------------- [5] token 构成统计 ----------------
    W("")
    W("[5] 全量 token 构成统计（扫描 %d t2i + %d i2t 样本）" % (stats["t2i"]["n"], stats["i2t"]["n"]))
    t = stats["t2i"]
    W("  t2i: 平均文档 %.0f tok | caption 平均 %.0f tok | latent 平均 %.0f tok" % (
        avg(t["doc_tok"]), avg(t["cap_tok"]), avg(t["lat_tok"])))
    W("       caption dropout 命中 %d/%d = %s（目标 10%%）" % (t["drop"], t["n"], fmt_pct(t["drop"] / t["n"])))
    W("       latent 规模分布(前8): %s" % ", ".join("%s×%d" % kv for kv in t["lat_res"].most_common(8)))
    u = stats["i2t"]
    W("  i2t: 平均文档 %.0f tok | 答案(CE)平均 %.0f tok | ViT 平均 %.0f tok" % (
        avg(u["doc_tok"]), avg(u["ans_tok"]), avg(u["vit_tok"])))
    W("       ViT 规模分布(前8): %s" % ", ".join("%s×%d" % kv for kv in u["vit_res"].most_common(8)))
    total_doc_tok = sum(t["doc_tok"]) + sum(u["doc_tok"])
    W("  预编码样本总 token: %d（t2i %d + i2t %d）" % (
        total_doc_tok, sum(t["doc_tok"]), sum(u["doc_tok"])))

    # ---------------- [6] 打包结果 ----------------
    W("")
    W("[6] 打包结果（packed-raw）")
    pm = json.load(open(PACKED + "/manifest.json"))
    W("  status=%s | batch 数=%s | 总 token=%s | 跳过样本=%s" % (
        pm["status"], pm.get("batch_count"), pm.get("total_tokens"), pm.get("skipped_sample_count")))
    batches = sorted(glob.glob(PACKED + "/batch-*.pt"))
    bstats = {"ce": 0, "mse": 0, "vit": 0, "t2i_doc": 0, "i2t_doc": 0, "tok": 0}
    sample_batches = [batches[0], batches[len(batches) // 2], batches[-1]]
    for bf in batches:
        b = torch.load(bf, map_location="cpu", weights_only=False)
        ids = b.token_ids.tolist()
        bstats["tok"] += len(ids)
        vit_set = set(b.vit_indexes.tolist()) if b.vit_indexes is not None else set()
        vae_set = set(b.vae_indexes.tolist()) if b.vae_indexes is not None else set()
        ce_n = b.ce_indexes.numel() if b.ce_indexes is not None else 0
        mse_n = b.mse_indexes.numel() if b.mse_indexes is not None else 0
        vit_n = b.vit_indexes.numel() if b.vit_indexes is not None else 0
        # video_pad 连续段 → 文档视觉 span，按 expert 归属分类
        run, prev = [], None
        runs = []
        for i, x in enumerate(ids):
            if x == vp_i:
                if prev is not None and i == prev + 1:
                    run.append(i)
                else:
                    if run:
                        runs.append(run)
                    run = [i]
                prev = i
        if run:
            runs.append(run)
        for r in runs:
            if r[0] in vae_set:
                bstats["t2i_doc"] += 1
            elif r[0] in vit_set:
                bstats["i2t_doc"] += 1
        bstats["ce"] += ce_n
        bstats["mse"] += mse_n
        bstats["vit"] += vit_n
        if bf in sample_batches:
            W("  --- %s: %d tok | 文档=%d(t2i %d + i2t %d) | CE %d tok | MSE %d tok | ViT %d tok | vit宽=%s" % (
                os.path.basename(bf), len(ids), len(runs),
                sum(1 for r in runs if r[0] in vae_set), sum(1 for r in runs if r[0] in vit_set),
                ce_n, mse_n, vit_n,
                b.vit_embeddings.shape[1] if b.vit_embeddings is not None else "-"))
    W("  全部 %d batch 合计: %d tok | t2i 文档 %d | i2t 文档 %d | CE %d tok | MSE %d tok | ViT %d tok" % (
        len(batches), bstats["tok"], bstats["t2i_doc"], bstats["i2t_doc"],
        bstats["ce"], bstats["mse"], bstats["vit"]))
    W("  平均每 batch: %.0f tok | 每步(8 rank 全量) CE %.0f tok / MSE %.0f tok" % (
        bstats["tok"] / len(batches), bstats["ce"] / len(batches) * 8, bstats["mse"] / len(batches) * 8))

    # ---------------- [7] 与旧 chat 格式对比 ----------------
    W("")
    W("[7] 与旧 chat 格式对比（同一 2646 对数据）")
    W("  旧 chat 格式/文档: <|im_start|>system\\n{4选1提示词}<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...")
    W("                    模板开销 = system 提示词(~35-60 tok) + 3 组 role 标记(~15 tok) ≈ 50-75 tok/文档")
    W("  新 raw 格式/segment: 文本独立 im_start/im_end，视觉独立 vision_start/vision_end")
    W("  旧数据           : Mobile-O-Pre-Train-packed（39 batch / 168 万 tok / 纯 t2i 单任务）")
    W("  新数据           : Mobile-O-Pre-Train-packed-raw（%s batch / %s tok / t2i+i2t 双任务 1:1）" % (
        pm.get("batch_count"), pm.get("total_tokens")))
    W("  注意: token 总量增加主要来自任务数翻倍（i2t 新增 ViT+caption），单文档开销反而大幅下降")

    # ---------------- [8] 附录 ----------------
    W("")
    W("[8] 附录：产物路径与复现命令")
    W("  编码日志  : %s" % ENCODE_LOG)
    W("  打包日志  : %s" % PACK_LOG)
    W("  样本目录  : %s/rank-XXXXX/{t2i,i2t}/sample-XXXXXXXX.pt" % PREENC)
    W("  manifest  : %s/rank-XXXXX/manifest.json（8 份）+ %s/manifest.json" % (PREENC, PACKED))
    W("  复现编码  : torchrun --nproc_per_node 8 --master_port 6011 scripts/prepare_lance_native_data.py \\")
    W("              --dataset-root %s --qwen-path %s \\" % (GEN, QWEN))
    W("              --vit-path /mnt/models/MODELS/Qwen2.5-VL-3B-Instruct --vae-path .../Wan2.2_VAE.pth \\")
    W("              --output %s --variant video --latent-patch-size 1 2 2 \\" % PREENC)
    W("              --max-latent-size 64 --max-num-frames 121 --emit-tasks t2i,i2t")
    W("  复现打包  : python3 scripts/pack_lance_native_data.py --input %s --output %s \\" % (PREENC, PACKED))
    W("              --variant video --llm-config %s/config.json --latent-patch-size 1 2 2 \\" % QWEN)
    W("              --max-latent-size 64 --max-num-frames 121 --task-weights t2i=1,i2t=1")
    W("")
    W("=" * 88)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPO, "examples/lance/config/train_local/raw_data_report_%s.txt" % ts)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("WROTE %s (%d lines)" % (out_path, len(out)))


if __name__ == "__main__":
    main()
