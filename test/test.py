import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

def read_sequence_from_file(path: str) -> str:
    """
    从文本文件读取 DNA 序列：
    - 忽略空行和首尾空白
    - 返回纯净的字符串（只包含 A/C/G/T）
    """
    with open(path, "r") as f:
        # 读取每一行，去掉首尾空白，过滤空行
        lines = [line.strip() for line in f if line.strip()]
    # 将多行拼接成一个连续序列
    return "".join(lines)
# ———— 1. 反向互补 ————
def revcomp(seq: str) -> str:
    comp = {'A':'T','T':'A','C':'G','G':'C'}
    return ''.join(comp[c] for c in reversed(seq))

# ———— 2. 双链 LSH 索引 ————
def build_bi_strand_index(reference: str, k: int, d: int):
    masks = list(combinations(range(k), k - d))
    rc_ref = revcomp(reference)
    idx_pos, idx_neg = [], []
    n = len(reference)
    for mask in masks:
        mpos, mneg = defaultdict(list), defaultdict(list)
        for i in range(n - k + 1):
            # 正向 k-mer
            kp = reference[i:i+k]
            keyp = hash(tuple(kp[p] for p in mask))
            mpos[keyp].append(i)
            # 倒向 k-mer：rc_ref 的片段正好是 reference[i:i+k] 的反向互补
            kn = rc_ref[n - (i+k) : n - i]
            keyn = hash(tuple(kn[p] for p in mask))
            mneg[keyn].append(i)
        idx_pos.append(mpos)
        idx_neg.append(mneg)
    return masks, idx_pos, idx_neg

# ———— 3. 双链种子匹配 ————
def match_bi_strand(query: str, k: int, d: int, masks, idx_pos, idx_neg):
    anchors = []  # (q0, r0, strand)
    for q in range(len(query) - k + 1):
        kmer = query[q:q+k]
        for mask, mpos, mneg in zip(masks, idx_pos, idx_neg):
            key = hash(tuple(kmer[p] for p in mask))
            for r in mpos.get(key, []):
                anchors.append((q, r, +1))
            for r in mneg.get(key, []):
                anchors.append((q, r, -1))
    return list(set(anchors))

# ———— 4. 带宽 Smith–Waterman（简洁版） ————
def banded_sw_simple(ref: str, query: str,
                     r0: int, q0: int,
                     W: int=50, band: int=10,
                     match: int=2, mismatch: int=-5, gap: int=-10):
    # 定义窗口
    q_lo = max(0, q0 - W); q_hi = min(len(query),  q0 + W + 1)
    r_lo = max(0, r0 - W); r_hi = min(len(ref),    r0 + W + 1)
    n, m = q_hi - q_lo, r_hi - r_lo

    # DP 表
    H = np.zeros((n+1, m+1), dtype=int)
    best, bi, bj = 0, 0, 0

    # 填表
    for i in range(1, n+1):
        j0 = max(1, i - band); j1 = min(m, i + band)
        for j in range(j0, j1+1):
            s = match if query[q_lo+i-1] == ref[r_lo+j-1] else mismatch
            v = max(0,
                    H[i-1, j-1] + s,
                    H[i-1, j]   + gap,
                    H[i,   j-1] + gap)
            H[i, j] = v
            if v > best:
                best, bi, bj = int(v), i, j

    # 回溯
    i, j = bi, bj
    while i>0 and j>0 and H[i, j] > 0:
        if H[i, j] == H[i-1, j-1] + (match if query[q_lo+i-1]==ref[r_lo+j-1] else mismatch):
            i, j = i-1, j-1
        elif H[i, j] == H[i-1, j] + gap:
            i -= 1
        else:
            j -= 1

    # 全局坐标
    q_start = q_lo + i
    q_end   = q_lo + bi - 1
    r_start = r_lo + j
    r_end   = r_lo + bj - 1
    return (q_start, q_end, r_start, r_end, best), H, (bi, bj), (q_lo, r_lo)

# ———— 5. 主流程 ————
# 读序列
reference = read_sequence_from_file("ref1.txt")
query     = read_sequence_from_file("que1.txt")
reference_rc = revcomp(reference)

# 构建索引 & 匹配
k, d = 11, 2
masks, idx_pos, idx_neg = build_bi_strand_index(reference, k, d)
anchors = match_bi_strand(query, k, d, masks, idx_pos, idx_neg)

# 对每个锚点做局部对齐
fragments = []
for q0, r0, strand in anchors:
    if strand == +1:
        (qs, qe, rs, re, sc), H, pt, offs = banded_sw_simple(
            reference, query, r0, q0)
    else:
        # 倒位：用 reference_rc，对 query 也取 rc
        (qs0, qe0, rs, re, sc), H, pt, offs = banded_sw_simple(
            reference_rc, revcomp(query), 
            # 在 rc_ref 上位置对应同一个 r0
            len(reference) - (r0 + k), 
            len(query) - (q0 + k)
        )
        # 把 qs0, qe0（在 rc_query 上的坐标）映回原 query
        qs = len(query) - (q0 + k) + qs0
        qe = len(query) - (q0 + k) + qe0

    fragments.append((qs, qe, rs, re, sc, strand, H, pt, offs))

# ———— 6. 可视化示例 ————
# 找一个倒位的例子
for qs, qe, rs, re, sc, strand, H, path, (q_lo, r_lo) in fragments:
    if strand == -1:
        print("倒位片段：", qs, qe, rs, re, "score=", sc)
        plt.imshow(H, aspect='auto')
        ys, xs = zip(*path)
        plt.plot(xs, ys, 'w-')
        plt.title(f"SW Heatmap strand=-1\nq0→[{qs}:{qe}] r0→[{rs}:{re}]")
        plt.xlabel("ref window idx"); plt.ylabel("query window idx")
        plt.show()
        print("query片段:", query[qs:qe+1])
        print("ref 片段:", reference[rs:re+1])
        break

# ———— 7. 片段合并（贪心链式） ————
def chain_fragments_greedy(fragments):
    frags = sorted(fragments, key=lambda x: (x[0], -x[4]))
    chained = []
    last_q_end = -1
    for qs, qe, rs, re, sc, *_ in frags:
        if qs > last_q_end:
            chained.append((qs, qe, rs, re))
            last_q_end = qe
    return chained

chain = chain_fragments_greedy(fragments)
print("最终合并：", chain)
