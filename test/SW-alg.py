import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smith_waterman(A: str, B: str,
                   match: int = 1,
                   mismatch: int = -1,
                   gap: int = -2):
    """
    返回：
    F       —— (m+1)×(n+1) 打分矩阵 (numpy.ndarray, int)
    trace   —— 同尺寸，0=终止 1=diag 2=up 3=left  (np.int8)
    max_sc  —— 局部比对最高分
    alnA/B  —— 最优局部对齐的两条字符串
    path    —— 最优路径坐标列表 [(row,col)...] 从 0 开始
    """
    m, n = len(A), len(B)
    F = np.zeros((m + 1, n + 1), dtype=int)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)

    max_score, max_pos = 0, (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = F[i - 1, j - 1] + (match if A[i - 1] == B[j - 1] else mismatch)
            up   = F[i - 1, j] + gap
            left = F[i, j - 1] + gap
            best = max(0, diag, up, left)

            F[i, j] = best
            if best == 0:
                trace[i, j] = 0
            elif best == diag:
                trace[i, j] = 1
            elif best == up:
                trace[i, j] = 2
            else:
                trace[i, j] = 3

            if best > max_score:
                max_score, max_pos = best, (i, j)

    # ---------- Traceback ----------
    i, j = max_pos
    alnA, alnB = [], []
    path = [(i, j)]
    while trace[i, j] and i > 0 and j > 0:
        d = trace[i, j]
        if d == 1:          # diag
            alnA.append(A[i - 1]); alnB.append(B[j - 1])
            i, j = i - 1, j - 1
        elif d == 2:        # up
            alnA.append(A[i - 1]); alnB.append('-')
            i -= 1
        else:               # left
            alnA.append('-'); alnB.append(B[j - 1])
            j -= 1
        path.append((i, j))

    return (
        F,
        trace,
        max_score,
        ''.join(reversed(alnA)),
        ''.join(reversed(alnB)),
        path[::-1]          # 从起点到终点顺序
    )


def plot_matrix(F, trace, path, A, B):
    """绘制打分矩阵 + 全部指针箭头 + 最优路径粗箭头"""
    m, n = len(A), len(B)
    fig, ax = plt.subplots(figsize=(n + 2, m + 2))
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, m + 0.5)
    ax.set_xticks(range(n + 1)); ax.set_xticklabels(['ϕ'] + list(B))
    ax.set_yticks(range(m + 1)); ax.set_yticklabels(['ϕ'] + list(A))

    # 网格
    for x in range(n + 2):
        ax.axvline(x - 0.5, color='grey', lw=0.5, zorder=0)
    for y in range(m + 2):
        ax.axhline(y - 0.5, color='grey', lw=0.5, zorder=0)
    ax.invert_yaxis()

    # 打分数字
    for i in range(m + 1):
        for j in range(n + 1):
            ax.text(j, i, F[i, j], ha='center', va='center')

    # 普通箭头 (trace)
    dir2vec = {1: (-1, -1), 2: (0, -1), 3: (-1, 0)}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d = trace[i, j]
            if d:
                dx, dy = dir2vec[d]
                ax.arrow(j, i, dx, dy,
                         head_width=0.12, length_includes_head=True)

    # 最优路径粗箭头
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        ax.arrow(c1, r1, c2 - c1, r2 - r1,
                 head_width=0.25, length_includes_head=True,
                 lw=2, color='black')

    ax.set_title("Smith–Waterman scoring matrix with traceback arrows")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ----------- 示例 -----------
    A = "GATTACA"
    B = "GCATAC"
    # A="GCTAGCTAGCTTAGCTAGCTAGATCG"
    # B="ACAGATGTACTAGCGT"
    F, trace, max_sc, alnA, alnB, path = smith_waterman(A, B)

    # 打印矩阵
    print("Scoring matrix:")
    print(pd.DataFrame(F, index=['ϕ'] + list(A), columns=['ϕ'] + list(B)))
    print("\nMaximum local alignment score:", max_sc)
    print(alnA)
    print(alnB)

    # 绘图
    plot_matrix(F, trace, path, A, B)
