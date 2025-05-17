import matplotlib.pyplot as plt
import pandas as pd
# import ace_tools as tools   # ChatGPT 环境特有

A = "GATTACA"
B = "GCATAC"
match_score, mismatch_score, gap_penalty = 1, -1, -2

# 1) 计算打分矩阵 + traceback
m, n = len(A), len(B)
F     = [[0]*(n+1) for _ in range(m+1)]
trace = [[None]*(n+1) for _ in range(m+1)]     # 'diag' / 'up' / 'left'
for i in range(1, m+1):
    F[i][0] = F[i-1][0] + gap_penalty
    trace[i][0] = 'up'
for j in range(1, n+1):
    F[0][j] = F[0][j-1] + gap_penalty
    trace[0][j] = 'left'
score   = lambda x, y: match_score if x == y else mismatch_score
for i in range(1, m+1):
    for j in range(1, n+1):
        diag = F[i-1][j-1] + score(A[i-1], B[j-1])
        up   = F[i-1][j]   + gap_penalty
        left = F[i][j-1]   + gap_penalty
        F[i][j] = max(diag, up, left)
        trace[i][j] = ['diag', 'up', 'left'][[diag, up, left].index(F[i][j])]

# 2) 回溯
i, j, path = m, n, [(m, n)]
alignA, alignB = [], []
while i > 0 or j > 0:
    if trace[i][j] == 'diag':
        alignA.append(A[i-1]); alignB.append(B[j-1]); i, j = i-1, j-1
    elif trace[i][j] == 'up':
        alignA.append(A[i-1]); alignB.append('-');      i -= 1
    else:  # left
        alignA.append('-');    alignB.append(B[j-1]);   j -= 1
    path.append((i, j))
alignA, alignB = ''.join(reversed(alignA)), ''.join(reversed(alignB))

# 3) 分数表（已显示在界面）
df = pd.DataFrame(F, index=["ϕ"]+list(A), columns=["ϕ"]+list(B))
# tools.display_dataframe_to_user("Needleman-Wunsch Scoring Matrix (with Scores)", df)




# 4) 绘图：数字 + 箭头
fig, ax = plt.subplots(figsize=(1+n, 1+m))
ax.set_xlim(-0.5, n+0.5)
ax.set_ylim(-0.5, m+0.5)
ax.set_xticks(range(n+1)); ax.set_xticklabels(["ϕ"]+list(B))
ax.set_yticks(range(m+1)); ax.set_yticklabels(["ϕ"]+list(A))
for x in range(n+2): ax.axvline(x-0.5, lw=0.5)
for y in range(m+2): ax.axhline(y-0.5, lw=0.5)
ax.invert_yaxis()                           # 让 ϕ 行在最上面
for r in range(m+1):
    for c in range(n+1):
        ax.text(c, r, str(F[r][c]), ha='center', va='center')
for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):    # 顺着 path 画箭头
    ax.arrow(c1, r1, c2-c1, r2-r1, head_width=0.15, length_includes_head=True)
ax.set_title("Scoring Matrix with Optimal Traceback Path")
plt.show()

print("Optimal Alignment:")
print(alignA)
print(alignB)
