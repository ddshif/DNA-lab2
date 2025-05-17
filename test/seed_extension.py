import matplotlib.pyplot as plt

def extend_from_seed(ref, query, r0, q0, k=11, match=2, mismatch=-3):
    """
    从种子 (q0,r0) 向左右两边延伸，BLAST 风格 X-drop（不考虑 gap 以简化）
    返回:
      q_start, q_end, r_start, r_end, total_score,
      left_scores, right_scores
    """
    # 左侧延伸
    score = 0
    best_left = 0
    left_q, left_r = q0, r0
    left_scores = [(q0, 0)]
    i, j = q0, r0
    while i > 0 and j > 0:
        i -= 1; j -= 1
        sc = match if query[i] == ref[j] else mismatch
        score += sc
        left_scores.append((i, score))
        if score > best_left:
            best_left = score
            left_q, left_r = i, j
        if score <= -10:
            break

    # 右侧延伸
    score = 0
    best_right = 0
    right_q, right_r = q0 + k - 1, r0 + k - 1
    right_scores = [(q0 + k - 1, 0)]
    i, j = q0 + k - 1, r0 + k - 1
    while i + 1 < len(query) and j + 1 < len(ref):
        i += 1; j += 1
        sc = match if query[i] == ref[j] else mismatch
        score += sc
        right_scores.append((i, score))
        if score > best_right:
            best_right = score
            right_q, right_r = i, j
        if score <= -10:
            break

    total_score = best_left + best_right
    return (left_q, right_q, left_r, right_r, total_score,
            left_scores, right_scores)

# 选第 11 个锚点（Python 下标从 0 开）
q0, r0 = anchors[20]
print(f"anchor q0={q0}, r0={r0}")

# 扩展并打印结果
k = 11
q_start, q_end, r_start, r_end, total_score, left_scores, right_scores = \
    extend_from_seed(reference, query, r0, q0, k=k)

print("Extended match range:")
print(f" query[{q_start}:{q_end}] = {query[q_start:q_end+1]}")
print(f" ref  [{r_start}:{r_end}] = {reference[r_start:r_end+1]}")
print("Total x-drop score:", total_score)

# 可视化左右延伸的累积得分
ls_pos, ls_score = zip(*left_scores)
rs_pos, rs_score = zip(*right_scores)

plt.figure(figsize=(10, 4))
plt.plot(ls_pos, ls_score, '-o', label='Left extension')
plt.plot(rs_pos, rs_score, '-o', label='Right extension')
plt.axvline(q0, color='k', linestyle='--', label='Seed start')
plt.axvline(q0 + k - 1, color='gray', linestyle='--', label='Seed end')
plt.xlabel('Query position')
plt.ylabel('Cumulative score')
plt.title(f'Seed extension around anchor q0={q0}')
plt.legend()
plt.show()