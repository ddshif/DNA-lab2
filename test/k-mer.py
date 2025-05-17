# ------------------ Seed Matching (k-mer anchors) ------------------
# Re‑run this cell if kernel was reset. 100% self‑contained.
#
# 1. build_kmer_index(reference, k)
# 2. query_kmers(query, k, index, max_mismatch=0/1, verbose=True)
#
# Works in pure Python; no external dependencies.

from collections import defaultdict

# ----- 2‑bit DNA encoding -----
DNA2BIT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def encode_kmer(seq: str) -> int:
    """Encode k‑mer into an integer using 2 bits/base."""
    code = 0
    for b in seq:
        code = (code << 2) | DNA2BIT[b]
    return code

# ----- precompute neighbours with one mismatch -----
def neighbours_one_mismatch(code: int, k: int):
    """All codes at Hamming distance 1 from given code."""
    mask = (1 << (2 * k)) - 1
    neigh = []
    for pos in range(k):
        shift = 2 * (k - pos - 1)
        old_bits = (code >> shift) & 3
        for repl in range(4):
            if repl == old_bits:
                continue
            alt = (code & ~(3 << shift)) | (repl << shift)
            neigh.append(alt & mask)
    return neigh

# ----- build index -----
def build_kmer_index(reference: str, k: int):
    """dict{encoded_kmer: [positions…]}"""
    idx = defaultdict(list)
    for i in range(len(reference) - k + 1):
        idx[encode_kmer(reference[i:i + k])].append(i)
    return idx

# ----- query side -----
def query_kmers(query: str, k: int, ref_idx,
                max_mismatch: int = 0, verbose: bool = False):
    """Return list of (q_pos, r_pos) anchors."""
    anchors = []
    for q in range(len(query) - k + 1):
        code = encode_kmer(query[q:q + k])
        # exact
        if code in ref_idx:
            anchors.extend((q, r) for r in ref_idx[code])
        # <=1 mismatch
        if max_mismatch >= 1:
            for alt in neighbours_one_mismatch(code, k):
                if alt in ref_idx:
                    anchors.extend((q, r) for r in ref_idx[alt])
    if verbose:
        print(f"Anchors found: {len(anchors)}")
    return anchors

# ----------------------- Demo -----------------------
if __name__ == "__main__":
    ref_demo = "ATGCTAGCTAGCTTTCGATCGATCGGCTAGCTA"
    qry_demo = "GCTAGCTTTCGATCG"
    K = 5

    idx_demo = build_kmer_index(ref_demo, K)
    anchors_0mm = query_kmers(qry_demo, K, idx_demo,
                               max_mismatch=0, verbose=True)
    anchors_1mm = query_kmers(qry_demo, K, idx_demo,
                               max_mismatch=1, verbose=True)

    print("First 10 exact_match anchors:", anchors_0mm[:10])
    print("First 10 ≤1_mismatch anchors:", anchors_1mm[:10])
