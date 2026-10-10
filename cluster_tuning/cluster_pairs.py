# Jalankan dari ROOT repo: python cluster_tuning/cluster_pairs.py <WILAYAH>   (lihat README)
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from cluster_tune import *
code = sys.argv[1]; K = 2
CAND = [("std_per_jam (awal)", "kmeans"), ("std_per_jam (awal)", "spectral"), ("bentuk_z + level", "kmeans"), ("bentuk_z_per_hari", "gmm_diag"), ("std_per_jam (awal)", "agglo_ward"), ("std_per_jam (awal)", "gmm_diag"), ("raw_MW", "kmeans")]
y = get_series(code); M, dts = daily_matrix(y); n = len(M); P = preps(M)
S1 = P["std_per_jam (awal)"]; S2 = P["bentuk_z_per_hari"]
rs = np.random.RandomState(1); res = {c: [] for c in CAND}
for rep in range(10):
    idx = rs.choice(n, int(0.7 * n), replace=False); D1 = pairwise_distances(S1[idx]); D2 = pairwise_distances(S2[idx])
    for (pn, alg) in CAND:
        lab = fit_labels(alg, K, P[pn][idx], seed=rep)
        res[(pn, alg)].append((silhouette_score(D1, lab, metric="precomputed") + silhouette_score(D2, lab, metric="precomputed")) / 2)
    print(code, "rep", rep, flush=True)
base = np.array(res[CAND[0]]); out = []
for c in CAND:
    v = np.array(res[c]); dlt = v - base
    out.append(dict(prep=c[0], alg=c[1], k=K, mean=float(v.mean()), sd=float(v.std(ddof=1)), diff_vs_kmeans_std=float(dlt.mean()), diff_sd=float(dlt.std(ddof=1)), wins=int((dlt > 0).sum())))
json.dump(out, open(os.path.join(HERE, f"cluster_pairs_{code}.json"), "w")); 
for o in out: print(o)
