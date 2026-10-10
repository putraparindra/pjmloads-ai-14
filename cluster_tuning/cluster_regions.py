# Jalankan dari ROOT repo: python cluster_tuning/cluster_regions.py <WILAYAH>   (lihat README)
import sys, json, warnings; warnings.filterwarnings("ignore")
import cluster_tune as ct
from cluster_tune import *
PAIRS = [("std_per_jam (awal)", "kmeans"), ("raw_MW", "kmeans"), ("bentuk_z + level", "kmeans"), ("std_per_jam (awal)", "agglo_ward"), ("std_per_jam (awal)", "gmm_diag")]
for code in sys.argv[1:]:
    y = get_series(code); M, dts = daily_matrix(y); n = len(M); P = preps(M)
    S1 = P["std_per_jam (awal)"]; S2 = P["bentuk_z_per_hari"]
    D1 = pairwise_distances(S1).astype(np.float32); D2 = pairwise_distances(S2).astype(np.float32); rows = []
    wk = (dts.dayofweek >= 5).astype(int); mo = dts.month.values
    for pn, alg in PAIRS:
        X = P[pn]; Dm = pairwise_distances(X).astype(np.float32)
        for k in KS:
            lab = fit_labels(alg, k, X); s1 = silhouette_score(D1, lab, metric="precomputed"); s2 = silhouette_score(D2, lab, metric="precomputed")
            r2, rm = recon(M, lab)
            rows.append(dict(prep=pn, alg=alg, k=k, sil_own=float(silhouette_score(Dm, lab, metric="precomputed")), sil_S1=float(s1), sil_S2=float(s2), score=float((s1 + s2) / 2), db=float(davies_bouldin_score(X, lab)), ch=float(calinski_harabasz_score(X, lab)),
                             stab=stability(alg, k, X), min_frac=float(np.bincount(lab).min() / n), recon_R2=r2, recon_RMSE_MW=rm, nmi_month=float(normalized_mutual_info_score(mo, lab)), nmi_weekend=float(normalized_mutual_info_score(wk, lab))))
        print(code, pn, alg, flush=True)
    json.dump(rows, open(os.path.join(HERE, f"cluster_tune_{code}.json"), "w")); print("SELESAI", code, flush=True)
