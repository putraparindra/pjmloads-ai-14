"""Tuning clustering profil beban harian: pra-pemrosesan x algoritma x k.
Jalankan dari ROOT repo: python cluster_tuning/cluster_tune.py PJME   (hasil: cluster_tuning/cluster_tune_PJME.json)
Aturan seleksi (ditetapkan SEBELUM melihat hasil):
  skor = rata-rata Silhouette di dua ruang pembanding netral (S1 = profil distandarkan per jam, S2 = bentuk profil, z-score per hari)
  syarat = stabilitas (ARI antar-subsampel) >= 0.8 dan cluster terkecil >= 5% hari
  pilih konfigurasi dengan skor tertinggi yang memenuhi syarat.
"""
import os, sys, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))  # cells_code.py / cells_mr.py
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances

SEED = 42
KS = [2, 3, 4, 5, 6]


def get_series(code):
    from cells_code import C
    from cells_mr import C2
    g = {}
    for k in ["setup", "load", "features"]: exec(C[k].replace("plt.show()", "pass"), g)
    for k in ["mr_setup", "mr_clean", "mr_funcs"]: exec(C2[k], g)
    y, lg = g["clean_region"](code)
    return y


def daily_matrix(y):
    cnt = y.groupby(y.index.normalize()).count(); good = cnt[cnt == 24].index
    M = y[y.index.normalize().isin(good)].values.reshape(-1, 24)
    return M, pd.to_datetime(good)


def preps(M):
    """Kembalikan dict nama -> matriks fitur."""
    P = {}
    P["raw_MW"] = M.copy()
    P["std_per_jam (awal)"] = StandardScaler().fit_transform(M)
    P["robust_per_jam"] = RobustScaler().fit_transform(M)
    P["minmax_per_jam"] = MinMaxScaler().fit_transform(M)
    P["log_lalu_std"] = StandardScaler().fit_transform(np.log(M))
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True)
    P["bentuk_z_per_hari"] = (M - mu) / sd
    P["bentuk_rasio_rata2"] = M / mu
    lvl = StandardScaler().fit_transform(mu)
    P["bentuk_z + level"] = np.hstack([P["bentuk_z_per_hari"], lvl * 2.0])
    S = P["std_per_jam (awal)"]
    P["pca_95%_var"] = PCA(0.95, random_state=SEED).fit_transform(S)
    P["pca_3_komponen"] = PCA(3, random_state=SEED).fit_transform(S)
    return P


def make(alg, k, seed=SEED):
    if alg == "kmeans": return KMeans(k, n_init=10, random_state=seed)
    if alg == "gmm_full": return GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-4)
    if alg == "gmm_diag": return GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed)
    if alg == "agglo_ward": return AgglomerativeClustering(k, linkage="ward")
    if alg == "agglo_average": return AgglomerativeClustering(k, linkage="average")
    if alg == "spectral": return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=seed, assign_labels="kmeans")
    raise ValueError(alg)


def fit_labels(alg, k, X, seed=SEED):
    m = make(alg, k, seed)
    return m.fit_predict(X)


def stability(alg, k, X, n_rep=5):
    rs = np.random.RandomState(0); n = len(X); out = []
    for r in range(n_rep):
        a = rs.choice(n, int(0.7 * n), replace=False); b = rs.choice(n, int(0.7 * n), replace=False)
        common = np.intersect1d(a, b)
        la = fit_labels(alg, k, X[a], seed=r); lb = fit_labels(alg, k, X[b], seed=r + 100)
        ia = {j: i for i, j in enumerate(a)}; ib = {j: i for i, j in enumerate(b)}
        out.append(adjusted_rand_score([la[ia[j]] for j in common], [lb[ib[j]] for j in common]))
    return float(np.mean(out))


def recon(M, lab):
    mu = M.mean(0); sst = ((M - mu) ** 2).sum(); sse = 0.0
    for c in np.unique(lab): sse += ((M[lab == c] - M[lab == c].mean(0)) ** 2).sum()
    return float(1 - sse / sst), float(np.sqrt(sse / M.size))


def tune(code, algs_full, algs_spectral_preps, verbose=True, subset=None):
    y = get_series(code); M, dts = daily_matrix(y); n = len(M)
    P = preps(M)
    S1 = P["std_per_jam (awal)"]; S2 = P["bentuk_z_per_hari"]
    D1 = pairwise_distances(S1).astype(np.float32); D2 = pairwise_distances(S2).astype(np.float32)
    wk = (dts.dayofweek >= 5).astype(int); mo = dts.month.values
    rows = []; t0 = time.time()
    for pn, X in P.items():
        if subset and pn not in subset: continue
        Dm = pairwise_distances(X).astype(np.float32)
        for alg in algs_full:
            if alg == "spectral" and pn not in algs_spectral_preps: continue
            for k in KS:
                try:
                    lab = fit_labels(alg, k, X)
                    if len(np.unique(lab)) < 2: continue
                    s_own = silhouette_score(Dm, lab, metric="precomputed"); s1 = silhouette_score(D1, lab, metric="precomputed"); s2 = silhouette_score(D2, lab, metric="precomputed")
                    db = davies_bouldin_score(X, lab); ch = calinski_harabasz_score(X, lab)
                    stb = stability(alg, k, X)
                    r2, rm = recon(M, lab)
                    rows.append(dict(prep=pn, alg=alg, k=k, sil_own=float(s_own), sil_S1=float(s1), sil_S2=float(s2), score=float((s1 + s2) / 2), db=float(db), ch=float(ch), stab=stb,
                                     min_frac=float(np.bincount(lab).min() / n), recon_R2=r2, recon_RMSE_MW=rm,
                                     nmi_month=float(normalized_mutual_info_score(mo, lab)), nmi_weekend=float(normalized_mutual_info_score(wk, lab))))
                    if verbose: print(code, pn, alg, k, "score %.3f stab %.2f minfrac %.2f R2 %.3f" % (rows[-1]["score"], stb, rows[-1]["min_frac"], r2), "t=%ds" % (time.time() - t0), flush=True)
                except Exception as e:
                    print("GAGAL", code, pn, alg, k, repr(e)[:100], flush=True)
    return rows, M, dts


if __name__ == "__main__":
    code = sys.argv[1]
    rows, M, dts = tune(code, ["kmeans", "gmm_full", "gmm_diag", "agglo_ward", "agglo_average", "spectral"], {"std_per_jam (awal)", "bentuk_z_per_hari", "pca_95%_var"})
    json.dump(rows, open(os.path.join(HERE, f"cluster_tune_{code}.json"), "w"))
    print("SELESAI", code, len(rows))
