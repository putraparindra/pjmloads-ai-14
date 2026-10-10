"""Sel notebook untuk tuning clustering (menggantikan cluster_prep / cluster_fit lama). C3 = dict kode sel."""
C3 = {}

C3["cluster_prep"] = r'''# Satu baris = satu hari; 24 kolom = beban tiap jam (MW). Clustering bersifat DESKRIPTIF (memakai data hari itu sendiri).
daily = y.groupby(y.index.date).apply(lambda s: s.values)
M = np.vstack(daily.values); dates = pd.to_datetime(daily.index)
print("matriks profil harian:", M.shape)
'''

C3["cluster_funcs"] = r'''# ---- Fungsi bantu untuk tuning clustering: pra-pemrosesan x algoritma x k ----
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score, pairwise_distances

def cluster_preps(M):
    """Variasi pra-pemrosesan profil harian (24 nilai per hari)."""
    P = {}
    P["raw_MW"] = M.copy()                                                  # tanpa penskalaan (satuan MW)
    P["std_per_jam (awal)"] = StandardScaler().fit_transform(M)             # konfigurasi awal proyek
    P["robust_per_jam"] = RobustScaler().fit_transform(M)
    P["minmax_per_jam"] = MinMaxScaler().fit_transform(M)
    P["log_lalu_std"] = StandardScaler().fit_transform(np.log(M))
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True)
    P["bentuk_z_per_hari"] = (M - mu) / sd                                  # hanya bentuk profil (level dibuang)
    P["bentuk_rasio_rata2"] = M / mu
    P["bentuk_z + level"] = np.hstack([P["bentuk_z_per_hari"], StandardScaler().fit_transform(mu) * 2.0])
    P["pca_95%_var"] = PCA(0.95, random_state=SEED).fit_transform(P["std_per_jam (awal)"])
    P["pca_3_komponen"] = PCA(3, random_state=SEED).fit_transform(P["std_per_jam (awal)"])
    return P

def cluster_make(alg, k, seed=SEED):
    if alg == "kmeans": return KMeans(k, n_init=10, random_state=seed)
    if alg == "gmm_full": return GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-4)
    if alg == "gmm_diag": return GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed)
    if alg == "agglo_ward": return AgglomerativeClustering(k, linkage="ward")
    if alg == "agglo_average": return AgglomerativeClustering(k, linkage="average")
    if alg == "spectral": return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=seed, assign_labels="kmeans")
    raise ValueError(alg)

def cluster_stability(alg, k, X, n_rep=5):
    """Stabilitas = rata-rata Adjusted Rand Index antara dua model yang dilatih pada dua subsampel 70% berbeda (dibandingkan pada hari yang sama)."""
    rs = np.random.RandomState(0); n = len(X); out = []
    for r in range(n_rep):
        a = rs.choice(n, int(0.7 * n), replace=False); b = rs.choice(n, int(0.7 * n), replace=False); common = np.intersect1d(a, b)
        la = cluster_make(alg, k, r).fit_predict(X[a]); lb = cluster_make(alg, k, r + 100).fit_predict(X[b])
        ia = {j: i for i, j in enumerate(a)}; ib = {j: i for i, j in enumerate(b)}
        out.append(adjusted_rand_score([la[ia[j]] for j in common], [lb[ib[j]] for j in common]))
    return float(np.mean(out))

def cluster_recon(M, lab):
    """R2 rekonstruksi: seberapa besar variasi profil harian (dalam MW) yang dijelaskan oleh profil rata-rata tiap cluster."""
    mu = M.mean(0); sst = ((M - mu) ** 2).sum(); sse = sum(((M[lab == c] - M[lab == c].mean(0)) ** 2).sum() for c in np.unique(lab))
    return float(1 - sse / sst), float(np.sqrt(sse / M.size))

def cluster_grid(M, dts, P, plan, ks=(2, 3, 4, 5, 6), verbose=True):
    """plan = daftar (nama_prep, algoritma, daftar_k). Metrik:
       sil_S1 / sil_S2 = Silhouette pada dua ruang pembanding netral (S1 = profil distandarkan per jam, S2 = bentuk profil per hari);
       score = rata-rata sil_S1 dan sil_S2; stab = stabilitas (ARI); recon_R2 = R2 rekonstruksi profil dalam MW."""
    n = len(M); D1 = pairwise_distances(P["std_per_jam (awal)"]).astype(np.float32); D2 = pairwise_distances(P["bentuk_z_per_hari"]).astype(np.float32)
    wk = (dts.dayofweek >= 5).astype(int); mo = dts.month.values; rows = []; t0 = time.time()
    for pn, alg, kk in plan:
        X = P[pn]; Dm = pairwise_distances(X).astype(np.float32)
        for k in kk:
            lab = cluster_make(alg, k).fit_predict(X)
            if len(np.unique(lab)) < 2: continue
            s1 = silhouette_score(D1, lab, metric="precomputed"); s2 = silhouette_score(D2, lab, metric="precomputed"); r2, rm = cluster_recon(M, lab)
            rows.append(dict(prep=pn, alg=alg, k=k, sil_own=float(silhouette_score(Dm, lab, metric="precomputed")), sil_S1=float(s1), sil_S2=float(s2), score=float((s1 + s2) / 2),
                             db=float(davies_bouldin_score(X, lab)), ch=float(calinski_harabasz_score(X, lab)), stab=cluster_stability(alg, k, X), min_frac=float(np.bincount(lab).min() / n),
                             recon_R2=r2, recon_RMSE_MW=rm, nmi_month=float(normalized_mutual_info_score(mo, lab)), nmi_weekend=float(normalized_mutual_info_score(wk, lab))))
        if verbose: print(f"{pn:22s} {alg:14s} selesai ({time.time() - t0:.0f} dtk)")
    return pd.DataFrame(rows)

def cluster_final(M, ks=range(2, 7)):
    """Konfigurasi terpilih: K-Means pada profil dalam MW (tanpa standardisasi); k dengan Silhouette tertinggi.
       Konfigurasi awal (K-Means + standardisasi per jam) disimpan sebagai pembanding."""
    Xstd = StandardScaler().fit_transform(M); srows = []
    for k in ks:
        lb = KMeans(k, n_init=10, random_state=SEED).fit_predict(M); lb0 = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xstd)
        srows.append(dict(k=k, silhouette=float(silhouette_score(M, lb)), davies_bouldin=float(davies_bouldin_score(M, lb)), min_cluster=int(np.bincount(lb).min()),
                          silhouette_std=float(silhouette_score(Xstd, lb0)), ari_vs_std=float(adjusted_rand_score(lb, lb0))))
    kb = max(srows, key=lambda r: r["silhouette"])["k"]; lab = KMeans(kb, n_init=10, random_state=SEED).fit_predict(M)
    order = np.argsort([M[lab == c].mean() for c in range(kb)]); rm = {int(o): i for i, o in enumerate(order)}
    return srows, kb, np.array([rm[int(l)] for l in lab])
'''

C3["cluster_grid"] = r'''# ---- Tuning clustering (grid ringkas dijalankan di sini) ----
# Pra-pemrosesan: 10 variasi (K-Means) | Algoritma lain pada data standar: GMM diagonal, Agglomerative Ward (k = 2..6), Spectral (k = 2, 3)
CP = cluster_preps(M)
plan = [(pn, "kmeans", (2, 3, 4, 5, 6)) for pn in CP] + [("std_per_jam (awal)", "gmm_diag", (2, 3, 4, 5, 6)), ("std_per_jam (awal)", "agglo_ward", (2, 3, 4, 5, 6)), ("std_per_jam (awal)", "spectral", (2, 3))]
ct = cluster_grid(M, dates, CP, plan)
ok = ct[(ct.stab >= 0.8) & (ct.min_frac >= 0.05)]
best_k = ct.loc[ct.groupby(["prep", "alg"]).score.idxmax()].sort_values("score", ascending=False)
show = ["prep", "alg", "k", "score", "sil_own", "db", "stab", "min_frac", "recon_R2"]
print("\nKonfigurasi terbaik per (pra-pemrosesan, algoritma), diurutkan menurut skor ruang netral:")
print(best_k[show].round(3).to_string(index=False))
RES["cluster_tuning"] = dict(grid_ringkas=ct.round(4).to_dict("records"), n_eligible=int(len(ok)), n_total=int(len(ct)))
'''

C3["cluster_full"] = r'''# ---- Hasil grid lengkap (dijalankan terpisah dengan cluster_tune.py, +-45 menit) dan perbandingan antar-wilayah ----
import glob, os
full = pd.DataFrame(json.load(open("cluster_tuning/cluster_tune_PJME.json")))                  # 265 konfigurasi: 10 pra-pemrosesan x 6 algoritma x k = 2..6
okf = full[(full.stab >= 0.8) & (full.min_frac >= 0.05)].sort_values("score", ascending=False)
print("Grid lengkap PJME:", len(full), "konfigurasi,", len(okf), "memenuhi syarat (stabilitas >= 0,8 dan cluster terkecil >= 5% hari)")
cols = ["prep", "alg", "k", "score", "sil_own", "db", "stab", "min_frac", "recon_R2"]
print("\n10 teratas menurut aturan seleksi awal:"); print(okf[cols].head(10).round(3).to_string(index=False))
print("\nKonfigurasi awal (K-Means + standardisasi per jam):"); print(full[(full.prep == "std_per_jam (awal)") & (full.alg == "kmeans")][cols].round(3).to_string(index=False))
pairs = json.load(open("cluster_tuning/cluster_pairs_PJME.json"))
print("\nPerbandingan berpasangan pada 10 subsampel 70% (k = 2), selisih skor terhadap konfigurasi awal:")
for o in pairs: print(f"  {o['prep']:22s} {o['alg']:12s} skor {o['mean']:.3f} | selisih {o['diff_vs_kmeans_std']:+.4f} +- {o['diff_sd']:.4f} | lebih baik pada {o['wins']}/10 subsampel")
# Perbandingan 5 konfigurasi kandidat pada 10 wilayah (tiap konfigurasi memakai k terbaik menurut skor, di antara yang memenuhi syarat)
CAND = [("std_per_jam (awal)", "kmeans"), ("raw_MW", "kmeans"), ("bentuk_z + level", "kmeans"), ("std_per_jam (awal)", "agglo_ward"), ("std_per_jam (awal)", "gmm_diag")]
reg_rows = []
for f in sorted(glob.glob("cluster_tuning/cluster_tune_*.json")):
    reg = os.path.basename(f).replace("cluster_tune_", "").replace(".json", "")
    for r in json.load(open(f)): r["wilayah"] = reg; reg_rows.append(r)
rg = pd.DataFrame(reg_rows); rg = rg[[(p, a) in CAND for p, a in zip(rg.prep, rg.alg)]]
best_rg = []
for (reg, pn, alg), g in rg.groupby(["wilayah", "prep", "alg"]):
    e = g[(g.stab >= 0.8) & (g.min_frac >= 0.05)]; best_rg.append((e if len(e) else g).sort_values("score", ascending=False).iloc[0])
best_rg = pd.DataFrame(best_rg)
agg = best_rg.groupby(["prep", "alg"])[["score", "sil_own", "db", "stab", "recon_R2", "min_frac"]].mean().reindex(CAND)
print("\nRata-rata pada", best_rg.wilayah.nunique(), "wilayah:"); print(agg.round(3).to_string())
k2 = rg[rg.k == 2].pivot_table(index="wilayah", columns=["prep", "alg"], values=["score", "recon_R2", "stab", "db"])
cmp_raw = {m: int((k2[m][("raw_MW", "kmeans")] > k2[m][("std_per_jam (awal)", "kmeans")]).sum()) for m in ["score", "recon_R2", "stab"]}
cmp_raw["db_lebih_kecil"] = int((k2["db"][("raw_MW", "kmeans")] < k2["db"][("std_per_jam (awal)", "kmeans")]).sum())
print("\nK-Means tanpa standardisasi vs standardisasi (k = 2, per wilayah, dari 10): ", cmp_raw)
RES["cluster_tuning"].update(
    pjme_top10=okf[cols].head(10).round(4).to_dict("records"), pjme_awal=full[(full.prep == "std_per_jam (awal)") & (full.alg == "kmeans")][cols].round(4).to_dict("records"),
    pjme_n_total=int(len(full)), pjme_n_eligible=int(len(okf)), pairs=pairs,
    best_per_alg=full[(full.stab >= 0.8) & (full.min_frac >= 0.05)].sort_values("score", ascending=False).groupby("alg").head(1)[cols].round(4).to_dict("records"),
    best_per_prep=full[(full.stab >= 0.8) & (full.min_frac >= 0.05)].sort_values("score", ascending=False).groupby("prep").head(1)[cols].round(4).to_dict("records"),
    regions=[dict(prep=p, alg=a, **{k: float(v) for k, v in agg.loc[(p, a)].items()}) for p, a in CAND], n_regions=int(best_rg.wilayah.nunique()),
    score_by_region={reg: {f"{p}|{a}": float(v) for (p, a), v in g.set_index(["prep", "alg"]).score.items()} for reg, g in best_rg.groupby("wilayah")},
    raw_vs_std=cmp_raw)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
lab_cfg = {"std_per_jam (awal)|kmeans": "K-Means\nstandar (awal)", "raw_MW|kmeans": "K-Means\nMW asli", "bentuk_z + level|kmeans": "K-Means\nshape + level", "std_per_jam (awal)|agglo_ward": "Agglomerative\nWard", "std_per_jam (awal)|gmm_diag": "GMM\ndiagonal"}
x = np.arange(len(CAND)); w = .26
for i, (m, nm) in enumerate([("score", "Silhouette (ruang netral)"), ("sil_own", "Silhouette (ruang sendiri)"), ("recon_R2", "R² rekonstruksi profil")]):
    ax[0].bar(x + (i - 1) * w, [agg.loc[c, m] for c in CAND], w, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels([lab_cfg[f"{p}|{a}"] for p, a in CAND], fontsize=8); ax[0].set_title(f"Rata-rata {best_rg.wilayah.nunique()} wilayah (k terbaik tiap konfigurasi)"); ax[0].legend(fontsize=8)
ax[1].scatter(full.score, full.recon_R2, s=14, alpha=.35, label="semua konfigurasi PJME")
aw = full[(full.prep == "std_per_jam (awal)") & (full.alg == "kmeans")]; rw = full[(full.prep == "raw_MW") & (full.alg == "kmeans")]
ax[1].scatter(aw.score, aw.recon_R2, s=45, c="tab:orange", label="K-Means standar (awal)"); ax[1].scatter(rw.score, rw.recon_R2, s=45, c="tab:red", marker="s", label="K-Means MW asli (terpilih)")
ax[1].set_xlabel("skor Silhouette ruang netral"); ax[1].set_ylabel("R² rekonstruksi profil (MW)"); ax[1].set_title("PJME: kualitas pemisahan vs kemampuan mewakili profil"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_cluster_tuning.png", bbox_inches="tight"); plt.show()
'''

C3["cluster_fit"] = r'''# ---- Konfigurasi terpilih: K-Means pada profil dalam MW (tanpa standardisasi); k dari Silhouette tertinggi ----
srows, K_BEST, lab = cluster_final(M)
sil = pd.DataFrame(srows); print(sil.round(3).to_string(index=False))
RES["silhouette"] = sil.round(4).to_dict("records")
names = {0: "Beban rendah", 1: "Beban tinggi"} if K_BEST == 2 else {i: f"Cluster {i}" for i in range(K_BEST)}
info = pd.DataFrame({"date": dates, "cl": lab}); info["dow"] = info.date.dt.dayofweek; info["month"] = info.date.dt.month
info["weekend"] = (info.dow >= 5).astype(int); info["mean_mw"] = M.mean(1); info["peak_h"] = M.argmax(1)
hol = holidays.US(years=range(2001, 2020)); info["holiday"] = info.date.dt.date.map(lambda x: int(x in hol))
summ = info.groupby("cl").agg(n_hari=("cl", "size"), rata2_MW=("mean_mw", "mean"), persen_akhir_pekan=("weekend", "mean"),
                              persen_libur=("holiday", "mean"), jam_puncak_median=("peak_h", "median")).round(3)
print(summ.to_string())
print("\nKomposisi bulan (% hari dalam cluster):"); print((pd.crosstab(info.cl, info.month, normalize="index") * 100).round(0).astype(int).to_string())
sk = sil.loc[sil.k == K_BEST].iloc[0]
print("Silhouette k=%d: %.3f (konfigurasi awal: %.3f) | kesesuaian dengan konfigurasi awal (ARI): %.3f" % (K_BEST, sk.silhouette, sk.silhouette_std, sk.ari_vs_std))
RES["cluster"] = dict(k=K_BEST, silhouette=float(sk.silhouette), silhouette_std=float(sk.silhouette_std), ari_vs_std=float(sk.ari_vs_std), summary=summ.reset_index().to_dict("records"),
                      month_pct=(pd.crosstab(info.cl, info.month, normalize="index") * 100).round(1).to_dict("index"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for c in range(K_BEST):
    prof = M[lab == c]; ax[0].plot(prof.mean(0), lw=2, label=f"{names[c]} (n={len(prof)})"); ax[0].fill_between(range(24), np.percentile(prof, 10, 0), np.percentile(prof, 90, 0), alpha=.15)
ax[0].set_title("Profil beban tiap cluster (garis = rata-rata, area = P10-P90)"); ax[0].set_xlabel("jam"); ax[0].set_ylabel("MW"); ax[0].legend()
ax[1].plot(sil.k, sil.silhouette, "o-", label="K-Means MW asli (terpilih)"); ax[1].plot(sil.k, sil.silhouette_std, "s--", label="K-Means standar (awal)"); ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k"); ax[1].legend()
pc = PCA(2).fit_transform(M); ax[2].scatter(pc[:, 0], pc[:, 1], c=lab, s=4, cmap="coolwarm"); ax[2].set_title("Proyeksi PCA 2D (warna = cluster)")
plt.tight_layout(); plt.savefig("results/fig_cluster.png", bbox_inches="tight"); plt.show()
'''

C3["mr_cluster_tuning"] = r'''# ---- Ringkasan clustering antarwilayah (konfigurasi terpilih) ----
rows_ = [dict(wilayah=c, k=RES2["regions"][c]["cluster"]["k"], silhouette=RES2["regions"][c]["cluster"]["silhouette"], silhouette_std=RES2["regions"][c]["cluster"]["silhouette_std"],
              ari_vs_std=RES2["regions"][c]["cluster"]["ari_vs_std"]) for c in REGIONS]
cl_tab = pd.DataFrame(rows_); print(cl_tab.round(3).to_string(index=False)); RES2["cluster_summary"] = cl_tab.round(4).to_dict("records")
'''
