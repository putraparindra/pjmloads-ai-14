# ---- Kumpulan sel kode (dipakai oleh build_notebook.py; juga dapat dijalankan sebagai skrip) ----
C = {}
C["setup"] = '''\
import json, os, warnings, time
os.makedirs("results", exist_ok=True)   # semua keluaran (JSON, gambar) disimpan di folder results/
import numpy as np, pandas as pd, holidays
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})
RES = {}          # semua angka hasil dikumpulkan di sini -> results.json (dipakai laporan)
SEED = 42
'''
C["load"] = '''\
raw = pd.read_csv("data_pjm/PJME_hourly.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
print("baris mentah:", len(raw), "|", raw.index.min(), "->", raw.index.max())
print("timestamp ganda:", raw.index.duplicated().sum())
y = raw["PJME_MW"]
y = y[~y.index.duplicated(keep="first")]                      # tanggal ganda (pergantian DST) -> ambil yang pertama
full = pd.date_range(y.index.min(), y.index.max(), freq="h")
missing = full.difference(y.index)
print("jam hilang:", len(missing), "| gap terpanjang (jam):",
      int(pd.Series(missing).diff().ne(pd.Timedelta("1h")).cumsum().value_counts().max()) if len(missing) else 0)
y = y.reindex(full).interpolate(limit=6)                      # interpolasi linear hanya untuk gap kecil
y = y["2002-01-02":"2018-08-02"]                              # hanya hari penuh (00:00-23:00)
print("sisa NaN:", int(y.isna().sum()), "| jam akhir:", len(y), "| hari:", len(y)//24)
print(y.describe().round(0))
RES["data"] = dict(rows_raw=int(len(raw)), dup=int(raw.index.duplicated().sum()), missing=int(len(missing)),
                   hours=int(len(y)), days=int(len(y)//24), start=str(y.index[0].date()), end=str(y.index[-1].date()),
                   mean=float(y.mean()), std=float(y.std()), mn=float(y.min()), mx=float(y.max()))
'''
C["eda"] = '''\
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ym = y.resample("MS").mean(); ax[0].plot(ym.index, ym.values, lw=1); ax[0].set_title("Rata-rata beban bulanan (MW)")
tmp = pd.DataFrame({"y": y}); tmp["hour"] = tmp.index.hour; tmp["dow"] = tmp.index.dayofweek; tmp["season"] = tmp.index.month.map(lambda m: "Des-Feb" if m in (12,1,2) else "Jun-Agu" if m in (6,7,8) else "Lainnya")
for s, g in tmp.groupby("season"): ax[1].plot(g.groupby("hour").y.mean(), label=s)
ax[1].set_title("Profil rata-rata per jam menurut musim"); ax[1].set_xlabel("jam"); ax[1].legend()
ax[2].bar(range(7), tmp.groupby("dow").y.mean().values); ax[2].set_xticks(range(7)); ax[2].set_xticklabels(["Sen","Sel","Rab","Kam","Jum","Sab","Min"])
ax[2].set_ylim(25000, 35000); ax[2].set_title("Rata-rata per hari dalam pekan (MW)")
plt.tight_layout(); plt.savefig("results/fig_eda.png", bbox_inches="tight"); plt.show()
RES["eda"] = dict(dow_mean=[float(v) for v in tmp.groupby("dow").y.mean().values])
'''
C["cluster_prep"] = '''\
# Satu baris = satu hari; 24 kolom = beban tiap jam (MW). Clustering bersifat DESKRIPTIF (memakai data hari itu sendiri).
daily = y.groupby(y.index.date).apply(lambda s: s.values)
M = np.vstack(daily.values); dates = pd.to_datetime(daily.index)
print("matriks profil harian:", M.shape)
Xs = StandardScaler().fit_transform(M)
rows = []
for k in range(2, 7):
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)
    rows.append(dict(k=k, silhouette=silhouette_score(Xs, lab), davies_bouldin=davies_bouldin_score(Xs, lab), min_cluster=int(np.bincount(lab).min())))
sil = pd.DataFrame(rows); print(sil.round(3).to_string(index=False))
RES["silhouette"] = sil.round(4).to_dict("records")
'''
C["cluster_fit"] = '''\
K_BEST = int(sil.loc[sil.silhouette.idxmax(), "k"])          # k dipilih dari Silhouette tertinggi (bukan dari interpretasi)
km = KMeans(K_BEST, n_init=10, random_state=SEED).fit(Xs); lab = km.labels_
# Urutkan label berdasar rata-rata beban agar nama klaster konsisten (0 = terendah)
order = np.argsort([M[lab == c].mean() for c in range(K_BEST)]); remap = {int(o): i for i, o in enumerate(order)}
lab = np.array([remap[int(l)] for l in lab])
names = {0: "Beban rendah", 1: "Beban tinggi"} if K_BEST == 2 else {i: f"Klaster {i}" for i in range(K_BEST)}
info = pd.DataFrame({"date": dates, "cl": lab}); info["dow"] = info.date.dt.dayofweek; info["month"] = info.date.dt.month
info["weekend"] = (info.dow >= 5).astype(int); info["mean_mw"] = M.mean(1); info["peak_h"] = M.argmax(1)
hol = holidays.US(years=range(2001, 2020)); info["holiday"] = info.date.dt.date.map(lambda x: int(x in hol))
summ = info.groupby("cl").agg(n_hari=("cl", "size"), rata2_MW=("mean_mw", "mean"), persen_akhir_pekan=("weekend", "mean"),
                              persen_libur=("holiday", "mean"), jam_puncak_median=("peak_h", "median")).round(3)
print(summ.to_string())
print("\\nKomposisi bulan (% hari dlm klaster):"); print((pd.crosstab(info.cl, info.month, normalize="index") * 100).round(0).astype(int).to_string())
print("Silhouette k=%d: %.3f" % (K_BEST, silhouette_score(Xs, lab)))
RES["cluster"] = dict(k=K_BEST, silhouette=float(silhouette_score(Xs, lab)), summary=summ.reset_index().to_dict("records"),
                      month_pct=(pd.crosstab(info.cl, info.month, normalize="index") * 100).round(1).to_dict("index"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for c in range(K_BEST):
    prof = M[lab == c]; ax[0].plot(prof.mean(0), lw=2, label=f"{names[c]} (n={len(prof)})"); ax[0].fill_between(range(24), np.percentile(prof, 10, 0), np.percentile(prof, 90, 0), alpha=.15)
ax[0].set_title("Profil beban tiap klaster (garis = rata-rata, area = P10-P90)"); ax[0].set_xlabel("jam"); ax[0].set_ylabel("MW"); ax[0].legend()
ax[1].plot(sil.k, sil.silhouette, "o-"); ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k")
pc = PCA(2).fit_transform(Xs); ax[2].scatter(pc[:, 0], pc[:, 1], c=lab, s=4, cmap="coolwarm"); ax[2].set_title("Proyeksi PCA 2D (warna = klaster)")
plt.tight_layout(); plt.savefig("results/fig_cluster.png", bbox_inches="tight"); plt.show()
'''
C["features"] = '''\
# ---- Pembentukan feature dan target untuk prediksi day-ahead ----
# Skema: prediksi dibuat pukul 00:00 hari D untuk 24 jam hari D (day-ahead). Hanya data sampai 23:00 hari D-1 yang boleh dipakai.
# Setiap feature berbasis beban memakai lag >= 24 jam, atau nilai snapshot pada pukul 23:00 D-1.
def build_features(y):
    d = pd.DataFrame({"y": y}); ix = d.index
    d["hour"] = ix.hour; d["dow"] = ix.dayofweek; d["month"] = ix.month
    hol = holidays.US(years=range(2001, 2020)); dt = pd.Series(ix.date, index=ix)
    d["is_hol"] = dt.map(lambda x: int(x in hol)).values
    d["hol_prev"] = dt.map(lambda x: int((x - pd.Timedelta(days=1)) in hol)).values
    d["hol_next"] = dt.map(lambda x: int((x + pd.Timedelta(days=1)) in hol)).values
    for L in (24, 48, 72, 168, 336): d[f"lag{L}"] = d.y.shift(L)
    d["wk_mean"] = pd.concat([d.y.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)    # rata-rata jam yang sama 7 hari terakhir
    d["dmean_prev"] = d.y.shift(24).rolling(24).mean()
    # feature snapshot D-1: nilai pada 23:00 D-1 (indeks t-(hour+1)) dan statistik 24 jam / 7 hari yang berakhir di titik itu
    yv = d.y.values; n = len(d); pos = np.arange(n) - (d.hour.values + 1); ok = pos >= 0; p = np.clip(pos, 0, None)
    r24 = d.y.rolling(24).mean().values; r24max = d.y.rolling(24).max().values; r24min = d.y.rolling(24).min().values; r168 = d.y.rolling(168).mean().values
    for nm, off in (("last_obs", 0), ("last_obs_m3", 3), ("last_obs_m6", 6)):
        q = pos - off; d[nm] = np.where(q >= 0, yv[np.clip(q, 0, None)], np.nan)
    d["d1_mean"] = np.where(ok, r24[p], np.nan); d["d1_max"] = np.where(ok, r24max[p], np.nan); d["d1_min"] = np.where(ok, r24min[p], np.nan)
    d["w1_mean"] = np.where(ok, r168[p], np.nan); d["d1_trend"] = d.d1_mean - np.where(ok, np.roll(r24, 24)[p], np.nan)
    return d.dropna()

d = build_features(y)
CAL = ["hour", "dow", "month", "is_hol", "hol_prev", "hol_next"]
LAG = ["lag24", "lag48", "lag168", "lag336", "dmean_prev", "wk_mean"]
ISSUE = ["last_obs", "last_obs_m3", "last_obs_m6", "d1_mean", "d1_max", "d1_min", "w1_mean", "d1_trend", "lag72"]
SETS = {"A. Kalender saja": CAL, "B. Kalender + lag >=24 jam": CAL + LAG, "C. B + fitur saat-terbit (D-1)": CAL + LAG + ISSUE}
print("jumlah feature pada set final:", len(SETS["C. B + fitur saat-terbit (D-1)"]), "| baris data setelah feature dibentuk:", len(d))
'''
C["split"] = '''\
# ---- Split kronologis (tanpa shuffling) ----
SPLIT = "2015-01-01"
tr, te = d[d.index < SPLIT], d[d.index >= SPLIT]
print(f"train: {tr.index.min().date()} -> {tr.index.max().date()} ({len(tr):,} jam)")
print(f"test : {te.index.min().date()} -> {te.index.max().date()} ({len(te):,} jam) | porsi test {len(te)/len(d):.1%}")
RES["split"] = dict(n_train=int(len(tr)), n_test=int(len(te)), test_share=float(len(te)/len(d)), train_start=str(tr.index.min().date()), train_end=str(tr.index.max().date()),
                    test_start=str(te.index.min().date()), test_end=str(te.index.max().date()), test_mean=float(te.y.mean()), test_std=float(te.y.std()), n_features=len(SETS["C. B + fitur saat-terbit (D-1)"]))
def metrics(t, p):
    mae = mean_absolute_error(t, p); rm = mean_squared_error(t, p) ** .5
    return dict(MAE=float(mae), RMSE=float(rm), R2=float(r2_score(t, p)), rel=float(100 * mae / np.mean(t)))
'''
C["leak"] = '''\
# ---- Uji data leakage (perturbation test): ganti SELURUH nilai beban hari D dengan angka acak, hitung ulang semua feature,
# lalu cek apakah feature pada jam-jam hari D berubah. Jika ya, feature itu memakai informasi dari hari yang sedang diprediksi (leakage).
FINAL = SETS["C. B + fitur saat-terbit (D-1)"]
rng_l = np.random.RandomState(0)
uniq_days = pd.to_datetime(d.index.date).unique()[400:-5]
test_days = rng_l.choice(uniq_days, 25, replace=False)
changed = {}
for day in test_days:
    yp = y.copy(); msk = (yp.index >= day) & (yp.index < day + pd.Timedelta(days=1))
    yp[msk] = rng_l.uniform(10000, 60000, msk.sum())
    d_pert = build_features(yp)
    rows_ = d.index[(d.index >= day) & (d.index < day + pd.Timedelta(days=1))]
    diff = (d.loc[rows_, FINAL] - d_pert.loc[rows_, FINAL]).abs().max()
    for c_ in diff[diff > 1e-9].index: changed[c_] = changed.get(c_, 0) + 1
print("feature yang berubah saat nilai hari D diganti angka acak:", changed if changed else "TIDAK ADA -> tidak ada data leakage dari hari yang diprediksi (25 hari acak diuji)")
RES["leak_test"] = dict(n_days=25, n_features=len(FINAL), changed_features=changed)
'''
C["baseline"] = '''\
# ---- Baseline sederhana (tanpa belajar) ----
T = te.y.values
base = {"Persistensi harian (lag24)": te.lag24.values,
        "Persistensi mingguan (lag168)": te.lag168.values,
        "Rata-rata lag24/48/168": te[["lag24", "lag48", "lag168"]].mean(axis=1).values,
        "Rata-rata jam sama 7 hari": te.wk_mean.values}
RES["baseline"] = {k: metrics(T, v) for k, v in base.items()}
print(pd.DataFrame(RES["baseline"]).T.round(3).to_string())
BEST_BASE = min(RES["baseline"], key=lambda k: RES["baseline"][k]["MAE"]); print("\\nbaseline terbaik:", BEST_BASE)
'''
C["tune"] = '''\
# ---- Hyperparameter tuning HistGradientBoosting (HGB): GridSearchCV + TimeSeriesSplit(4) HANYA pada data train ----
GRID = {"learning_rate": [0.03, 0.06], "max_iter": [400, 800], "max_leaf_nodes": [31, 63], "min_samples_leaf": [50, 100]}
FINAL = SETS["C. B + fitur saat-terbit (D-1)"]
t0 = time.time()
gs = GridSearchCV(HistGradientBoostingRegressor(random_state=SEED), GRID, cv=TimeSeriesSplit(4), scoring="neg_mean_absolute_error", n_jobs=-1).fit(tr[FINAL], tr.y)
print("parameter terbaik:", gs.best_params_, "| MAE cross-validation (train):", round(-gs.best_score_, 1), f"| {time.time()-t0:.0f}s")
BP = gs.best_params_; RES["tuning"] = dict(best=BP, cv_mae=float(-gs.best_score_), grid_size=int(np.prod([len(v) for v in GRID.values()])))
'''
C["ablation"] = '''\
# ---- Ablation study: set feature mana yang layak? Dinilai dengan cross-validation kronologis DI DALAM TRAIN (independen dari test), lalu dilaporkan juga di test ----
abl = []
for nm, cols in SETS.items():
    cvm = []
    for a, b in TimeSeriesSplit(4).split(tr):
        m = HistGradientBoostingRegressor(random_state=SEED, **BP).fit(tr.iloc[a][cols], tr.iloc[a].y)
        cvm.append(100 * mean_absolute_error(tr.iloc[b].y, m.predict(tr.iloc[b][cols])) / tr.iloc[b].y.mean())
    m = HistGradientBoostingRegressor(random_state=SEED, **BP).fit(tr[cols], tr.y); mt = metrics(T, m.predict(te[cols]))
    abl.append(dict(set=nm, n_fitur=len(cols), cv_rel_pct=float(np.mean(cvm)), **mt))
abl = pd.DataFrame(abl); print(abl.round(3).to_string(index=False)); RES["ablation"] = abl.to_dict("records")
'''
C["models"] = '''\
# ---- Perbandingan model dan baseline pada feature final (set C) ----
Xtr, Xte = tr[FINAL], te[FINAL]
models = {}
hgb = HistGradientBoostingRegressor(random_state=SEED, **BP).fit(Xtr, tr.y); models["HistGradientBoosting (tuned)"] = hgb.predict(Xte)
rf = RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED).fit(Xtr, tr.y); models["Random Forest"] = rf.predict(Xte)
def onehot(df): return pd.get_dummies(df[FINAL].astype({"hour": "category", "dow": "category", "month": "category"}), columns=["hour", "dow", "month"], dtype=float)
Otr = onehot(tr); Ote = onehot(te).reindex(columns=Otr.columns, fill_value=0); sc = StandardScaler().fit(Otr)
rg = Ridge(alpha=10).fit(sc.transform(Otr), tr.y); models["Ridge (one-hot kalender)"] = rg.predict(sc.transform(Ote))
RES["models"] = {k: metrics(T, v) for k, v in models.items()}
tab = pd.DataFrame({**{f"[baseline] {k}": v for k, v in RES["baseline"].items()}, **RES["models"]}).T
print(tab.round(3).to_string())
best_base = RES["baseline"][BEST_BASE]; h = RES["models"]["HistGradientBoosting (tuned)"]
RES["skill"] = dict(mae_reduction_pct=float(100 * (1 - h["MAE"] / best_base["MAE"])), rmse_reduction_pct=float(100 * (1 - h["RMSE"] / best_base["RMSE"])))
print("\\nPenurunan MAE vs baseline terbaik (%s): %.1f%% | RMSE: %.1f%%" % (BEST_BASE, RES["skill"]["mae_reduction_pct"], RES["skill"]["rmse_reduction_pct"]))
pred = models["HistGradientBoosting (tuned)"]
'''
C["rolling"] = '''\
# ---- Evaluasi rolling-origin: 8 fold (train meluas, testing = 6 bulan berikutnya), semua data ----
rows = []
for i, (a, b) in enumerate(TimeSeriesSplit(8, test_size=4380).split(d)):
    A_, B_ = d.iloc[a], d.iloc[b]
    m = HistGradientBoostingRegressor(random_state=SEED, **BP).fit(A_[FINAL], A_.y); pm = m.predict(B_[FINAL]); mu = B_.y.mean()
    rows.append(dict(lipatan=i + 1, uji_mulai=str(B_.index[0].date()), uji_akhir=str(B_.index[-1].date()),
                     rel_HGB=100 * mean_absolute_error(B_.y, pm) / mu, rel_lag24=100 * mean_absolute_error(B_.y, B_.lag24) / mu, R2_HGB=r2_score(B_.y, pm)))
ro = pd.DataFrame(rows); print(ro.round(3).to_string(index=False))
print("\\nrata-rata: HGB %.2f%% (std %.2f) | lag24 %.2f%% | R2 %.3f" % (ro.rel_HGB.mean(), ro.rel_HGB.std(), ro.rel_lag24.mean(), ro.R2_HGB.mean()))
RES["rolling"] = dict(rel_hgb_mean=float(ro.rel_HGB.mean()), rel_hgb_std=float(ro.rel_HGB.std()), rel_lag24_mean=float(ro.rel_lag24.mean()), r2_mean=float(ro.R2_HGB.mean()),
                      hgb_wins=int((ro.rel_HGB < ro.rel_lag24).sum()), n=int(len(ro)), rows=ro.round(3).to_dict("records"))
'''
C["errors"] = '''\
# ---- Analisis error (data testing) ----
E = te[["y"]].copy(); E["pred"] = pred; E["err"] = E.pred - E.y; E["ae"] = E.err.abs(); E["pe"] = 100 * E.ae / E.y
E["hour"] = te.hour; E["dow"] = te.dow; E["month"] = te.month; E["is_hol"] = te.is_hol; E["date"] = te.index.date
by_hour = E.groupby("hour").pe.mean(); by_dow = E.groupby("dow").pe.mean(); by_month = E.groupby("month").pe.mean()
daytype = E.assign(tipe=np.where(E.is_hol == 1, "Libur federal", np.where(E.dow >= 5, "Akhir pekan", "Hari kerja"))).groupby("tipe").pe.mean()
daily_err = E.groupby("date").agg(mape=("pe", "mean"), mean_y=("y", "mean")); worst = daily_err.sort_values("mape", ascending=False).head(10)
print("MAPE per tipe hari:", daytype.round(2).to_dict()); print("MAPE per bulan:", by_month.round(2).to_dict())
print("10 hari terburuk:\\n", worst.round(2).to_string()); print("bias rata-rata (pred - aktual): %.0f MW" % E.err.mean())
wk10 = int(sum(pd.Timestamp(i).dayofweek >= 5 for i in worst.index)); print("dari 10 hari terburuk, %d jatuh pada akhir pekan" % wk10)
print("P90 error absolut: %.0f MW | P99: %.0f MW | maks: %.0f MW" % (E.ae.quantile(.9), E.ae.quantile(.99), E.ae.max()))
RES["errors"] = dict(by_hour=[float(v) for v in by_hour.values], by_dow=[float(v) for v in by_dow.values], by_month=[float(v) for v in by_month.values],
                     daytype={k: float(v) for k, v in daytype.items()}, bias=float(E.err.mean()), p90=float(E.ae.quantile(.9)), p99=float(E.ae.quantile(.99)), max=float(E.ae.max()),
                     worst=[dict(date=str(i), mape=float(r.mape)) for i, r in worst.head(5).iterrows()], worst10_weekend=wk10)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].bar(by_hour.index, by_hour.values); ax[0].set_title("MAPE per jam (%)"); ax[0].set_xlabel("jam")
ax[1].bar(by_month.index, by_month.values); ax[1].set_title("MAPE per bulan (%)"); ax[1].set_xlabel("bulan")
ax[2].hist(E.err, bins=80); ax[2].set_title("Distribusi error (MW), prediksi - aktual"); ax[2].axvline(0, c="k", lw=.8)
plt.tight_layout(); plt.savefig("results/fig_errors.png", bbox_inches="tight"); plt.show()
'''
C["plots"] = '''\
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
w0 = pd.Timestamp("2017-07-17"); sl = (te.index >= w0) & (te.index < w0 + pd.Timedelta(days=14))
ax[0].plot(te.index[sl], te.y[sl], label="Aktual", lw=1.6); ax[0].plot(te.index[sl], pred[sl], label="Prediksi HGB (day-ahead)", lw=1.2)
ax[0].plot(te.index[sl], te.lag24[sl], label="Baseline lag24", lw=.9, alpha=.6, ls="--"); ax[0].set_title("Dua pekan contoh (17-30 Jul 2017, musim panas)"); ax[0].set_ylabel("MW"); ax[0].legend(loc="upper left", fontsize=8)
ax[1].scatter(te.y, pred, s=2, alpha=.25); lo, hi = te.y.min(), te.y.max(); ax[1].plot([lo, hi], [lo, hi], "r--", lw=1); ax[1].set_xlabel("Aktual (MW)"); ax[1].set_ylabel("Prediksi (MW)"); ax[1].set_title("Aktual vs prediksi (test)")
plt.tight_layout(); plt.savefig("results/fig_forecast.png", bbox_inches="tight"); plt.show()
'''
C["importance"] = '''\
# ---- Permutation importance (sub-sampel 6.000 jam test, MAE) ----
rng = np.random.RandomState(SEED); sub = rng.choice(len(te), 6000, replace=False)
pi = permutation_importance(hgb, te[FINAL].iloc[sub], te.y.iloc[sub], n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FINAL).sort_values(ascending=False); print(imp.round(1).head(12).to_string())
RES["importance"] = {k: float(v) for k, v in imp.head(10).items()}
fig, ax = plt.subplots(figsize=(7, 4)); imp.head(12)[::-1].plot.barh(ax=ax); ax.set_title("Permutation importance (kenaikan MAE, MW)"); plt.tight_layout(); plt.savefig("results/fig_importance.png", bbox_inches="tight"); plt.show()
'''
C["export"] = '''\
# ---- Ekspor data untuk website statis (assets/data.js) ----
td = pd.DataFrame({"y": te.y.values, "p": pred, "b": te.lag24.values}, index=te.index)
days_out = {}
for dd, g_ in td.groupby(td.index.date):
    if len(g_) == 24: days_out[str(dd)] = [[int(round(v)) for v in g_[c].values] for c in ("y", "p", "b")]
cl_out = [dict(name=names[c], n=int((lab == c).sum()), mean=[int(round(v)) for v in M[lab == c].mean(0)],
               p10=[int(round(v)) for v in np.percentile(M[lab == c], 10, 0)], p90=[int(round(v)) for v in np.percentile(M[lab == c], 90, 0)]) for c in range(K_BEST)]
imp_out = {k: round(float(v), 1) for k, v in imp.head(10).items()}
WEB = dict(days=days_out, clusters=cl_out, importance=imp_out)
json.dump(RES, open("results.json", "w"), indent=1, default=float)
open("web_data.js", "w").write("window.PJM_RES=" + json.dumps(RES, default=float) + ";\\nwindow.PJM_WEB=" + json.dumps(WEB, separators=(",", ":")) + ";")
print("web_data.js:", round(len(open("web_data.js").read()) / 1e6, 2), "MB |", len(days_out), "hari uji")
'''
C["save"] = '''\
json.dump(RES, open("results/results.json", "w"), indent=1, default=float); print("results/results.json tersimpan")
'''
from cells_cluster import C3
C.update(C3)
ORDER = ["setup", "load", "features", "split", "leak", "baseline", "tune", "cluster_prep", "cluster_funcs", "cluster_grid", "eda", "models", "plots", "ablation", "rolling", "errors", "importance", "cluster_full", "cluster_fit", "save"]
if __name__ == "__main__":
    open("run_all.py", "w").write("\n".join(C[k] for k in ORDER))
