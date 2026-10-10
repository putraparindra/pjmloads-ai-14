# ---- Sel kode perluasan multi-wilayah (dipakai build_notebook.py; juga dapat dijalankan sebagai skrip) ----
C2 = {}
C2["mr_setup"] = '''\
import os
DATA_DIR = "data_pjm"        # folder berisi <WILAYAH>_hourly.csv (kolom Datetime + beban MW)
REGIONS = ["PJME", "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "PJMW"]
EXCLUDED = {"NI": "seri berhenti Januari 2011 (cakupan wilayah berbeda dari seri lain)", "PJM_Load": "seri berhenti Januari 2002 (cakupan wilayah berbeda)"}
SPLIT_MR = "2016-01-01"      # periode uji bersama semua wilayah: 2016-01-01 s.d. 2018-08-02
LOW_FRAC, BORDER_FRAC = 0.35, 0.50
RES2 = {"split": SPLIT_MR, "excluded": EXCLUDED, "regions": {}}
os.makedirs("results/regions_out", exist_ok=True)
'''
C2["mr_clean"] = '''\
def load_raw(code):
    s = pd.read_csv(f"{DATA_DIR}/{code}_hourly.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    return s.iloc[:, 0].astype(float)

def clean_region(code):
    """Pra-proses satu wilayah: stempel ganda -> ambil pertama; jam hilang -> NaN; nilai tak masuk akal (<=0 atau < 35% median
    jam-dalam-pekan lokal) -> NaN; gap <= 6 jam diinterpolasi linear; hanya hari penuh dipakai. Mengembalikan (seri, log)."""
    s = load_raw(code); n_raw = int(len(s)); n_dup = int(s.index.duplicated().sum()); s = s[~s.index.duplicated(keep="first")]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h"); yy = s.reindex(full); n_missing = int(yy.isna().sum())
    how = yy.index.dayofweek * 24 + yy.index.hour
    med = yy.groupby(how).transform(lambda x: x.rolling(9, center=True, min_periods=3).median()); ratio = yy / med
    low = ((yy <= 0) | (ratio < LOW_FRAC)).fillna(False); border = ((ratio >= LOW_FRAC) & (ratio < BORDER_FRAC)).fillna(False)
    anomalies = [dict(t=str(i), nilai=float(yy[i]), rasio=float(ratio[i]) if np.isfinite(ratio[i]) else 0.0) for i in yy.index[low]]
    yc = yy.mask(low); na = yc.isna(); gid = (na != na.shift()).cumsum(); glen = na.groupby(gid).sum(); gfirst = na.groupby(gid).first()
    glen = glen[gfirst]; n_gaps = int(len(glen)); longest = int(glen.max()) if n_gaps else 0; n_long = int((glen > 6).sum())
    yc = yc.interpolate(limit=6)
    start = yc.index[0].ceil("D"); end = yc.index[-1] if yc.index[-1].hour == 23 else yc.index[-1].floor("D") - pd.Timedelta(hours=1)
    yc = yc[start:end]
    log = dict(rows_raw=n_raw, dup=n_dup, start=str(yc.index[0].date()), end=str(yc.index[-1].date()), hours=int(len(yc)), days=int(len(yc) // 24),
               missing_hours=n_missing, anomalies=anomalies, n_anomalies=len(anomalies), borderline=int(border.sum()), n_gaps=n_gaps, longest_gap=longest,
               gaps_over_6h=n_long, border_days={str(k): int(v) for k, v in border[border].groupby(border[border].index.date).size().sort_values(ascending=False).head(4).items()}, nan_left=int(yc.isna().sum()), mean=float(yc.mean()), mn=float(yc.min()), mx=float(yc.max()), raw_min=float(s.min()))
    return yc, log
'''
C2["mr_audit"] = '''\
# ---- Audit & pra-proses seluruh wilayah ----
CLEAN = {}
for code in REGIONS:
    CLEAN[code], lg = clean_region(code); RES2["regions"][code] = dict(audit=lg)
aud = pd.DataFrame({c: {k: v for k, v in RES2["regions"][c]["audit"].items() if k != "anomalies"} for c in REGIONS}).T
print(aud[["rows_raw", "start", "end", "dup", "missing_hours", "n_gaps", "longest_gap", "n_anomalies", "borderline", "nan_left", "days", "mean"]].round(0).to_string())
for c in REGIONS:
    for a in RES2["regions"][c]["audit"]["anomalies"][:5]: print(f"  {c}: nilai tak masuk akal dibuang -> {a['t']}  {a['nilai']:.0f} MW (rasio terhadap median lokal {a['rasio']:.2f})")
'''
C2["mr_funcs"] = '''\
def cluster_region(y):
    cnt = y.groupby(y.index.normalize()).count(); good = cnt[cnt == 24].index; Mx = y[y.index.normalize().isin(good)].values.reshape(-1, 24)
    srows, kb, lab_ = cluster_final(Mx)
    return srows, kb, lab_, Mx, pd.to_datetime(good)

def run_region(code, y, split=SPLIT_MR, n_leak=10, export=True):
    t0 = time.time(); R_ = {}; FIN = SETS["C. B + fitur saat-terbit (D-1)"]
    d_ = build_features(y); tr_, te_ = d_[d_.index < split], d_[d_.index >= split]; T_ = te_.y.values
    R_["split"] = dict(n_train=int(len(tr_)), n_test=int(len(te_)), train_start=str(tr_.index.min().date()), train_end=str(tr_.index.max().date()),
                       test_start=str(te_.index.min().date()), test_end=str(te_.index.max().date()), test_mean=float(te_.y.mean()), test_std=float(te_.y.std()))
    # baseline
    bs = {"Persistensi harian (lag24)": te_.lag24.values, "Persistensi mingguan (lag168)": te_.lag168.values,
          "Rata-rata lag24/48/168": te_[["lag24", "lag48", "lag168"]].mean(axis=1).values, "Rata-rata jam sama 7 hari": te_.wk_mean.values}
    R_["baseline"] = {k: metrics(T_, v) for k, v in bs.items()}; bb = min(R_["baseline"], key=lambda k: R_["baseline"][k]["MAE"]); R_["best_baseline"] = bb
    # penalaan: hanya di train, TimeSeriesSplit(4)
    GRID_ = {"learning_rate": [0.06], "max_iter": [400, 800], "max_leaf_nodes": [31, 63], "min_samples_leaf": [50, 100]}
    gs_ = GridSearchCV(HistGradientBoostingRegressor(random_state=SEED), GRID_, cv=TimeSeriesSplit(4), scoring="neg_mean_absolute_error", n_jobs=-1).fit(tr_[FIN], tr_.y)
    bp = gs_.best_params_; R_["tuning"] = dict(best=bp, cv_mae=float(-gs_.best_score_), grid_size=8)
    # ablasi (uji holdout, 3 set)
    ab = []
    for nm, cols in SETS.items():
        m_ = HistGradientBoostingRegressor(random_state=SEED, **bp).fit(tr_[cols], tr_.y); ab.append(dict(set=nm, n_fitur=len(cols), **metrics(T_, m_.predict(te_[cols]))))
    R_["ablation"] = ab
    # model
    hgb_ = HistGradientBoostingRegressor(random_state=SEED, **bp).fit(tr_[FIN], tr_.y); ph = hgb_.predict(te_[FIN])
    oh = lambda df: pd.get_dummies(df[FIN].astype({"hour": "category", "dow": "category", "month": "category"}), columns=["hour", "dow", "month"], dtype=float)
    Otr = oh(tr_); Ote = oh(te_).reindex(columns=Otr.columns, fill_value=0); sc_ = StandardScaler().fit(Otr); rg_ = Ridge(alpha=10).fit(sc_.transform(Otr), tr_.y)
    R_["models"] = {"HistGradientBoosting (tuned)": metrics(T_, ph), "Ridge (one-hot kalender)": metrics(T_, rg_.predict(sc_.transform(Ote)))}
    h_ = R_["models"]["HistGradientBoosting (tuned)"]; b_ = R_["baseline"][bb]
    R_["skill"] = dict(mae_reduction_pct=float(100 * (1 - h_["MAE"] / b_["MAE"])), rmse_reduction_pct=float(100 * (1 - h_["RMSE"] / b_["RMSE"])))
    # rolling-origin
    nsp = int(min(8, (len(d_) - 8760) // 4380)); rows_ = []
    for i, (a, b) in enumerate(TimeSeriesSplit(nsp, test_size=4380).split(d_)):
        A_, B_ = d_.iloc[a], d_.iloc[b]; m_ = HistGradientBoostingRegressor(random_state=SEED, **bp).fit(A_[FIN], A_.y); pm_ = m_.predict(B_[FIN]); mu_ = B_.y.mean()
        rows_.append(dict(lipatan=i + 1, uji_mulai=str(B_.index[0].date()), uji_akhir=str(B_.index[-1].date()), rel_HGB=100 * mean_absolute_error(B_.y, pm_) / mu_,
                          rel_lag24=100 * mean_absolute_error(B_.y, B_.lag24) / mu_, R2_HGB=r2_score(B_.y, pm_)))
    ro_ = pd.DataFrame(rows_)
    R_["rolling"] = dict(rel_hgb_mean=float(ro_.rel_HGB.mean()), rel_hgb_std=float(ro_.rel_HGB.std()), rel_lag24_mean=float(ro_.rel_lag24.mean()), r2_mean=float(ro_.R2_HGB.mean()),
                         hgb_wins=int((ro_.rel_HGB < ro_.rel_lag24).sum()), n=int(len(ro_)), rows=ro_.round(3).to_dict("records"))
    # galat
    E_ = te_[["y"]].copy(); E_["pred"] = ph; E_["err"] = E_.pred - E_.y; E_["ae"] = E_.err.abs(); E_["pe"] = 100 * E_.ae / E_.y; E_["dow"] = te_.dow; E_["is_hol"] = te_.is_hol; E_["month"] = te_.month; E_["hour"] = te_.hour; E_["date"] = te_.index.date
    dt_ = E_.assign(tipe=np.where(E_.is_hol == 1, "Libur federal", np.where(E_.dow >= 5, "Akhir pekan", "Hari kerja"))).groupby("tipe").pe.mean(); dd_ = E_.groupby("date").agg(mape=("pe", "mean")).sort_values("mape", ascending=False)
    R_["errors"] = dict(by_hour=[float(v) for v in E_.groupby("hour").pe.mean().values], by_month=[float(E_[E_.month == m].pe.mean()) if (E_.month == m).any() else 0.0 for m in range(1, 13)],
                        daytype={k: float(v) for k, v in dt_.items()}, bias=float(E_.err.mean()), p90=float(E_.ae.quantile(.9)), p99=float(E_.ae.quantile(.99)), max=float(E_.ae.max()),
                        worst=[dict(date=str(i), mape=float(r.mape)) for i, r in dd_.head(5).iterrows()], worst10_weekend=int(sum(pd.Timestamp(i).dayofweek >= 5 for i in dd_.head(10).index)))
    # importance
    rs_ = np.random.RandomState(SEED); sub_ = rs_.choice(len(te_), min(4000, len(te_)), replace=False)
    pi_ = permutation_importance(hgb_, te_[FIN].iloc[sub_], te_.y.iloc[sub_], n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error")
    imp_ = pd.Series(pi_.importances_mean, index=FIN).sort_values(ascending=False); R_["importance"] = {k: float(v) for k, v in imp_.head(10).items()}
    # clustering (hari lengkap saja): K-Means pada profil MW (konfigurasi hasil tuning), lihat cells_cluster.py
    srows, kb, lab_, Mx, dts = cluster_region(y)
    inf_ = pd.DataFrame({"date": dts, "cl": lab_}); inf_["weekend"] = (inf_.date.dt.dayofweek >= 5).astype(int); inf_["mean_mw"] = Mx.mean(1); inf_["peak_h"] = Mx.argmax(1); inf_["month"] = inf_.date.dt.month
    sm_ = inf_.groupby("cl").agg(n_hari=("cl", "size"), rata2_MW=("mean_mw", "mean"), persen_akhir_pekan=("weekend", "mean"), jam_puncak_median=("peak_h", "median")).round(3)
    sk_ = [r for r in srows if r["k"] == kb][0]
    R_["silhouette"] = srows; R_["cluster"] = dict(k=int(kb), silhouette=sk_["silhouette"], silhouette_std=sk_["silhouette_std"], ari_vs_std=sk_["ari_vs_std"], summary=sm_.reset_index().to_dict("records"),
                                                    month_pct=(pd.crosstab(inf_.cl, inf_.month, normalize="index") * 100).round(1).to_dict("index"))
    # uji kebocoran (perturbasi)
    rl_ = np.random.RandomState(0); ud_ = pd.to_datetime(d_.index.date).unique()[400:-5]; chg = {}
    for day in rl_.choice(ud_, n_leak, replace=False):
        yp_ = y.copy(); mk_ = (yp_.index >= day) & (yp_.index < day + pd.Timedelta(days=1)); yp_[mk_] = rl_.uniform(0.3 * y.mean(), 1.9 * y.mean(), mk_.sum())
        dp_ = build_features(yp_); rw_ = d_.index[(d_.index >= day) & (d_.index < day + pd.Timedelta(days=1))]
        df_ = (d_.loc[rw_, FIN] - dp_.reindex(rw_)[FIN]).abs().max()
        for c_ in df_[df_ > 1e-9].index: chg[c_] = chg.get(c_, 0) + 1
    R_["leak_test"] = dict(n_days=n_leak, n_features=len(FIN), changed_features=chg)
    if export:   # data web + model + fixture
        td_ = pd.DataFrame({"y": te_.y.values, "p": ph, "b": te_.lag24.values}, index=te_.index); days_ = {}
        for dd, g_ in td_.groupby(td_.index.date):
            if len(g_) == 24: days_[str(dd)] = [[int(round(v)) for v in g_[c].values] for c in ("y", "p", "b")]
        cl_ = [dict(name=f"Cluster {c}", n=int((lab_ == c).sum()), mean=[int(round(v)) for v in Mx[lab_ == c].mean(0)], p10=[int(round(v)) for v in np.percentile(Mx[lab_ == c], 10, 0)],
                    p90=[int(round(v)) for v in np.percentile(Mx[lab_ == c], 90, 0)]) for c in range(kb)]
        R_["_web"] = dict(days=days_, clusters=cl_, importance={k: round(float(v), 1) for k, v in imp_.head(10).items()})
        trees = []
        for it in hgb_._predictors:
            nd = it[0].nodes
            trees.append(dict(f=[int(x) if not lf else -1 for x, lf in zip(nd["feature_idx"], nd["is_leaf"])], t=[float(x) for x in nd["num_threshold"]], l=[int(x) for x in nd["left"]],
                              r=[int(x) for x in nd["right"]], v=[float(f"{x:.12g}") for x in nd["value"]], mg=[int(x) for x in nd["missing_go_to_left"]]))
        R_["_model"] = dict(features=FIN, base=float(hgb_._baseline_prediction.ravel()[0]), trees=trees)
        ix_ = np.random.RandomState(1).choice(len(te_), 40, replace=False)
        R_["_fixture"] = dict(X=te_[FIN].iloc[ix_].values.tolist(), y=[float(v) for v in hgb_.predict(te_[FIN].iloc[ix_])])
    R_["seconds"] = round(time.time() - t0)
    return R_
'''
C2["mr_run"] = '''\
# ---- Jalankan pipeline yang sama untuk tiap wilayah (penalaan hanya di train; uji bersama 2016-01-01 s.d. 2018-08-02) ----
for code in REGIONS:
    r_ = run_region(code, CLEAN[code], export=(code != "PJME"))        # PJME di sini hanya untuk membandingkan protokol sama; model utamanya di bagian 1-8
    RES2["regions"][code].update({k: v for k, v in r_.items() if not k.startswith("_")})
    if code != "PJME":
        json.dump(dict(web=r_["_web"], model=r_["_model"], fixture=r_["_fixture"]), open(f"results/regions_out/{code}.json", "w"), separators=(",", ":"))
    h = r_["models"]["HistGradientBoosting (tuned)"]
    print(f"{code:7s} test {r_['split']['n_test']//24:4d} hari | MAE {h['MAE']:8.1f} MW | rel {h['rel']:.2f}% | R2 {h['R2']:.3f} | vs baseline {r_['skill']['mae_reduction_pct']:.1f}% | "
          f"rolling menang {r_['rolling']['hgb_wins']}/{r_['rolling']['n']} | k={r_['cluster']['k']} | leak:{'bocor' if r_['leak_test']['changed_features'] else 'tidak ada'} | {r_['seconds']}s")
'''
C2["mr_compare"] = '''\
# ---- Perbandingan antarwilayah (protokol sama) ----
rows = []
for c in REGIONS:
    r = RES2["regions"][c]; h = r["models"]["HistGradientBoosting (tuned)"]; bb = r["baseline"][r["best_baseline"]]
    rows.append(dict(wilayah=c, rata2_MW=r["split"]["test_mean"], hari_uji=r["split"]["n_test"] // 24, MAE=h["MAE"], rel_HGB=h["rel"], rel_base=bb["rel"], R2=h["R2"], R2_base=bb["R2"],
                     turun_MAE_pct=r["skill"]["mae_reduction_pct"], rolling_menang=f"{r['rolling']['hgb_wins']}/{r['rolling']['n']}", rolling_rel=r["rolling"]["rel_hgb_mean"], k=r["cluster"]["k"], silhouette=r["cluster"]["silhouette"]))
cmp_ = pd.DataFrame(rows); print(cmp_.round(3).to_string(index=False)); RES2["compare"] = cmp_.round(4).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
o_ = cmp_.sort_values("rata2_MW", ascending=False); xx = np.arange(len(o_))
ax[0].bar(xx - .2, o_.rel_base, .4, label="baseline terbaik"); ax[0].bar(xx + .2, o_.rel_HGB, .4, label="HGB"); ax[0].set_xticks(xx); ax[0].set_xticklabels(o_.wilayah, rotation=30); ax[0].set_ylabel("relative error (%)"); ax[0].set_title("Relative error per wilayah (urut: beban rata-rata besar -> kecil)"); ax[0].legend()
ax[1].scatter(cmp_.rata2_MW, cmp_.rel_HGB); [ax[1].annotate(w, (m, e), fontsize=9, xytext=(3, 3), textcoords="offset points") for w, m, e in zip(cmp_.wilayah, cmp_.rata2_MW, cmp_.rel_HGB)]
ax[1].set_xscale("log"); ax[1].set_xlabel("beban rata-rata test (MW, skala log)"); ax[1].set_ylabel("relative error HGB (%)"); ax[1].set_title("Relative error vs ukuran wilayah")
plt.tight_layout(); plt.savefig("results/fig_regions.png", bbox_inches="tight"); plt.show()
cor_ = float(np.corrcoef(np.log(cmp_.rata2_MW), cmp_.rel_HGB)[0, 1]); print("korelasi log(beban rata-rata) vs galat relatif HGB: %.2f (n=%d wilayah; indikatif saja)" % (cor_, len(cmp_))); RES2["size_corr"] = cor_
'''
C2["mr_save"] = '''\
json.dump(RES2, open("results/results_regions.json", "w"), indent=1, default=float); print("results/results_regions.json tersimpan")
'''
from cells_cluster import C3
C2["mr_cluster_tuning"] = C3["mr_cluster_tuning"]
ORDER2 = ["mr_setup", "mr_clean", "mr_audit", "mr_funcs", "mr_run", "mr_compare", "mr_cluster_tuning", "mr_save"]
