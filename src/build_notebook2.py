import json, nbformat as nbf
from cells_code import C
from cells_mr import C2
R = json.load(open("results/results.json")); R2 = json.load(open("results/results_regions.json"))
D, S, B, M_, AB, RO, ER, CL, SK, TU = R["data"], R["split"], R["baseline"], R["models"], R["ablation"], R["rolling"], R["errors"], R["cluster"], R["skill"], R["tuning"]
H = M_["HistGradientBoosting (tuned)"]; RFm = M_["Random Forest"]; RGm = M_["Ridge (one-hot kalender)"]
BB = B["Persistensi harian (lag24)"]
sil = {r["k"]: r["silhouette"] for r in R["silhouette"]}
f0 = lambda v: f"{v:,.0f}".replace(",", ".")
f1 = lambda v: f"{v:.1f}".replace(".", ",")
f2 = lambda v: f"{v:.2f}".replace(".", ",")
f3 = lambda v: f"{v:.3f}".replace(".", ",")
RG_ = R2["regions"]; CMP = {r["wilayah"]: r for r in R2["compare"]}; ORD = [c for c in CMP if c != "PJME"]
n_reg = len(CMP); wins_reg = sum(1 for r in CMP.values() if r["turun_MAE_pct"] > 0)
lo_c = min(CMP, key=lambda c: CMP[c]["rel_HGB"]); hi_c = max(CMP, key=lambda c: CMP[c]["rel_HGB"])
tmin = min(r["turun_MAE_pct"] for r in CMP.values()); tmax = max(r["turun_MAE_pct"] for r in CMP.values())
roll_all = sum(RG_[c]["rolling"]["hgb_wins"] for c in CMP); roll_n = sum(RG_[c]["rolling"]["n"] for c in CMP)
anom_all = [(c, a) for c in CMP for a in RG_[c]["audit"]["anomalies"]]
bord = {c: RG_[c]["audit"]["borderline"] for c in CMP if RG_[c]["audit"]["borderline"] > 0}
in_test = [c for c in bord if any(d >= R2["split"] for d in RG_[c]["audit"]["border_days"])]
leak_any = [c for c in CMP if RG_[c]["leak_test"]["changed_features"]]
corr_ = R2["size_corr"]
corr_txt = "cenderung lebih rendah pada wilayah dengan beban rata-rata lebih besar" if corr_ < -0.5 else "cenderung lebih tinggi pada wilayah dengan beban rata-rata lebih besar" if corr_ > 0.5 else "tidak menunjukkan hubungan yang jelas dengan ukuran wilayah"
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s)); code = lambda k: cells.append(nbf.v4.new_code_cell(C[k] if k in C else C2[k]))

md(f"""\
# Peramalan Beban Listrik *Day-Ahead* dan Segmentasi Profil Beban Harian
### Studi kasus: beban per jam jaringan PJM (AS): PJM East sebagai analisis utama ({D['start'][:4]}–{D['end'][:4]}), diperluas ke 9 wilayah PJM lain

**Mini Project — Artificial Intelligence untuk Teknik Elektro** &nbsp;|&nbsp; Kelompok: Alfito Putra Parindra (24/534768/TK/59290) · Muhammad Fasha (24/535041/TK/59325) · Najamudin Fawwaz Haq (24/539273/TK/59804) · Qais Faqih Basyaev (24/542181/TK/60189) · Muhammad Zeva Hardi Seswanto (24/542780/TK/60292) &nbsp;|&nbsp; Teknik Elektro, Universitas Gadjah Mada

Dua pendekatan AI pada satu data: **Clustering** (K-Means, dipilih lewat tuning *preprocessing* dan perbandingan algoritma) untuk mengelompokkan profil beban harian, dan **Regression**
(HistGradientBoosting, dibandingkan dengan Random Forest, Ridge, dan baseline sederhana) untuk meramal beban tiap jam **sehari ke depan**.

> **Perluasan multi-wilayah.** Analisis utama memakai PJM East (PJME). Pra-proses yang terdokumentasi dan pipeline yang sama dijalankan juga pada 9 wilayah lain (Bagian 3.2 dan 5.4); nama lengkap tiap kode wilayah tidak dijelaskan di sumber data sehingga ditulis sebagai kode saja.
>
> **Catatan.** Beban agregat wilayah jauh lebih teratur daripada konsumsi satu rumah tangga. Akurasi yang tinggi di sini terutama berasal dari sifat data
> (agregat, per jam, riwayat sangat berkorelasi), bukan dari model yang canggih (lihat Bagian 7).

**Cara menjalankan** (dari *root* repo; Python 3.10+ dan paket di `requirements.txt`): buka notebook ini lalu *Run All* (sekitar 40 menit pada 2 CPU).
Data dibaca dari `data_pjm/`, hasil tuning *clustering* penuh dari `cluster_tuning/` (dihasilkan terpisah oleh skrip di folder itu), dan semua keluaran (JSON, gambar) ditulis ke `results/`.
""")

md("""\
## 1. Problem Definition
Operator sistem tenaga harus menjadwalkan pembangkit sehari sebelumnya, sehingga membutuhkan **ramalan beban per jam untuk 24 jam ke depan**.
Selain itu, perencana perlu memahami **bentuk khas beban harian** (kapan puncak, seberapa tinggi, pada hari seperti apa).

Pertanyaan: (1) seberapa akurat beban per jam dapat diramal sehari ke depan hanya dari riwayat beban dan kalender, dibanding baseline sederhana?
(2) apakah hari-hari dapat dikelompokkan menjadi profil beban yang bermakna?
""")

md("""\
## 2. AI Formulation
| Tugas | Pendekatan | Input | Output | Ukuran keberhasilan |
|---|---|---|---|---|
| Segmentasi profil | **Clustering** K-Means pada profil dalam MW tanpa standardisasi (*preprocessing* dan algoritma dituning; k dari Silhouette) | profil 24 jam per hari (MW) | label *cluster* per hari | Silhouette, Davies-Bouldin, stabilitas, R² rekonstruksi, kebermaknaan *cluster* |
| Peramalan day-ahead | **Regression** HistGradientBoosting (+ RF, Ridge, baseline) | kalender + riwayat beban sampai 23:00 hari D-1 | beban (MW) tiap jam hari D | MAE, RMSE, R², **galat relatif** (MAE/rata-rata), dan penurunan galat terhadap baseline |

Satu model regresi dipakai untuk semua jam (jam sebagai fitur). Tidak ada tugas klasifikasi.
""")

md(f"""\
## 3. Data / Knowledge Preparation
**Sumber.** *PJME_hourly.csv*: beban listrik per jam (MW) wilayah PJM East, dari data PJM Interconnection. README repositori yang kami pakai menyatakan data berasal dari situs PJM dan tersedia di Kaggle,
dalam satuan MW, serta bahwa cakupan wilayah berubah dari tahun ke tahun. Berkas diambil dari salinan di GitHub (`panambY/Hourly_Energy_Consumption`), **bukan** langsung dari PJM; kami tidak memverifikasi ke sumber asli.
Zona waktu stempel waktu tidak dinyatakan di sumber; kami memperlakukannya sebagai waktu lokal.

**Pembersihan.** {D['rows_raw']:,} baris mentah ({D['dup']} stempel ganda, {D['missing']} jam hilang, semua gap 1 jam). Stempel ganda diambil yang pertama, jam hilang diinterpolasi linear (batas gap 6 jam),
lalu dipakai hanya hari penuh: **{D['start']} s.d. {D['end']} = {D['hours']:,} jam / {D['days']:,} hari**. Rata-rata {f0(D['mean'])} MW, std {f0(D['std'])} MW, rentang {f0(D['mn'])}–{f0(D['mx'])} MW.

**Skema peramalan & pencegahan kebocoran.** Ramalan "diterbitkan" pukul 00:00 hari D untuk 24 jam hari D. Hanya data sampai 23:00 hari D-1 yang boleh dipakai:
semua fitur beban memakai lag ≥ 24 jam atau nilai yang dibekukan pada 23:00 D-1 (`last_obs`, `d1_mean`, dst.). Tidak ada fitur cuaca (tidak tersedia pada data).

**Split.** Kronologis, tanpa pengacakan: train {S['train_start']} s.d. {S['train_end']} ({S['n_train']:,} jam), test {S['test_start']} s.d. {S['test_end']} ({S['n_test']:,} jam = {S['test_share']*100:.1f}%).
Penalaan hanya dengan `TimeSeriesSplit(4)` di dalam train. Skala/encoding dipasang hanya pada train.

**Fitur** (21 pada set final): kalender (jam, hari dalam pekan, bulan, libur federal AS menurut library `holidays`, H-1/H+1 libur); riwayat (lag 24/48/72/168/336 jam, rata-rata jam yang sama 7 hari terakhir, rata-rata 24 jam);
dan *fitur saat-terbit* (beban pukul 23:00, 20:00, 17:00 hari D-1; rata-rata/maks/min 24 jam D-1; rata-rata 7 hari; selisih tren harian).
""")
code("setup"); code("load")
md(f"""\
### 3.2 Pra-proses seluruh wilayah (data baru: 9 wilayah PJM lain)
Berkas tambahan berformat sama dengan PJME (kolom `Datetime` dan beban MW). Sepuluh seri dipakai: {", ".join(R2["regions"].keys())}. Dua seri dikecualikan: **NI** (berhenti Januari 2011) dan **PJM_Load** (berhenti Januari 2002); cakupan wilayahnya berbeda dari seri lain dan tidak ada data baru untuk periode uji.
Aturan pra-proses yang sama untuk semua wilayah, dan setiap langkah dihitung dampaknya di tabel di bawah:
1. stempel waktu ganda (pergantian jam musim) → ambil yang pertama; 2. jam yang hilang → NaN; 3. nilai tak masuk akal (≤ 0, atau < 35% dari median lokal pada jam-dalam-pekan yang sama, jendela 9 minggu) → NaN; 4. gap ≤ 6 jam → interpolasi linear; 5. hanya hari penuh dipakai.
Nilai antara 35% dan 50% dari median lokal **dicatat tetapi dibiarkan**, karena tidak bisa dibuktikan salah. Nilai tinggi (> 1,6× median lokal) tidak dibuang: puncak beban nyata (cuaca ekstrem) wajar bernilai tinggi.
""")
code("mr_setup"); code("mr_clean"); code("mr_audit")
md(f"""\
**Hasil audit.** Setiap wilayah punya {sorted(set(RG_[c]['audit']['dup'] for c in CMP))[0]} stempel ganda (pergantian jam musim) dan hanya gap 1 jam (gap terpanjang {max(RG_[c]['audit']['longest_gap'] for c in CMP)} jam), jadi interpolasi dibatasi pada kasus paling ringan.
Nilai tak masuk akal yang dibuang: {"; ".join(f"**{c}** {a['t'][:13]} ({f0(a['nilai'])} MW, {a['rasio']*100:.0f}% dari median lokal)" for c, a in anom_all)}.
Nilai diragukan yang **dibiarkan**: {"; ".join(f"{c} ({v} jam; terbanyak pada " + ", ".join(f"{d} ({k} jam)" for d, k in RG_[c]["audit"]["border_days"].items()) + ")" for c, v in bord.items())}. Yang jatuh di periode uji (mulai {R2["split"]}) dapat menaikkan galat wilayah itu sedikit: {", ".join(in_test) if in_test else "tidak ada"}; sisanya jatuh di periode train. Kami tidak dapat memeriksanya ke sumber asli.
""")
code("eda")
md(f"""\
**Catatan EDA.** Pola khas: beban terendah dini hari, dua musim puncak (musim panas dan dingin, kemungkinan karena pendingin dan pemanas ruang), dan akhir pekan lebih rendah
(rata-rata Sabtu {f0(R['eda']['dow_mean'][5])} MW dan Minggu {f0(R['eda']['dow_mean'][6])} MW, dibanding Senin {f0(R['eda']['dow_mean'][0])} MW).
""")

md("## 4. Implementation\n### 4.1 Clustering profil beban harian\nSatu baris = satu hari, 24 kolom = beban tiap jam. Ini analisis **deskriptif** (memakai data hari itu sendiri), bukan prediksi, sehingga tidak ada isu kebocoran. Bagian ini terdiri dari: (a) tuning *preprocessing* dan perbandingan algoritma, (b) pemilihan konfigurasi akhir, (c) hasil dan interpretasinya.")
CT = R["cluster_tuning"]; PR = {(p["prep"], p["alg"]): p for p in CT["pairs"]}
top = CT["pjme_top10"][0]; kraw = PR[("raw_MW", "kmeans")]; sh = PR[("bentuk_z + level", "kmeans")]; spc = PR[("std_per_jam (awal)", "spectral")]
md(f"""\
#### 4.1.1 Rancangan tuning
**Yang dibandingkan** (data PJME, {CT['pjme_n_total']} konfigurasi, k = 2..6):
- *Preprocessing* (10): MW mentah; standardisasi per jam (konfigurasi awal); *robust scaling*; *min-max*; log lalu standardisasi; bentuk saja (z-score per hari); rasio terhadap rata-rata harian; bentuk + level; PCA 95% variansi; PCA 3 komponen.
- Algoritma (6): K-Means, *Gaussian Mixture* (kovarians *full* dan *diagonal*), *Agglomerative* (Ward dan *average*), *Spectral clustering*.
- Ukuran: Silhouette, Davies-Bouldin, Calinski-Harabasz, **stabilitas** (rata-rata ARI antara dua *subsample* 70% pada hari yang sama), dan **R² rekonstruksi** (seberapa baik rata-rata *cluster* mewakili profil MW tiap hari).

**Masalah perbandingan yang adil.** Silhouette dihitung di ruang fitur masing-masing konfigurasi, sehingga tidak bisa dibandingkan antar-*preprocessing* (jarak di ruang berbeda). Karena itu dipakai dua ruang pembanding netral: S1 (standardisasi per jam) dan S2 (bentuk saja). `score` = rata-rata Silhouette di S1 dan S2.

**Aturan seleksi (ditetapkan sebelum melihat hasil):** pilih `score` tertinggi dengan stabilitas ≥ 0,8 dan *cluster* terkecil ≥ 5% hari. Sebanyak {CT['pjme_n_eligible']} dari {CT['pjme_n_total']} konfigurasi memenuhi syarat.
""")
code("cluster_prep"); code("cluster_funcs"); code("cluster_grid")
md("""\
Sel di atas menjalankan **grid ringkas** di dalam notebook (semua *preprocessing* dengan K-Means, plus GMM-diag, Ward, dan *spectral* pada data standar). Grid penuh (265 konfigurasi) dan pengujian pada 9 wilayah lain dijalankan lewat skrip terpisah `cluster_tune.py`, `cluster_pairs.py`, dan `cluster_regions.py` (hasilnya `cluster_tune_*.json`, dimuat oleh sel berikut) karena memakan waktu lama.
""")
code("cluster_full")
md(f"""\
#### 4.1.2 Hasil tuning dan pemilihan konfigurasi
- **Aturan yang dideklarasikan akan memilih** *{top['alg']}* pada *{top['prep']}* (k = {top['k']}, `score` {f3(top['score'])}). Selisihnya terhadap K-Means standar kecil: +{f3(spc['diff_vs_kmeans_std'])} ± {f3(spc['diff_sd'])} (*paired* 70%-*subsample*). Stabilitasnya {f2(top['stab'])} (lebih rendah daripada K-Means, {f2(CT['pjme_awal'][0]['stab'])}), dan *spectral clustering* tidak punya fungsi `predict` untuk hari baru.
- **Bentuk + level** (K-Means) menghasilkan `score` serupa (+{f3(sh['diff_vs_kmeans_std'])}), dan R² rekonstruksi di PJME hampir sama ({f2(CT['pjme_top10'][1]['recon_R2'])} vs {f2(CT['pjme_awal'][0]['recon_R2'])} pada konfigurasi awal), tetapi rata-rata 10 wilayah jauh lebih rendah ({f2(CT['regions'][2]['recon_R2'])} vs {f2(CT['regions'][0]['recon_R2'])}): *cluster*-nya memisahkan hari menurut bentuk sehingga kurang mewakili profil MW. Skor S2-nya juga cenderung menguntungkan konfigurasi ini karena ruang fiturnya mirip S2.
- **Keputusan: K-Means pada MW tanpa standardisasi.** Ini **bukan** pemenang `score` tertinggi menurut aturan di atas; ini keputusan pertimbangan yang diambil setelah melihat hasil, dengan alasan: selisih ketiganya kecil, K-Means mentah lebih stabil daripada *spectral* dan bentuk + level ({f2(CT['pjme_top10'][2]['stab'])} vs {f2(top['stab'])} dan {f2(CT['pjme_top10'][1]['stab'])}), mewakili profil MW sebaik K-Means standar atau lebih baik (R² {f2(CT['pjme_top10'][2]['recon_R2'])}), bisa memberi label hari baru, dan paling sederhana. Di 10 wilayah, dibanding standardisasi per jam, MW mentah lebih tinggi pada R² rekonstruksi di **{CT['raw_vs_std']['recon_R2']} dari 10** wilayah dan pada `score` di **{CT['raw_vs_std']['score']} dari 10**; selisih `score` PJME +{f3(kraw['diff_vs_kmeans_std'])} (menang di {kraw['wins']} dari 10 *resample*).
- **Tidak ada model yang jelas lebih baik.** Pada PJME semua algoritma memilih k = 2, *preprocessing* hampir tidak mengubah hasil K-Means (`score` 0,271–0,276), dan proyeksi PCA tidak menunjukkan celah antar *cluster*. Perubahan dari konfigurasi awal berupa perbaikan kecil yang konsisten, bukan lompatan.
- **Silhouette bergantung pada ruang fitur.** Angka Silhouette konfigurasi baru ({f3(CL['silhouette'])}) dan awal ({f3(CL['silhouette_std'])}) dihitung di ruangnya masing-masing, jadi selisihnya tidak bisa dibaca sebagai "model baru lebih bagus sebesar itu"; ukuran yang bisa dibandingkan adalah `score` dan R² rekonstruksi di atas. Kedua partisi mirip (ARI {f2(CL['ari_vs_std'])}).
""")
code("cluster_fit")
md(f"""\
**Interpretasi *cluster* (k = {CL['k']}).** *Cluster* *beban tinggi* ({CL['summary'][1]['n_hari']:,} hari, rata-rata {f0(CL['summary'][1]['rata2_MW'])} MW) terkonsentrasi pada Juni–Agustus dan Desember–Februari,
dengan puncak sore (median jam {CL['summary'][1]['jam_puncak_median']:.0f}). *Cluster* *beban rendah* ({CL['summary'][0]['n_hari']:,} hari, rata-rata {f0(CL['summary'][0]['rata2_MW'])} MW) berisi musim peralihan (Mar–Mei, Sep–Nov) dan lebih banyak akhir pekan
({CL['summary'][0]['persen_akhir_pekan']*100:.0f}% vs {CL['summary'][1]['persen_akhir_pekan']*100:.0f}%).

**Peringatan.** Silhouette {f3(CL['silhouette'])} (k = 2) tertinggi, tetapi proyeksi PCA memperlihatkan **tidak ada celah** antar *cluster*: beban harian membentuk kontinum dan K-Means hanya membelahnya pada tingkat beban sekitar tengah.
Jadi *cluster* ini berguna sebagai ringkasan ("hari ekstrem musim" vs "hari moderat"), bukan bukti bahwa ada dua jenis hari yang terpisah alami. Silhouette untuk k = 3 adalah {f3(sil[3])}.
""")

md("### 4.2 Regression: peramalan day-ahead\nPertama dibangun fitur dan split, lalu baseline, lalu penalaan HGB.")
code("features")
code("leak")
md("**Uji kebocoran (perturbasi).** Seluruh nilai beban satu hari D dirusak dengan angka acak, lalu semua fitur dihitung ulang. Jika ada fitur pada jam-jam hari D yang ikut berubah, fitur itu memakai informasi dari hari yang diramal. Pengujian pada 25 hari acak: lihat keluaran sel di atas (harus tidak ada fitur yang berubah).")
code("baseline")
md("Baseline terbaik adalah **persistensi harian** (beban jam yang sama kemarin). Model harus mengalahkannya agar dianggap punya kemampuan prediksi.")
code("tune"); code("ablation")
md(f"""\
**Ablasi fitur.** Kenaikan akurasi terbesar datang dari informasi, bukan dari algoritma: kalender saja → galat relatif {f1(AB[0]['rel'])}% di test; + lag ≥ 24 jam → {f1(AB[1]['rel'])}%; + fitur saat-terbit → {f1(AB[2]['rel'])}%.
Urutan yang sama muncul pada CV kronologis di dalam train ({f1(AB[0]['cv_rel_pct'])}% → {f1(AB[1]['cv_rel_pct'])}% → {f1(AB[2]['cv_rel_pct'])}%), yang tidak menyentuh test.

**Catatan kejujuran.** Set fitur saat-terbit dirancang setelah uji coba awal yang juga melihat data test, jadi angka test set C sedikit optimistis.
CV di dalam train dan rolling-origin di bawah memberi konfirmasi yang tidak bergantung pada pilihan itu.
""")
md("## 5. Results\n### 5.1 Perbandingan model di test (2015-2018)")
code("models")
code("plots")
md("### 5.2 Ketahanan: rolling-origin 8 lipatan (uji 6 bulan per lipatan, train meluas)")
code("rolling")
md("### 5.3 Analisis galat dan fitur penting")
code("errors"); code("importance")
md(f"""\
### 5.4 Perluasan: pipeline yang sama untuk 10 wilayah
Setiap wilayah memperoleh model sendiri dengan pipeline yang sama (fitur aman kebocoran, baseline, penalaan hanya di train, rolling-origin, clustering, uji kebocoran). Perbedaan dengan analisis utama PJME di atas:
- **Periode uji bersama** 2016-01-01 s.d. 2018-08-02 untuk semua wilayah (agar adil dan agar wilayah dengan data pendek seperti EKPC, mulai 2013, masih punya train ≥ 2,5 tahun). PJME diulang pada protokol ini **hanya untuk tabel perbandingan**; hasil utama PJME (uji 2015+) tetap yang di atas, sehingga angka PJME di tabel sedikit berbeda.
- **Set fitur dan rentang hiperparameter tidak disetel ulang per wilayah**: dirancang pada PJME. Untuk wilayah lain ini berarti pemilihan fitur tidak melihat data uji mereka; hanya hiperparameter yang dicari ulang di train (grid 8 kombinasi, `TimeSeriesSplit(4)`).
- Untuk menghemat waktu, Random Forest tidak dijalankan per wilayah (pembandingnya Ridge + 4 baseline), ablasi hanya 3 set di holdout (tanpa CV), uji kebocoran pada 10 hari acak, importance pada 4.000 jam.
- Prediksi uji dan model 21 fitur tiap wilayah disimpan ke `results/regions_out/` (berkas besar, tidak ikut di repo; dibutuhkan hanya untuk dashboard web).
""")
code("mr_funcs"); code("mr_run"); code("mr_compare"); code("mr_cluster_tuning"); code("mr_save")
md(f"""\
**Bacaan.** HGB mengalahkan baseline terbaik (MAE) di **{wins_reg} dari {n_reg}** wilayah, dengan penurunan MAE {f1(tmin)}%–{f1(tmax)}%; pada rolling-origin HGB menang di {roll_all} dari {roll_n} lipatan gabungan seluruh wilayah.
Galat relatif HGB terendah di **{lo_c}** ({f2(CMP[lo_c]['rel_HGB'])}%) dan tertinggi di **{hi_c}** ({f2(CMP[hi_c]['rel_HGB'])}%). Korelasi log(beban rata-rata) dengan galat relatif r = {f2(corr_)}: galat relatif {corr_txt} (hanya {n_reg} titik, indikatif saja).
Uji kebocoran: {"tidak ada fitur yang berubah pada seluruh wilayah" if not leak_any else "fitur berubah pada: " + ", ".join(leak_any)}.
**Batas klaim.** Tiap wilayah diuji pada satu periode uji ≈ 2,6 tahun tanpa data cuaca; angka antarwilayah tidak diuji signifikansinya, dan kesamaan pola antarwilayah yang bertetangga (mis. cuaca yang sama) berarti hasilnya tidak independen.
""")

md(f"""\
## 6. Evaluation
| Ukuran | Baseline terbaik (lag24) | HistGradientBoosting |
|---|---|---|
| MAE (MW) | {f0(BB['MAE'])} | **{f0(H['MAE'])}** |
| RMSE (MW) | {f0(BB['RMSE'])} | **{f0(H['RMSE'])}** |
| R² | {f3(BB['R2'])} | **{f3(H['R2'])}** |
| Galat relatif (MAE / rata-rata beban test {f0(S['test_mean'])} MW) | {f2(BB['rel'])}% | **{f2(H['rel'])}%** |

- **Cara membaca R²:** R² {f3(H['R2'])} menggelembung oleh siklus harian yang mudah ditebak: baseline lag24 saja sudah R² {f3(BB['R2'])}. Ukuran yang lebih jujur adalah penurunan galat terhadap baseline dan galat relatif, bukan R² mentah.
- **Kemampuan prediksi nyata:** HGB menurunkan MAE sebesar **{f1(SK['mae_reduction_pct'])}%** dan RMSE **{f1(SK['rmse_reduction_pct'])}%** dibanding persistensi harian.
- **Konsisten antarwaktu:** pada rolling-origin, HGB mengalahkan lag24 di **{RO['hgb_wins']} dari {RO['n']}** lipatan; galat relatif rata-rata {f2(RO['rel_hgb_mean'])}% (std {f2(RO['rel_hgb_std'])}) vs {f2(RO['rel_lag24_mean'])}% untuk lag24; R² rata-rata {f3(RO['r2_mean'])}.
  Lipatan terakhir (2018) paling sulit (galat relatif {f2(RO['rows'][-1]['rel_HGB'])}%), jadi ada indikasi galat naik pada periode terbaru; penyebabnya tidak diselidiki.
- **Algoritma bukan faktor utama:** Ridge sederhana mencapai R² {f3(RGm['R2'])} (galat relatif {f2(RGm['rel'])}%) dan Random Forest {f3(RFm['R2'])} ({f2(RFm['rel'])}%), selisih kecil dari HGB.
- **Galat:** bias rata-rata {f0(ER['bias'])} MW (cenderung meramal sedikit rendah), P90 galat absolut {f0(ER['p90'])} MW, P99 {f0(ER['p99'])} MW, maksimum {f0(ER['max'])} MW. Hari kerja {f2(ER['daytype']['Hari kerja'])}%, akhir pekan {f2(ER['daytype']['Akhir pekan'])}%, libur federal {f2(ER['daytype']['Libur federal'])}% (MAPE).
  Hari terburuk didominasi akhir pekan (8 dari 10 hari terburuk; periksa keluaran sel di atas).
- **Clustering:** Silhouette {f3(CL['silhouette'])} untuk k = {CL['k']} (K-Means pada MW tanpa standardisasi, hasil tuning); ini merupakan pembelahan kontinum (lihat Bagian 4.1). Tuning {CT['pjme_n_total']} konfigurasi tidak menemukan model yang jelas lebih baik.
""")

md(f"""\
## 7. Engineering Interpretation
1. **Mengapa akurasinya tinggi?** Beban agregat banyak pelanggan meredam perilaku acak tiap rumah, dan data per jam memberi riwayat yang sangat berkorelasi (beban sekarang sangat mirip beban jam sebelumnya).
   R² ≈ {f2(H['R2'])} dan galat relatif ≈ {f1(H['rel'])}% di sini terutama karena sifat data, bukan karena algoritma yang lebih baik (Ridge dan Random Forest hampir sama baiknya).
2. **Informasi terbaik adalah kondisi terbaru.** Fitur terpenting adalah beban pukul 23:00 hari sebelumnya dan beban jam yang sama kemarin (lihat permutation importance); hari dalam pekan, jam, dan rata-rata beban harian menyusul, sedangkan bulan relatif kecil. Ini sejalan dengan fakta bahwa beban dipengaruhi cuaca yang persisten dari hari ke hari.
3. **Keterbatasan fisik: cuaca tidak ada.** Beban sensitif terhadap suhu (pendingin/pemanas), sedangkan data ini tidak memuat prakiraan suhu. Galat terbesar terjadi pada hari yang berbeda dari hari sebelumnya (terutama akhir pekan setelah hari kerja dengan cuaca berbeda).
   Dugaan ini konsisten dengan temuan, tetapi **tidak diuji** karena tidak ada data cuaca. Menambahkan prakiraan suhu adalah langkah berikutnya yang paling masuk akal.
4. **Makna galat dalam MW.** MAE sekitar {f1(H['MAE']/1000)} GW; P99 sekitar {f1(ER['p99']/1000)} GW. Galat sebesar ini yang harus diantisipasi operator dengan cadangan; kami tidak menghitung biaya karena tidak ada data tarif atau cadangan.
5. **Apakah hasil PJME khas?** Pada 9 wilayah lain dengan pipeline yang sama, HGB mengalahkan baseline terbaik di {wins_reg} dari {n_reg} wilayah (galat relatif {f2(CMP[lo_c]['rel_HGB'])}%–{f2(CMP[hi_c]['rel_HGB'])}%), jadi pola "beban agregat per jam sangat dapat diramal dari kalender dan riwayat" tidak hanya berlaku di PJME. Galat relatif {corr_txt}.
6. **Keterbatasan generalisasi.** Data satu wilayah di Amerika Serikat (iklim, kalender libur, dan perilaku konsumsi berbeda dari Indonesia). Pola yang sama tidak otomatis berlaku di Indonesia (mis. tanpa musim dingin, ada Lebaran); hal itu perlu data lokal.
""")

md(f"""\
## 8. Conclusion
- Dengan fitur kalender + riwayat beban yang aman dari kebocoran, HGB meramal beban per jam sehari ke depan dengan **MAE {f0(H['MAE'])} MW (galat relatif {f2(H['rel'])}%), RMSE {f0(H['RMSE'])} MW, R² {f3(H['R2'])}** di test 2015–2018,
  **{f1(SK['mae_reduction_pct'])}% lebih baik** daripada persistensi harian, dan konsisten pada {RO['hgb_wins']} dari {RO['n']} lipatan rolling-origin.
- Angka paling jujur adalah rata-rata rolling-origin: galat relatif {f2(RO['rel_hgb_mean'])}% dan R² {f3(RO['r2_mean'])}; angka test sedikit optimistis karena set fitur saat-terbit dirancang setelah melihat data test.
- Akurasi yang tinggi berasal dari **jenis data** (agregat, per jam) dan **fitur riwayat**, bukan dari model yang lebih kompleks; Ridge hampir sama baiknya.
- Clustering memberi ringkasan "hari ekstrem musim vs moderat" (Silhouette {f3(CL['silhouette'])}), tetapi datanya kontinum sehingga *cluster* bukan kelompok alami. Tuning *preprocessing* dan perbandingan 6 algoritma hanya menghasilkan perbaikan kecil; K-Means tanpa standardisasi dipilih sebagai keputusan pertimbangan (bukan pemenang `score` menurut aturan yang dideklarasikan).
- Perluasan 10 wilayah: HGB mengalahkan baseline terbaik di {wins_reg} dari {n_reg} wilayah (penurunan MAE {f1(tmin)}%–{f1(tmax)}%, galat relatif {f2(CMP[lo_c]['rel_HGB'])}%–{f2(CMP[hi_c]['rel_HGB'])}%) tanpa menyetel ulang fitur per wilayah; pra-proses hanya membuang {len(anom_all)} nilai tak masuk akal.
- Langkah lanjut: tambahkan prakiraan suhu, uji ramalan probabilistik (interval), dan validasi pada data beban Indonesia bila tersedia.

## 9. Daftar Pustaka
Dataset acuan (rujukan tim; berkas yang dipakai berasal dari salinan GitHub dan paket data yang diunggah, bukan langsung dari PJM):
- [D1] Kaggle: robikscube, Hourly Energy Consumption, https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- [D2] Kaggle: ashishkumarrana, PJM Hourly Energy Consumption, https://www.kaggle.com/datasets/ashishkumarrana/pjm-hourly-energy-consumption
- [D3] IEEE DataPort: PJM Hourly Energy Consumption, https://ieee-dataport.org/documents/pjm-hourly-energy-consumption
- [D4] GitHub: navyasancheti/PJME-Hourly-Energy-Consumption, https://github.com/navyasancheti/PJME-Hourly-Energy-Consumption
- [D5] GitHub: panambY/Hourly_Energy_Consumption, https://github.com/panambY/Hourly_Energy_Consumption
- Berkas `PJME_hourly.csv` dibaca dari repositori `panambY/Hourly_Energy_Consumption` [D5]; README-nya menyebut asal data: situs PJM dan Kaggle.
- Pedregosa dkk., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011 (K-Means, HistGradientBoosting, Random Forest, Ridge, TimeSeriesSplit).
- Library Python `holidays` (kalender libur federal AS, `holidays.US`).

## 10. Lampiran
Seluruh angka pada laporan berasal dari keluaran notebook ini (`results/results.json` untuk PJME, `results/results_regions.json` untuk 10 wilayah). Data wilayah ada di folder `data_pjm/`. Benih acak 42. Waktu eksekusi notebook sekitar 40 menit pada 2 CPU, belum termasuk grid penuh *clustering* di `cluster_tuning/` (sekitar 45 menit per wilayah untuk PJME).
""")
md("### Lampiran A. Menyimpan hasil\nSel berikut menyimpan seluruh angka ke `results/results.json`.")
code("save")

import re as _re
def _reduce_comment(l):
    if not l.lstrip().startswith("#"): return l
    for a, b in _VOC: l = l.replace(a, b)
    return _re.sub(r"\bfitur\b", "feature", l)
_VOC = [("aman kebocoran", "bebas data leakage"), ("Uji kebocoran", "Uji data leakage"), ("uji kebocoran", "uji data leakage"), ("kebocoran", "data leakage"), ("Kebocoran", "Data leakage"),
        ("peramalan", "prediksi"), ("Peramalan", "Prediksi"), ("meramal", "memprediksi"), ("diramal", "diprediksi"), ("ramalan", "prediksi"), ("Ramalan", "Prediksi"),
        ("Galat relatif", "Relative error"), ("galat relatif", "relative error"), ("Galat", "Error"), ("galat", "error"),
        ("klaster", "cluster"), ("stempel waktu", "timestamp"), ("stempel ganda", "timestamp ganda"), ("stempel", "timestamp"), ("Klaster", "Cluster"), ("Penalaan", "Tuning"), ("penalaan", "tuning"), ("antarlipatan", "antar-fold"),
        ("Ablasi fitur", "Ablation study"), ("Ablasi", "Ablation study"), ("ablasi", "ablation study"), ("saat-terbit", "snapshot D-1"), ("hiperparameter", "hyperparameter"),
        ("perturbasi", "perturbation"), ("benih acak", "random seed")]
for _c in cells:
    if _c.cell_type == "code":
        _c.source = "\n".join(_reduce_comment(l) for l in _c.source.split("\n"))
    if _c.cell_type == "markdown":
        t = _c.source
        for a, b in _VOC: t = t.replace(a, b)
        t = _re.sub(r"\bfitur\b", "feature", t); t = _re.sub(r"\bFitur\b", "Feature", t)
        t = _re.sub(r"(?<![.\w\"'])[lL]ipatan\b", lambda m: "Fold" if m.group(0)[0] == "L" else "fold", t)
        _c.source = t
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "pjm_load_forecasting.ipynb"); print("notebook ditulis:", len(cells), "sel")
