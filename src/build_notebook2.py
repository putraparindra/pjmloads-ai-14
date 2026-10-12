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
CT = R["cluster_tuning"]; PR = {(p["prep"], p["alg"]): p for p in CT["pairs"]}
top = CT["pjme_top10"][0]; kraw = PR[("raw_MW", "kmeans")]; sh = PR[("bentuk_z + level", "kmeans")]; spc = PR[("std_per_jam (awal)", "spectral")]
import os, re
RT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_text.json"), encoding="utf8"))

def ref(t):
    """Teks narasi laporan memakai nomor tabel/gambar; di notebook keluarannya muncul di sel, sehingga rujukan dibuat relatif."""
    t = t.replace("pada Tabel 12", "pada hasil uji berpasangan di atas")
    t = t.replace("(Subbab 4.3)", "(Subbab 3.1.3)").replace("(Subbab 5.7)", "(Subbab 3.2.7)").replace("(Subbab 6.3)", "(Subbab 3.3.3)")
    t = t.replace("(Gambar 7), angka-angka", "(Subbab 3.2.7), angka-angka")
    t = t.replace("akhir pekan (Gambar 1).", "akhir pekan (Subbab 3.2.1).").replace("beban puncak (Gambar 3)", "beban puncak (Subbab 3.2.5)")
    t = t.replace("(Gambar 4 dan Tabel 7)", "(Subbab 3.2.3 dan 3.2.5)").replace("prediksi diterbitkan", "prediksi dibuat")
    t = re.sub(r"\(Gambar \d+ dan Tabel \d+\)", "(lihat gambar dan tabel di atas)", t)
    t = re.sub(r"\(Gambar \d+\)", "(lihat gambar di atas)", t)
    t = re.sub(r"Pada (Tabel|Gambar) \d+", lambda m: "Pada " + ("tabel" if m.group(1) == "Tabel" else "gambar") + " di atas", t)
    t = re.sub(r"Berdasarkan Tabel \d+", "Berdasarkan tabel di atas", t)
    t = re.sub(r"Tabel \d+ menunjukkan", "Tabel di atas menunjukkan", t)
    t = re.sub(r"Tabel \d+", "tabel di atas", t)
    t = re.sub(r"Gambar \d+", "gambar di atas", t)
    return t.strip()
P = lambda k: ref(RT[k])

# ---- pemeriksaan: angka pada narasi statis harus sama dengan keluaran notebook (results/*.json) ----
_chk = [(f0(H["MAE"]), "con0"), (f0(H["RMSE"]), "con0"), (f3(H["R2"]), "con0"), (f2(H["rel"]), "con0"), (f1(SK["mae_reduction_pct"]), "con0"),
        (f0(BB["MAE"]), "reg1"), (f3(BB["R2"]), "reg1"), (f0(AB[0]["MAE"]), "abl"), (f0(AB[2]["MAE"]), "abl"), (f2(RO["rel_hgb_mean"]), "roll"),
        (f0(CL["summary"][0]["n_hari"]), "cl_c1"), (f0(CL["summary"][1]["n_hari"]), "cl_c1"), (f3(sil[2]["silhouette"] if isinstance(sil[2], dict) else sil[2]), "cl_sil"),
        (f0(ER["p99"]), "err3"), (f1(tmin), "mr1"), (f1(tmax), "mr1")]
for _v, _k in _chk: assert _v in RT[_k], (_v, _k)

# =====================================================================================================================
md(f"""\
# Prediksi Beban Listrik per Jam Satu Hari ke Depan (*Day-Ahead Load Forecasting*) dan Segmentasi Profil Beban Harian pada Jaringan PJM
**Laporan Mini-Project *Machine Learning* · Kecerdasan Artifisial (TKEE262513)** &nbsp;|&nbsp; **Kelompok 14:** Alfito Putra Parindra (24/534768/TK/59290) · Muhammad Fasha (24/535041/TK/59325) · Najamudin Fawwaz Haq (24/539273/TK/59804) · Qais Faqih Basyaev (24/542181/TK/60189) · Muhammad Zeva Hardi Seswanto (24/542780/TK/60292) &nbsp;|&nbsp; Departemen Teknik Elektro dan Teknologi Informasi, Fakultas Teknik, Universitas Gadjah Mada, 2026

Dua pendekatan AI pada satu data beban per jam jaringan PJM (Amerika Serikat, {D['start'][:4]}–{D['end'][:4]}): ***regression*** (HistGradientBoosting, dibandingkan dengan Random Forest, Ridge, dan *baseline* sederhana) untuk memprediksi beban tiap jam **sehari ke depan**, dan ***clustering*** (K-Means, dipilih lewat *tuning preprocessing* dan perbandingan algoritma) untuk mengelompokkan profil beban harian. PJM East (PJME) menjadi analisis utama, lalu pipeline yang sama dijalankan pada 9 wilayah PJM lain.

Notebook ini mengikuti struktur laporan: **Bab 1** Pendahuluan, **Bab 2** Metode dan Model AI (*AI formulation* dan *data/knowledge preparation*), **Bab 3** Hasil dan Pembahasan (*implementation*, *results*, *evaluation*, *engineering interpretation*), dan **Bab 4** Kesimpulan. Pada *implementation* dan *results*, *regression* dibahas lebih dulu karena merupakan model utama, dan *clustering* menyusul sebagai analisis pendukung.

**Cara menjalankan** (dari *root* repo; Python 3.10+ dan paket di `requirements.txt`): buka notebook ini lalu *Run All* (sekitar 40 menit pada 2 CPU). Data dibaca dari `data_pjm/`, hasil *tuning clustering* penuh dari `cluster_tuning/` (dihasilkan terpisah oleh skrip di folder itu), dan semua keluaran (JSON, gambar) ditulis ke `results/`.

> **Catatan.** Beban agregat wilayah jauh lebih teratur daripada konsumsi satu rumah tangga. Akurasi yang tinggi di sini terutama berasal dari sifat data (agregat, per jam, riwayat sangat berkorelasi), bukan dari model yang canggih (lihat Subbab 3.4).
""")

# ---------------------------------------------------------------- BAB 1
md("""\
## BAB 1. PENDAHULUAN
### 1.1 Latar Belakang
Listrik merupakan infrastruktur yang menopang hampir seluruh kegiatan ekonomi dan sosial, dan kebutuhannya di Indonesia terus bertambah. Konsumsi listrik per kapita Indonesia mencapai 1.285 kWh pada Oktober 2023 [1], sedangkan ESDM menyebut rata-rata ASEAN sebesar 3.672 kWh per kapita [2]. Pemerintah menargetkan tambahan kapasitas pembangkit 69,5 GW dalam RUPTL 2025–2034 [3], dan PLN memproyeksikan kebutuhan listrik naik dari 306 TWh menjadi 511 TWh pada 2034 [4]. Pertumbuhan ini harus dikelola oleh operator sistem yang setiap saat menyeimbangkan daya yang dibangkitkan dengan daya yang dikonsumsi.

Sistem Jawa-Bali menunjukkan skala masalahnya. Beban puncak malam sistem ini tercatat 27.070 MW pada 17 Oktober 2018 [5], dan kelebihan pasok (*oversupply*) turun dari 7 GW pada akhir 2022 menjadi 4 GW pada akhir 2023 [6]. Prediksi beban yang terlalu rendah berisiko membuat cadangan daya tidak mencukupi, sedangkan prediksi yang terlalu tinggi membuat pembangkit beroperasi lebih banyak dari kebutuhan. Di sisi lain, 42,6 GW dari 69,5 GW kapasitas baru dalam RUPTL berasal dari energi terbarukan dan 10,3 GW dari penyimpanan energi [3], sehingga perencanaan operasi harian makin menuntut prediksi beban yang akurat. Nilai ekonominya pun nyata: perkiraan kasar menyebut penurunan *error* prediksi beban jangka pendek sebesar 1 poin persentase (MAPE dari 4% menjadi 3%) menghemat sekitar US$300.000 per tahun untuk utilitas dengan beban puncak 1 GW [7].

Prediksi beban jangka pendek dengan *machine learning* sudah diteliti pada sistem Indonesia. Ainuri dkk. memakai *Long Short-Term Memory* (LSTM) pada sistem Jawa-Bali dan memperoleh MAPE 5,67% serta RMSE 1.683,09 MW pada model terbaiknya [8]. Namun, dalam penelusuran kami tidak ditemukan data beban per jam sistem Indonesia yang dapat diunduh secara publik, sehingga metode sulit direproduksi dan dibandingkan secara terbuka. Sebagai gantinya, data beban per jam PJM Interconnection tersedia secara publik [9]–[13]. PJM melayani 65 juta penduduk di 13 negara bagian dan District of Columbia, dengan perkiraan beban puncak musim panas sekitar 149.000 MW dan kapasitas terpasang sekitar 185.000 MW [14]. Skala ini cukup besar untuk menguji metode pada beberapa wilayah dengan karakter beban berbeda.

Berdasarkan uraian tersebut, mini project ini membangun model prediksi beban *day-ahead* per jam dengan HistGradientBoosting (HGB) pada data PJM, dan mengelompokkan hari berdasarkan profil beban 24 jam dengan K-Means. Model diuji pada sepuluh wilayah dengan pembagian data kronologis dan dibandingkan dengan *baseline* sederhana, sehingga peningkatan akurasinya dapat diukur secara terbuka. Hasilnya diharapkan menjadi acuan metode yang dapat diterapkan pada data beban sistem Indonesia ketika data tersebut tersedia.

### 1.2 Input, Output, dan Perangkat Uji Coba
Masukan, keluaran, dan perangkat uji coba sistem pada setiap pendekatan AI disajikan dalam Tabel 1 sebagai berikut.

*Tabel 1. Input, output, dan perangkat uji coba sistem.*

| Komponen | Keterangan |
|---|---|
| Input (*regression*) | Informasi kalender, meliputi jam, hari dalam pekan, bulan, dan hari libur nasional Amerika Serikat, serta riwayat beban yang tersedia hingga pukul 23:00 hari D−1. |
| Output (*regression*) | Sebanyak 24 nilai prediksi beban dalam MW, masing-masing untuk setiap jam pada hari D. |
| Input (*clustering*) | Satu profil beban yang terdiri atas 24 nilai per jam untuk setiap hari. |
| Output (*clustering*) | Label *cluster* setiap hari dan uraian karakteristik masing-masing *cluster*. |
| Bahasa pemrograman | Python 3.11 |
| *Library* | pandas 3.0, NumPy 2.4, Matplotlib 3.10, scikit-learn 1.8 [15], *holidays* 0.106 [16] |
| *Platform* | Jupyter Notebook (`pjm_load_forecasting.ipynb`) |
| Sistem operasi | Linux |
| Perangkat keras | 2 CPU, RAM 7 GB, tanpa GPU |
| *Random seed* | 42 |
""")

md(f"""\
### 1.3 Tujuan Penggunaan AI
Beban listrik dipengaruhi oleh pola harian, mingguan, dan tahunan, serta keberadaan hari libur. Pengaruh berbagai pola tersebut terhadap besarnya beban tidak bersifat linear dan dapat saling berinteraksi. Sebagai contoh, profil beban harian pada musim panas dapat berbeda dari musim dingin, sebagaimana profil hari kerja dapat berbeda dari akhir pekan. Keragaman kombinasi tersebut sulit diwakili secara menyeluruh melalui aturan IF–THEN.

*Machine learning* memungkinkan hubungan tersebut dipelajari dari data historis dalam jumlah besar. Pada wilayah PJME, tersedia {f0(D['rows_raw'])} baris data mentah yang digunakan sebagai dasar analisis. Dengan pendekatan ini, pola beban diidentifikasi melalui pembelajaran dari data, sehingga tidak bergantung pada penyusunan aturan secara manual.

### 1.4 Batasan Masalah
Batasan ruang lingkup mini project ini ditetapkan sebagai berikut.
1. Informasi yang digunakan model terbatas pada kalender dan riwayat beban. Karena *dataset* tidak menyediakan data cuaca, pengaruh suhu tidak dimodelkan.
2. Prediksi dibuat satu kali setiap hari, yaitu pukul 00:00, untuk mencakup 24 jam pada hari tersebut (*day-ahead*).
3. Analisis dilakukan pada 10 wilayah PJM. Seri NI dan PJM_Load yang juga tersedia dalam paket data tidak disertakan karena masing-masing berakhir pada Januari 2011 dan Januari 2002, sehingga tidak mencakup periode *testing*.
4. Data yang dianalisis berasal dari Amerika Serikat. Oleh karena itu, hasil yang diperoleh belum tentu dapat diterapkan secara langsung pada sistem kelistrikan Indonesia.
5. Keluaran model berupa satu nilai prediksi untuk setiap jam, tanpa menyertakan *prediction interval*.
""")

# ---------------------------------------------------------------- BAB 2
md("""\
## BAB 2. METODE DAN MODEL AI
### 2.1 AI Formulation
Pendekatan kecerdasan artifisial yang digunakan pada proyek adalah *regression* dalam prediksi beban dengan keluaran yang diprediksi berupa nilai numerik kontinu, yaitu beban listrik dalam MW. Sementara itu, digunakan *clustering* (K-Means) sebagai analisis pendukung untuk mengelompokkan profil beban harian tanpa label kelas. Hasil *clustering* dipakai untuk membantu menjelaskan karakteristik data dan tidak dipakai sebagai masukan model *regression*. Formulasi AI yang digunakan pada proyek ditunjukkan pada Tabel 2 sebagai berikut.

*Tabel 2. Formulasi AI yang digunakan.*

| Tujuan | Pendekatan | Masukan/Input | Keluaran/Output dan Evaluasi |
|---|---|---|---|
| Prediksi beban *day-ahead* | *Regression*: HistGradientBoosting (HGB) dengan pembanding Random Forest, Ridge, dan empat *baseline*. | Kalender dan riwayat beban sampai 23:00 hari D-1 (21 *feature*) | Beban (MW) tiap jam pada hari D; MAE, RMSE, R², *relative error* |
| Segmentasi profil beban harian | *Clustering*: K-Means pada profil dalam MW tanpa standardisasi, hasil *tuning preprocessing* dan perbandingan 6 algoritma; jumlah *cluster* (k) dipilih dengan Silhouette Score | Profil 24 jam per hari dalam MW | Label *cluster*; Silhouette Score, Davies-Bouldin Index, stabilitas, R² rekonstruksi |

Pendekatan utama yang dipilih adalah *regression*, dengan HGB sebagai model utama karena data berbentuk tabel dengan banyak *feature* campuran dan jumlah data yang cukup besar, serta mampu menangkap interaksi antar-*feature* tanpa rekayasa manual. Input model berupa kalender dan riwayat beban sampai pukul 23:00 hari D-1, sedangkan output berupa beban listrik tiap jam pada hari D. HGB dibandingkan dengan Random Forest dan Ridge Regression untuk melihat apakah peningkatan akurasi berasal dari algoritma atau dari *feature* yang digunakan. Empat *baseline* tanpa proses *learning* juga dipakai sebagai pembanding minimum, sehingga ketika model tidak lebih baik dari *baseline* tersebut, penggunaan *machine learning* tidak memberikan manfaat. Pada proyek ini, satu model dipakai untuk seluruh jam dalam sehari, dengan jam sebagai salah satu *feature*.

Pendekatan pendukung adalah *clustering* dengan *unsupervised learning*, karena data beban tidak memiliki label jenis hari sehingga mampu untuk meringkas pola beban harian dalam pengelompokan sesuai dengan kemiripan bentuk profil. Input berupa profil beban 24 jam per hari dan output berupa label *cluster* untuk tiap hari dengan algoritma yang dipakai adalah K-Means. K-Means mengelompokkan setiap hari ke pusat *cluster* (*centroid*) terdekat dengan *centroid* berupa rata-rata profil beban dalam MW sehingga setiap *cluster* mudah dibaca dan dikaitkan dengan musim atau akhir pekan. Pada proyek ini, jumlah *cluster* (k) ditentukan dari Silhouette Score.

### 2.2 Data/Knowledge Preparation
#### 2.2.1 Sumber dan Karakteristik Dataset
""")
md(f"""\
Data yang digunakan pada proyek ini adalah data riil/nyata dari *open-public sources* sebagai berikut.
1. Repositori GitHub: panambY (Hourly_Energy_Consumption) [13]
2. Repositori Kaggle: robikscube (Hourly Energy Consumption) [9]
3. Repositori Kaggle: ashishkumarrana (PJM Hourly Energy Consumption) [10]
4. Repositori GitHub: navyasancheti (PJME-Hourly-Energy-Consumption) [12]
5. IEEE DataPort: PJM Hourly Energy Consumption [11]

Berkas *PJME_hourly.csv* dibaca dari repositori panambY [13], sedangkan berkas sembilan wilayah lainnya berasal dari paket data yang diunggah tim. README pada [13] menyatakan bahwa data berasal dari situs PJM, dinyatakan dalam satuan MW, dan cakupan wilayahnya berubah dari tahun ke tahun. Data **tidak** diambil langsung dari PJM dan tidak diverifikasi ke sumber aslinya. Zona waktu pada *timestamp* tidak dijelaskan di sumber sehingga diperlakukan sebagai waktu lokal. Nama lengkap tiap kode wilayah juga tidak dijelaskan di sumber, sehingga ditulis sebagai kode saja.

Setiap berkas terdiri atas dua kolom, yaitu *Datetime* (*timestamp* per jam) dan beban listrik dalam MW. Pada PJME terdapat {f0(D['rows_raw'])} baris mentah, yang mencakup {D['dup']} *timestamp* ganda dan {D['missing']} jam yang hilang (*missing value*), seluruhnya berupa celah sepanjang 1 jam. Data PJME yang dipakai mencakup {D['start']} sampai {D['end']} dengan {f0(D['hours'])} jam ({f0(D['days'])} hari). Beban rata-rata {f0(D['mean'])} MW dengan simpangan baku {f0(D['std'])} MW, nilai minimum {f0(D['mn'])} MW, dan nilai maksimum {f0(D['mx'])} MW. Sel di bawah memuat dan membersihkan data PJME; keluarannya menampilkan angka-angka tersebut.
""")
code("setup"); code("load")

md("""\
#### 2.2.2 Pra-pemrosesan Data
Pra-pemrosesan data atau *preprocessing* dilakukan dengan aturan yang sama pada semua wilayah. Pertama, *duplicate timestamp* yang muncul akibat pergantian *daylight saving time* diambil yang pertama. Kedua, data diubah menjadi *hourly time series* tanpa *gap*, sehingga jam yang hilang menjadi *missing value*. Ketiga, *outlier*, yaitu nilai nol, nilai negatif, atau nilai di bawah 35% dari *local median* pada *hour-of-week* yang sama (*window* 9 minggu), diubah menjadi *missing value*. Keempat, *gap* sampai 6 jam diisi dengan *linear interpolation*. Kelima, hanya *complete day* (24 jam penuh) yang dipakai. Nilai tinggi tidak dibuang karena *peak load* yang ekstrem merupakan kondisi nyata yang justru perlu diprediksi. Nilai di antara 35% dan 50% dari *local median* dicatat sebagai *suspicious value*, tetapi tetap dipertahankan karena tidak dapat dibuktikan salah. Hasilnya untuk setiap wilayah dihitung oleh sel berikut dan diringkas pada Tabel 3.
""")
code("mr_setup"); code("mr_clean"); code("mr_audit")
_rows3 = "\n".join(f"| {c} | {RG_[c]['audit']['start'][:7]} s.d. {RG_[c]['audit']['end'][:7]} | {f0(RG_[c]['audit']['days'])} | {RG_[c]['audit']['missing_hours']} | {RG_[c]['audit']['dup']} | {RG_[c]['audit']['n_anomalies']} | {RG_[c]['audit']['borderline']} |" for c in RG_)
md(f"""\
*Tabel 3. Hasil pra-pemrosesan data setiap wilayah.*

| Wilayah | Periode | Hari | *Missing hour* | *Duplicate timestamp* | *Removed value* | *Suspicious value* |
|---|---|---:|---:|---:|---:|---:|
{_rows3}

Berdasarkan tabel di atas, setiap wilayah hanya memiliki *gap* data sepanjang 1 jam (*gap* terpanjang 1 jam) dan 4 *duplicate timestamp*. Jumlah *missing hour* tergolong sangat kecil dibanding jumlah data sehingga *interpolation* tidak mengubah pola beban secara berarti. *Outlier* yang dibuang hanya 3 titik, yaitu DOM pada 2009-12-12 00:00 (1.253 MW, 14% dari *local median*); FE pada 2011-06-01 01:00 (0 MW, 0% dari *local median*); PJMW pada 2003-05-29 00:00 (487 MW, 10% dari *local median*). *Suspicious value* yang dipertahankan terdapat pada DAYTON (1 jam), DEOK (30 jam), EKPC (10 jam). Pada DEOK, 29 dari 30 jam tersebut berada pada 22–25 April 2018 (21 jam pada 22 April) sehingga masuk ke periode *testing* dan dapat sedikit menaikkan *error* DEOK. Pada EKPC, 10 jam tersebut berada pada tahun 2013 (periode *training*). Pemeriksaan ini hanya berdasarkan aturan, tidak diverifikasi ke sumber asli.
""")

md("""\
#### 2.2.3 Pembentukan Feature dan Target
*Target* yang diprediksi adalah beban pada jam *t* (MW). Prediksi dibuat pukul 00:00 hari *D* untuk 24 jam hari *D*, sehingga hanya data sampai pukul 23:00 hari *D-1* yang boleh dipakai. Oleh karena itu, setiap *feature* berbasis beban memakai *lag* minimal 24 jam atau nilai *snapshot* pada pukul 23:00 hari *D-1*. Aturan ini mencegah *data leakage*, yaitu masuknya informasi dari waktu yang sedang diprediksi ke dalam *feature*. Seluruh 21 *feature* dirinci pada Tabel 4 berikut.

*Tabel 4. Daftar feature yang digunakan.*

| Kelompok | Nama *feature* | Keterangan |
|---|---|---|
| Kalender (6) | `hour`, `dow`, `month` | Jam (0–23), hari dalam pekan (0 = Senin), dan bulan |
| | `is_hol`, `hol_prev`, `hol_next` | Penanda hari libur nasional AS pada hari D, hari sebelumnya, dan hari sesudahnya (*library* `holidays`) |
| *Lag* (6) | `lag24`, `lag48`, `lag168`, `lag336` | Beban pada jam yang sama 1, 2, 7, dan 14 hari sebelumnya |
| | `wk_mean` | Rata-rata beban pada jam yang sama selama 7 hari terakhir |
| | `dmean_prev` | Rata-rata beban 24 jam yang berakhir pada jam t−24 |
| *Snapshot* D-1 (9) | `last_obs`, `last_obs_m3`, `last_obs_m6` | Beban pukul 23:00, 20:00, dan 17:00 pada hari D-1 |
| | `d1_mean`, `d1_max`, `d1_min` | Rata-rata, maksimum, dan minimum beban 24 jam yang berakhir pukul 23:00 hari D-1 |
| | `w1_mean`, `d1_trend` | Rata-rata beban 7 hari yang berakhir pukul 23:00 hari D-1, dan selisih `d1_mean` dengan rata-rata 24 jam sebelumnya |
| | `lag72` | Beban pada jam yang sama 3 hari sebelumnya |
""")
code("features")

md(f"""\
#### 2.2.4 Pembagian Data dan Uji Data Leakage
Data dibagi secara kronologis (*chronological split*) tanpa *shuffling*. Data *training* mencakup {S['train_start']} sampai {S['train_end']} ({f0(S['n_train'])} jam), sedangkan data *testing* mencakup {S['test_start']} sampai {S['test_end']} ({f0(S['n_test'])} jam, {f1(S['test_share']*100)}% dari seluruh data). Pembagian kronologis dipilih karena pada kondisi nyata model selalu memprediksi masa depan dari data masa lalu. *Hyperparameter tuning* hanya dilakukan di dalam data *training*, dan *scaling* serta *encoding* hanya di-*fit* pada data *training*. Untuk analisis sepuluh wilayah, periode *testing* disamakan, yaitu 2016-01-01 sampai 2018-08-02, agar hasil antarwilayah dapat dibandingkan.
""")
code("split")
md("""\
*Data leakage* diuji dengan *perturbation test*. Seluruh nilai beban pada satu hari *D* diganti dengan angka acak, kemudian semua 21 *feature* dihitung ulang. Jika ada *feature* pada jam-jam hari *D* yang ikut berubah, berarti *feature* tersebut memuat informasi dari hari yang sedang diprediksi. Pada 25 hari acak, tidak ada satu pun *feature* yang berubah. Hasil yang sama diperoleh pada 9 wilayah lain (10 hari acak per wilayah). Dengan demikian, rancangan *feature* memenuhi aturan bahwa hanya data sampai pukul 23:00 hari *D-1* yang digunakan.
""")
code("leak")

# ---------------------------------------------------------------- BAB 3.1
md("""\
## BAB 3. HASIL DAN PEMBAHASAN
### 3.1 Implementation
#### 3.1.1 Alur Implementasi
Alur sistem proyek dilakukan berdasarkan *workflow* yang ditunjukkan diagram alir pada Gambar 1 sebagai berikut.

![Gambar 1. Diagram alir implementasi model AI](report/gambar_alur_implementasi.png)

*Gambar 1. Diagram alir implementasi model AI.*
""")

md(f"""\
#### 3.1.2 Regression Day-Ahead
Model utama adalah *HistGradientBoostingRegressor* dari scikit-learn. Model ini membangun banyak *decision tree* secara berurutan, dengan setiap *tree* memperbaiki sisa *error* dari *tree* sebelumnya, dan memakai histogram untuk mempercepat pencarian titik pemisah. Model ini pun dilakukan *tuning* dengan *hyperparameter tuning* menggunakan *GridSearchCV* pada {TU['grid_size']} kombinasi parameter, *cross-validation* berbasis *TimeSeriesSplit* dengan 4 *fold*, dan skor MAE. Seluruh proses ini hanya memakai data *training*. Parameter terbaik adalah *learning_rate* {f2(TU['best']['learning_rate'])}, *max_iter* {TU['best']['max_iter']}, *max_leaf_nodes* {TU['best']['max_leaf_nodes']}, dan *min_samples_leaf* {TU['best']['min_samples_leaf']}.
""")
code("tune")
md("""\
Hasil yang diperoleh dibandingkan dengan pembanding, yaitu Random Forest (300 *tree*, *min_samples_leaf* 5, *max_features* 0,5) dan Ridge Regression (alpha 10) dengan *one-hot encoding* untuk *feature* kalender. Empat *baseline* tanpa proses *learning* juga dihitung (sel berikut): (1) persistensi harian (lag24), yaitu beban pada jam yang sama kemarin; (2) persistensi mingguan (lag168), yaitu beban pada jam yang sama minggu lalu; (3) rata-rata lag24, lag48, dan lag168; serta (4) rata-rata beban pada jam yang sama selama 7 hari terakhir. Model harus mengalahkan *baseline* terbaik agar dianggap memiliki kemampuan prediksi.
""")
code("baseline")

md(f"""\
#### 3.1.3 Clustering Profil Beban Harian
Pada proyek ini, setiap hari direpresentasikan oleh 24 nilai beban per jam ({f0(D['days'])} hari lengkap pada PJME). Analisis tersebut bersifat deskriptif karena memakai data hari itu sendiri sehingga tidak ada isu *data leakage*. Sementara itu, konfigurasi awal memakai data yang distandarkan per jam dan K-Means dengan k dipilih dari Silhouette Score. Konfigurasi tersebut kemudian dilakukan *tuning* dan dibandingkan dengan algoritma lain. Rancangan konfigurasi tersebut dirinci sebagai berikut.
1. ***Preprocessing* (10 varian):** standardisasi per jam (konfigurasi awal), MW asli tanpa standardisasi, *robust scaling*, *min-max*, log lalu standardisasi, bentuk saja (*z-score* per hari), bentuk saja (rasio ke rata-rata harian), bentuk + level (bentuk ditambah rata-rata harian terstandar), PCA 95% variansi, dan PCA 3 komponen.
2. **Algoritma (6 jenis):** K-Means, *Gaussian Mixture Model* (kovarians *full* dan *diagonal*), *Agglomerative clustering* (*linkage* Ward dan *average*), serta *spectral clustering*. Jumlah *cluster* k diuji dari 2 sampai 6, sehingga totalnya {CT['pjme_n_total']} konfigurasi pada PJME.
3. **Ukuran:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz, stabilitas, dan R² rekonstruksi. Stabilitas adalah rata-rata *Adjusted Rand Index* (ARI) antara dua hasil *clustering* pada dua *subsample* acak 70% data, dibandingkan pada hari yang sama (5 ulangan). R² rekonstruksi menunjukkan seberapa besar variasi profil MW yang dijelaskan oleh rata-rata *cluster*.

Silhouette Score dihitung di ruang *feature* milik tiap konfigurasi, sehingga angkanya tidak bisa dibandingkan langsung antar-*preprocessing* karena jarak diukur di ruang yang berbeda. Untuk perbandingan, setiap hasil *clustering* dinilai ulang di dua ruang pembanding netral, yaitu S1 (data distandarkan per jam) dan S2 (bentuk profil saja, *z-score* per hari), dan *score* didefinisikan sebagai rata-rata Silhouette di S1 dan S2. Sebelum hasil dilihat, ditetapkan aturan seleksi: pilih konfigurasi dengan *score* tertinggi yang memiliki stabilitas minimal 0,8 dan *cluster* terkecil minimal 5% dari seluruh hari. Perbedaan antar-kandidat teratas diuji dengan 10 kali *subsample* 70% berpasangan (kandidat dan K-Means standar dinilai pada *subsample* yang sama). Pengujian diulang pada 9 wilayah lain dengan 5 konfigurasi terpilih.
""")
code("cluster_prep"); code("cluster_funcs"); code("cluster_grid")
md(f"""\
Sel di atas menjalankan **grid ringkas** di dalam notebook (semua *preprocessing* dengan K-Means, plus GMM diagonal, Ward, dan *spectral* pada data standar). Grid penuh ({CT['pjme_n_total']} konfigurasi) dan pengujian pada 9 wilayah lain dijalankan lewat skrip terpisah `cluster_tune.py`, `cluster_pairs.py`, dan `cluster_regions.py` di folder `cluster_tuning/` karena memakan waktu lama. Hasilnya (`cluster_tune_*.json`) dimuat dan dibahas pada Subbab 3.2.7.

#### 3.1.4 Evaluasi Tambahan
Empat analisis tambahan dilakukan di luar evaluasi pada satu periode *testing*. *Ablation study* membandingkan tiga kelompok *feature* (kalender saja; kalender dan *lag*; semua *feature*) untuk melihat kontribusi tiap kelompok. Evaluasi *rolling-origin* mengulang proses *training* dan *testing* pada 8 periode berurutan (masing-masing sekitar 6 bulan) untuk melihat konsistensi hasil terhadap waktu. Analisis *error* melihat sebaran *error* menurut jam, bulan, dan jenis hari. *Permutation importance* mengukur seberapa besar MAE naik ketika nilai suatu *feature* diacak. Keempatnya dijalankan pada Subbab 3.2.3 sampai 3.2.5.

#### 3.1.5 Pipeline Multi-Wilayah
Setiap wilayah dilatih dengan model tersendiri menggunakan *pipeline* yang sama, yaitu 21 *feature* yang sama, empat *baseline*, *tuning* HGB pada grid 8 kombinasi dengan *TimeSeriesSplit* 4 *fold*, pembanding Ridge, evaluasi *rolling-origin*, *clustering* K-Means, dan *perturbation test*. Terdapat beberapa perbedaan dengan analisis utama PJME.
1. Periode *testing* disamakan pada 2016-01-01 sampai 2018-08-02 untuk semua wilayah. Wilayah dengan data pendek seperti EKPC (mulai pertengahan 2013) tetap memiliki data *training* sekitar 2,6 tahun. PJME diulang pada protokol ini hanya untuk tabel perbandingan, sedangkan hasil utama PJME (*testing* mulai 2015) tidak diubah. Karena itu, angka PJME pada tabel perbandingan sedikit berbeda dari hasil utama.
2. Set *feature* dan rentang *hyperparameter* tidak disetel ulang untuk tiap wilayah karena keduanya dirancang pada PJME. Untuk wilayah lain, hanya nilai *hyperparameter* yang dicari ulang pada data *training*.
3. Untuk menghemat waktu komputasi, Random Forest tidak dijalankan per wilayah, *ablation study* hanya memakai tiga set pada satu pembagian data *holdout*, *perturbation test* memakai 10 hari acak, dan *permutation importance* dihitung pada 4.000 jam.

Prediksi *testing* dan model 21 *feature* tiap wilayah disimpan ke `results/regions_out/` (berkas besar, tidak ikut di repo; dibutuhkan hanya untuk dashboard web). Sel berikut mendefinisikan fungsi *pipeline*; pemanggilannya ada pada Subbab 3.2.6.
""")
code("mr_funcs")

# ---------------------------------------------------------------- BAB 3.2
md("### 3.2 Results\n#### 3.2.1 Eksplorasi Data")
code("eda")
md(P("eda1") + "\n\n" + P("eda2"))

md("#### 3.2.2 Hasil Regression")
code("models")
md(P("reg1"))
code("plots")
md(P("reg2") + "\n\n" + P("reg3"))

md("#### 3.2.3 Ablation Study")
code("ablation")
md(P("abl"))

md("#### 3.2.4 Evaluasi Rolling-Origin")
code("rolling")
md(P("roll"))

md("#### 3.2.5 Analisis Error dan Feature Importance")
code("errors")
md(P("err1") + "\n\n" + P("err2") + "\n\n" + P("err3"))
code("importance")
md(P("imp"))

md("#### 3.2.6 Hasil pada Sepuluh Wilayah")
code("mr_run"); code("mr_compare")
md(P("mr1") + "\n\n" + P("mr2"))

md("""\
#### 3.2.7 Hasil Clustering
*Clustering* merupakan analisis pendukung yang tidak dipakai sebagai masukan model *regression*. Hasilnya disajikan dalam dua tahap. Tahap pertama adalah hasil *tuning preprocessing* dan perbandingan algoritma, yang dimuat dari hasil grid penuh di `cluster_tuning/` (sel pertama di bawah). Tahap kedua adalah hasil konfigurasi yang akhirnya dipilih (sel kedua).
""")
code("cluster_full")
md(f"""\
**Perbandingan algoritma.** {P('cl_alg')}

**Pengaruh *preprocessing*.** {P('cl_prep')}

**Uji perbedaan antar-kandidat.** Perbedaan antarkandidat diuji pada 10 *subsample* 70% (k = 2), dengan selisih *score* terhadap K-Means standar dihitung pada *subsample* yang sama. {P('cl_pair')}

**Konsistensi pada sepuluh wilayah.** {P('cl_reg')}

{P('cl_fig')}

**Konfigurasi yang dipilih.** {P('cl_chosen')}

{P('cl_nobetter')}
""")
code("cluster_fit")
md(P("cl_sil") + "\n\n" + P("cl_c1") + "\n\n" + P("cl_c2"))
code("mr_cluster_tuning")
md(P("lampB"))

# ---------------------------------------------------------------- 3.3 Evaluation
md(f"""\
### 3.3 Evaluation
#### 3.3.1 Metrik Evaluasi
{P('ev61_a')}

$$MAE = \\frac{{1}}{{n}} \\sum |y - \\hat{{y}}|$$
$$RMSE = \\sqrt{{\\frac{{1}}{{n}} \\sum (y - \\hat{{y}})^2}}$$
$$R^2 = 1 - \\frac{{\\sum (y - \\hat{{y}})^2}}{{\\sum (y - \\bar{{y}})^2}}$$
$$Relative\\ error = \\frac{{MAE}}{{\\bar{{y}}}} \\times 100\\%$$

{P('ev61_b')}

#### 3.3.2 Arti Nilai Evaluasi
*Ringkasan evaluasi HGB dibanding baseline terbaik pada PJME.*

| Ukuran | *Baseline* terbaik (lag24) | HistGradientBoosting |
|---|---:|---:|
| MAE (MW) | {f0(BB['MAE'])} | **{f0(H['MAE'])}** |
| RMSE (MW) | {f0(BB['RMSE'])} | **{f0(H['RMSE'])}** |
| R² | {f3(BB['R2'])} | **{f3(H['R2'])}** |
| *Relative error* (rata-rata beban *testing* {f0(S['test_mean'])} MW) | {f2(BB['rel'])}% | **{f2(H['rel'])}%** |

{P('ev62')}

#### 3.3.3 Catatan terhadap Validitas Hasil
1. {P('ev63_0')}
2. {P('ev63_1')}
3. {P('ev63_2')}
4. {P('ev63_3')}
5. {P('ev63_4')}

#### 3.3.4 Evaluasi Clustering
{P('ev64_0')}

{P('ev64_1')}

1. {P('ev64_2')}
2. {P('ev64_3')}
3. {P('ev64_4')}
4. {P('ev64_5')}
""")

# ---------------------------------------------------------------- 3.4 Engineering Interpretation
md(f"""\
### 3.4 Engineering Interpretation
#### 3.4.1 Interpretasi Hasil dari Perspektif Teknik Elektro
{P('int0')}

{P('int1')}

{P('int2')}

#### 3.4.2 Manfaat Hasil AI
1. {P('ben0')}
2. {P('ben1')}
3. {P('ben2')}

#### 3.4.3 Penggunaan dalam Sistem Teknik Elektro
{P('use0')}

#### 3.4.4 Keterbatasan
1. {P('lim0')}
2. {P('lim1')}
3. {P('lim2')}
4. {P('lim3')}
5. {P('lim4')}
6. {P('lim5')}

#### 3.4.5 Pengembangan untuk Kondisi Nyata
1. {P('dev0')}
2. {P('dev1')}
3. {P('dev2')}
4. {P('dev3')}
5. {P('dev4')}
""")

# ---------------------------------------------------------------- BAB 4 + pustaka + lampiran
md(f"""\
## BAB 4. KESIMPULAN
{P('con0')}

{P('con1')}

{P('con2')}

{P('con3')}

## DAFTAR PUSTAKA
[1] Kementerian ESDM, “Paparkan Capaian 2023, Menteri ESDM Klaim Peningkatan Rasio Elektrifikasi,” 15 Jan. 2024. [Online]. Available: https://www.esdm.go.id/id/berita-unit/direktorat-jenderal-ketenagalistrikan/paparkan-capaian-2023-menteri-esdm-klaim-peningkatan-rasio-elektrifikasi (Diakses pada 6 Oktober 2026).

[2] Kementerian ESDM, “Tertinggal dari Negara Lain, Kementerian ESDM Libatkan Stakeholder dalam Diskusi Peningkatan Konsumsi Listrik Per Kapita,” 25 Nov. 2022. [Online]. Available: https://esdm.go.id/id/berita-unit/direktorat-jenderal-ketenagalistrikan/tertinggal-dari-negara-lain-kementerian-esdm-libatkan-stakeholder-dalam-diskusi-peningkatan-konsumsi-listrik-per-kapita (Diakses pada 6 Oktober 2026).

[3] hukumonline, “Bahlil Resmi Umumkan RUPTL Tahun 2025–2034,” 26 Mei 2025. [Online]. Available: https://www.hukumonline.com/berita/a/bahlil-resmi-umumkan-ruptl-tahun-2025-2034-lt68344d6f0dc8a (Diakses pada 8 Oktober 2026).

[4] Investortrust, “PLN Bersiap Hadapi Lonjakan Permintaan Listrik 511 TWh, dari ‘Data Center’ hingga EV,” 2 Jun. 2025. [Online]. Available: https://investortrust.id/business/68024/pln-bersiap-hadapi-lonjakan-permintaan-listrik-511-twh-dari-data-center-hingga-ev (Diakses pada 8 Oktober 2026).

[5] Kementerian ESDM, “Kunjungi APB Ungaran, Menteri ESDM Pastikan Listrik Jawa Bali Aman Jelang Libur Akhir Tahun,” 13 Des. 2018. [Online]. Available: https://esdm.go.id/id/berita-unit/direktorat-jenderal-ketenagalistrikan/kunjungi-apb-ungaran-menteri-esdm-pastikan-listrik-jawa-bali-aman-jelang-libur-akhir-tahun (Diakses pada 11 Oktober 2026).

[6] Bisnis.com, “ESDM Catat Kelebihan Pasokan Listrik Jawa-Bali Turun jadi 4 GW pada 2023,” 18 Jan. 2024. [Online]. Available: https://ekonomi.bisnis.com/read/20240118/44/1733610/esdm-catat-kelebihan-pasokan-listrik-jawa-bali-turun-jadi-4-gw-pada-2023 (Diakses pada 12 Oktober 2026).

[7] T. Hong, “Crystal ball lessons in predictive analytics,” *EnergyBiz*, pp. 35–37, Spring 2015. [Online]. Available: https://blog.drhongtao.com/2015/04/crystal-ball-lessons-in-predictive-analytics.html (Diakses pada 12 Oktober 2026).

[8] M. F. Ainuri, Sarjiya, and I. Ardiyanto, “Short-term load forecasting with long short-term memory: A case study of Java-Bali system,” in *Proc. 12th Int. Conf. Inf. Technol. Electr. Eng. (ICITEE)*, Yogyakarta, Indonesia, 2020, pp. 257–262, doi: 10.1109/ICITEE49829.2020.9271763.

[9] robikscube, “Hourly Energy Consumption,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

[10] ashishkumarrana, “PJM Hourly Energy Consumption,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/ashishkumarrana/pjm-hourly-energy-consumption

[11] “PJM Hourly Energy Consumption,” IEEE DataPort. [Online]. Available: https://ieee-dataport.org/documents/pjm-hourly-energy-consumption

[12] navyasancheti, “PJME-Hourly-Energy-Consumption,” GitHub repository. [Online]. Available: https://github.com/navyasancheti/PJME-Hourly-Energy-Consumption

[13] panambY, “Hourly_Energy_Consumption,” GitHub repository. [Online]. Available: https://github.com/panambY/Hourly_Energy_Consumption

[14] PJM Interconnection, “PJM prepared to meet summer electricity demand,” *Inside Lines*, 11 Mei 2022. [Online]. Available: https://insidelines.pjm.com/?p=205370 (Diakses pada 12 Oktober 2026).

[15] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[16] Python package *holidays*, modul *holidays.US* (kalender libur nasional Amerika Serikat), versi 0.106.

## LAMPIRAN
**Lampiran 1. Dashboard website mini-project AI.** Hasil model dapat dicoba pada dashboard website di https://loads-ai.vercel.app.

**Lampiran 2. Tautan repositori GitHub dan folder Google Drive.**
1. Repositori GitHub: https://github.com/putraparindra/pjmloads-ai-14
2. Folder Google Drive: https://s.id/gOqlo

**Lampiran 3. Grid *hyperparameter* HGB.**

| Parameter | Nilai yang dicoba (PJME, analisis utama) | Nilai yang dicoba (analisis 10 wilayah) | Terbaik pada PJME (analisis utama) |
|---|---|---|---|
| *learning_rate* | 0,03; 0,06 | 0,06 (tetap) | {f2(TU['best']['learning_rate'])} |
| *max_iter* | 400; 800 | 400; 800 | {TU['best']['max_iter']} |
| *max_leaf_nodes* | 31; 63 | 31; 63 | {TU['best']['max_leaf_nodes']} |
| *min_samples_leaf* | 50; 100 | 50; 100 | {TU['best']['min_samples_leaf']} |

{P('lampA')}

**Lampiran 4. Hasil *clustering* pada sepuluh wilayah.** Tabel hasil K-Means MW asli pada setiap wilayah dibandingkan dengan konfigurasi awal ditampilkan oleh sel terakhir pada Subbab 3.2.7.

**Lampiran 5. Berkas pendukung.** {P('lampC') if 'lampC' in RT else ''}Seluruh angka pada laporan ini berasal dari keluaran notebook ini. Hasil PJME tersimpan pada `results/results.json` dan hasil sepuluh wilayah pada `results/results_regions.json`. Hasil *tuning clustering* tersimpan pada `cluster_tuning/cluster_tune_PJME.json`, `cluster_pairs_PJME.json`, dan `cluster_tune_<wilayah>.json`, yang dihasilkan oleh skrip `cluster_tune.py`, `cluster_pairs.py`, dan `cluster_regions.py`. Data tiap wilayah berada pada folder `data_pjm/`. *Random seed* yang dipakai adalah 42, dan waktu eksekusi notebook sekitar 40 menit pada 2 CPU (belum termasuk *tuning clustering* penuh yang dijalankan terpisah, sekitar 45 menit per wilayah untuk PJME).

**Lampiran 6. Menyimpan hasil.** Dua sel berikut menyimpan seluruh angka ke `results/results_regions.json` (sepuluh wilayah) dan `results/results.json` (PJME).
""")
code("mr_save"); code("save")

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
