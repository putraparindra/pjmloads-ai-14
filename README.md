# Prediksi Beban Listrik per Jam dan Segmentasi Profil Beban Harian (PJM)

Mini Project *Artificial Intelligence untuk Teknik Elektro*, Teknik Elektro UGM.
Dua pendekatan AI pada satu data beban per jam jaringan PJM (Amerika Serikat, 2002–2018):

- **Regression**: HistGradientBoosting (HGB) memprediksi beban tiap jam untuk **hari esok** (*day-ahead*), dibandingkan dengan Random Forest, Ridge, dan empat *baseline* sederhana.
- **Clustering**: K-Means mengelompokkan profil beban harian (24 jam) sebagai analisis pendukung. *Preprocessing* dan algoritma dituning dan dibandingkan.

Analisis utama memakai **PJM East (PJME)**; pipeline yang sama dijalankan juga pada 9 wilayah lain (AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, PJMW).

**Kelompok:** Alfito Putra Parindra, Muhammad Fasha, Najamudin Fawwaz Haq, Qais Faqih Basyaev, Muhammad Zeva Hardi Seswanto.

## Ringkasan hasil

Uji kronologis PJME (train 2002–2014, test 2015–2018), tanpa pengacakan dan tanpa data cuaca.

| | MAE (MW) | *Relative error* | R² |
|---|---|---|---|
| *Baseline* terbaik (beban jam yang sama kemarin) | 2.209 | 7,05% | 0,780 |
| **HGB** | **1.275** | **4,07%** | **0,916** |

- MAE turun **42,3%** dibanding *baseline*; HGB menang di **8 dari 8** *fold rolling-origin* (rata-rata *relative error* 4,00%).
- Pada 10 wilayah, HGB mengalahkan *baseline* terbaik di **10 dari 10** wilayah (*relative error* 3,38%–7,05%).
- Peningkatan akurasi terutama berasal dari **fitur riwayat beban**, bukan dari algoritma: Ridge dan Random Forest hanya sedikit di bawah HGB. Fitur dibuat tanpa *data leakage* (diuji dengan *perturbation test*).
- Angka test PJME kemungkinan sedikit optimistis, karena set fitur dirancang setelah percobaan awal yang juga melihat data test. Angka *rolling-origin* lebih aman.

**Clustering.** Dari 265 konfigurasi di PJME (10 *preprocessing* × 6 algoritma × k = 2–6; 5 konfigurasi utama diuji ulang di 10 wilayah), **tidak ada model yang jelas lebih baik**. Semua algoritma memilih k = 2, dan *preprocessing* hampir tidak mengubah hasil K-Means. Konfigurasi akhir adalah **K-Means pada profil dalam MW tanpa standardisasi** (Silhouette PJME 0,494; k = 3 hanya di DOM). Ini sedikit lebih baik dan konsisten dibanding konfigurasi awal, tetapi merupakan **keputusan pertimbangan setelah hasil dilihat**: aturan seleksi yang ditetapkan sebelumnya akan memilih *spectral clustering* (selisih skor hanya +0,007), yang tidak dipilih karena kurang stabil dan tidak bisa memberi label hari baru. Data beban harian membentuk kontinum (tidak ada celah pada proyeksi PCA), jadi *cluster* dibaca sebagai ringkasan deskriptif, bukan kelompok alami. Silhouette dihitung di ruang fitur masing-masing konfigurasi, sehingga selisih antar-konfigurasi tidak bisa dibaca langsung sebagai peningkatan kualitas.

## Struktur repo

```
.
├── pjm_load_forecasting.ipynb   notebook utama (sudah dijalankan; output tersimpan)
│                                 strukturnya mengikuti laporan: Bab 1 Pendahuluan, Bab 2 Metode dan Model AI,
│                                 Bab 3 Hasil dan Pembahasan (regression dulu, lalu clustering), Bab 4 Kesimpulan
├── data_pjm/                    beban per jam 10 wilayah (<WILAYAH>_hourly.csv)
├── cluster_tuning/              tuning clustering penuh (skrip + hasil JSON)
│   ├── cluster_tune.py          grid 10 preprocessing x 6 algoritma x k=2..6 (PJME)
│   ├── cluster_pairs.py         uji berpasangan pada 10 subsample 70%
│   ├── cluster_regions.py       5 konfigurasi utama di wilayah lain
│   └── cluster_tune_*.json, cluster_pairs_PJME.json   hasilnya (dibaca notebook)
├── results/                     keluaran acuan: results.json, results_regions.json, fig_*.png
├── report/                      laporan (docx dan pdf) dan diagram alir implementasi (gambar_alur_implementasi.png)
├── src/                         sumber sel notebook + pembangun notebook (opsional)
└── requirements.txt
```

## Cara menjalankan

Semua perintah dari *root* repo.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. notebook (± 40 menit pada 2 CPU): buka di Jupyter lalu Run All, atau
jupyter nbconvert --to notebook --execute --inplace pjm_load_forecasting.ipynb

# 2. (opsional) ulangi tuning clustering penuh; hasil ditulis ke cluster_tuning/
python cluster_tuning/cluster_tune.py PJME        # ± 45 menit
python cluster_tuning/cluster_pairs.py PJME
python cluster_tuning/cluster_regions.py AEP COMED DAYTON DEOK DOM DUQ EKPC FE PJMW
```

Notebook membaca `data_pjm/` dan `cluster_tuning/*.json`, lalu menulis keluarannya ke `results/` (`results/regions_out/` berukuran besar dan di-*ignore*). *Random seed* 42. Notebook dibangun dari `src/` dengan `python src/build_notebook2.py` (butuh `results/*.json` yang sudah ada, karena angka pada sel *markdown* diambil dari sana).

## Data

Berkas CSV berasal dari salinan dataset PJM Hourly Energy Consumption yang dipublikasikan pihak ketiga, **bukan diunduh langsung dari PJM** dan tidak diverifikasi ke sumber aslinya. Rujukan dataset tim:

1. https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
2. https://www.kaggle.com/datasets/ashishkumarrana/pjm-hourly-energy-consumption
3. https://ieee-dataport.org/documents/pjm-hourly-energy-consumption
4. https://github.com/navyasancheti/PJME-Hourly-Energy-Consumption
5. https://github.com/panambY/Hourly_Energy_Consumption (sumber berkas `PJME_hourly.csv`)

Nama lengkap tiap kode wilayah tidak dijelaskan di sumber data, sehingga ditulis sebagai kode saja. Pra-proses (nilai ganda, jam hilang, nilai tak masuk akal) didokumentasikan di notebook Subbab 2.2.2.

## Keterbatasan

- Tidak ada data cuaca; beban sensitif terhadap suhu, dan dugaan bahwa galat terbesar berkaitan dengan cuaca **tidak diuji**.
- Data dari satu jaringan di Amerika Serikat (iklim dan kalender libur berbeda dari Indonesia); hasil tidak otomatis berlaku di sana.
- Tiap wilayah hanya diuji pada satu periode ≈ 2,6 tahun; signifikansi statistik antarwilayah tidak diuji, dan wilayah bertetangga berbagi cuaca sehingga hasilnya tidak sepenuhnya independen. Set fitur dirancang di PJME dan tidak disetel ulang per wilayah.
- Model hanya memberi satu nilai prediksi (tanpa interval ketidakpastian).
