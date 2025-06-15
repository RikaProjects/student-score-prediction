# Laporan Proyek Machine Learning - Rika

## Domain Proyek: Pendidikan

### Latar Belakang
Pendidikan merupakan faktor kunci dalam pembangunan sumber daya manusia. Evaluasi performa siswa secara akurat dapat membantu sekolah memberikan intervensi lebih dini kepada siswa yang berisiko rendah. Dalam konteks ini, kemampuan untuk memprediksi nilai akhir siswa berdasarkan karakteristik awal seperti jenis kelamin, latar belakang orang tua, dan kursus persiapan menjadi sangat relevan.

---

## Business Understanding

### Problem Statements
1. Bagaimana memprediksi nilai rata-rata akhir siswa (`average_score`) berdasarkan karakteristik demografis?
2. Fitur-fitur manakah yang memiliki pengaruh terbesar terhadap performa siswa?

### Goals
1. Membangun model prediksi nilai akhir siswa menggunakan machine learning regresi.
2. Mengevaluasi performa model menggunakan metrik MAE, RMSE, dan R² Score.
3. Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap nilai akhir siswa berdasarkan analisis *feature importance* dari Random Forest.

### Solution Statements
- Menggunakan **Linear Regression** sebagai model baseline karena mudah diinterpretasikan.
- Menggunakan **Random Forest Regressor** untuk menangkap relasi non-linear dan menilai pentingnya setiap fitur.
- Evaluasi menggunakan MAE, RMSE, dan R² Score. Analisis fitur penting dilakukan menggunakan `.feature_importances_` dari Random Forest.

---

## Data Understanding

Dataset berasal dari [Kaggle - Students Performance Dataset](https://www.kaggle.com/spscientist/students-performance-in-exams).  
Dataset yang digunakan memiliki **1.000 baris (sampel)** dan **8 kolom fitur** sebelum dilakukan proses pembuatan fitur baru.

### Deskripsi Fitur:
1. **gender**  
   Jenis kelamin siswa. `male` atau `female`.
2. **race/ethnicity**  
   Kelompok etnis siswa, dari `group A` sampai `group E`.
3. **parental level of education**  
   Pendidikan terakhir orang tua siswa. Contoh: `some college`, `associate's degree`, `high school`.
4. **lunch**  
   Tipe makan siang: `standard` atau `free/reduced`.
5. **test preparation course**  
   Program persiapan ujian: `completed` atau `none`.
6. **math score**  
   Skor matematika (0–100).
7. **reading score**  
   Skor membaca (0–100).
8. **writing score**  
   Skor menulis (0–100).

### Hasil Pengecekan Data:
- Tidak terdapat **missing values**
- Tidak terdapat **data duplikat**
- Distribusi skor ujian relatif simetris
- Variasi skor akhir dipengaruhi oleh `lunch` dan `test preparation course`

---

## Data Preparation

1. Menambahkan kolom `average_score` sebagai target (rata-rata dari `math`, `reading`, dan `writing` score).
2. Menghapus skor individual untuk mencegah *data leakage*.
3. Melakukan **one-hot encoding** pada fitur kategorikal.
4. Melakukan **train-test split** 80:20.

---
## Model Development

Pada proyek ini, digunakan dua algoritma regresi, yaitu Linear Regression dan Random Forest Regressor. Implementasi dan parameter masing-masing model dijelaskan secara eksplisit di bawah.

### 1. Linear Regression
Linear Regression adalah algoritma regresi dasar yang bekerja dengan mencari garis lurus terbaik (line of best fit) yang meminimalkan **jumlah kuadrat dari galat (error)** antara nilai prediksi dan nilai aktual. Model ini cocok digunakan saat hubungan antar variabel bersifat linier.

- **Prinsip kerja**: Menyusun persamaan linear dalam bentuk `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`, dan mencari nilai β (koefisien) yang meminimalkan galat.
- **Kelebihan**: Sederhana dan mudah diinterpretasikan.
- **Kekurangan**: Kurang mampu menangkap hubungan non-linear antar fitur.

Model ini digunakan sebagai baseline untuk membandingkan performa dengan model yang lebih kompleks.

**Implementasi di Notebook:**

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

**Parameter yang Digunakan:**

* `fit_intercept=True` (default)
* `copy_X=True` (default)
* `n_jobs=None` (default)
* `positive=False` (default)

> Semua parameter di atas menggunakan nilai default, sehingga tidak ada parameter tambahan yang di-tune.

### 2. Random Forest Regressor
Random Forest adalah model **ensemble** yang terdiri dari banyak pohon keputusan (decision trees). Setiap pohon dilatih dengan subset data yang diambil secara acak (bootstrap), dan hasil prediksi akhir diambil dari **rata-rata prediksi semua pohon** (untuk kasus regresi).

- **Prinsip kerja**:
  - Membuat banyak decision tree dari subset data acak.
  - Masing-masing tree melakukan prediksi.
  - Output akhir adalah rata-rata dari semua prediksi pohon.

- **Kelebihan**:  
  - Mampu menangani hubungan non-linear antar variabel.  
  - Lebih tahan terhadap overfitting dibanding single decision tree.

- **Kekurangan**:  
  - Interpretasi model lebih sulit dibandingkan Linear Regression.  
  - Waktu pelatihan lebih lama, terutama dengan banyak pohon dan data besar.

**Implementasi di Notebook:**

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

**Parameter yang Digunakan:**

* `random_state=42` (diset eksplisit untuk memastikan reproducibility)
* `n_estimators=100` (default)
* `max_depth=None` (default)
* `min_samples_split=2` (default)
* `min_samples_leaf=1` (default)

> Parameter selain `random_state` menggunakan nilai default dari scikit-learn.

---

Model Random Forest dipilih karena mampu menangkap interaksi kompleks antar fitur yang mungkin tidak ditangkap oleh model linear sederhana.

---

Model dievaluasi menggunakan metrik seperti **Mean Squared Error (MSE)** dan **R² Score**. Hasil menunjukkan bahwa meskipun Random Forest dapat memberikan prediksi yang lebih fleksibel, **Linear Regression justru memberikan nilai R² yang lebih tinggi pada data ini**, yang menunjukkan bahwa hubungan antar fitur cukup linear.


---

## Evaluation

### Metrik Evaluasi:

| Model               | MAE   | RMSE  | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression  | 10.49 | 13.40 | 0.16     |
| Random Forest      | 11.48 | 14.78 | -0.02    |

> Linear Regression memberikan performa lebih baik dibanding Random Forest dalam hal prediksi.

### Visualisasi: Actual vs Predicted

- **Linear Regression** menunjukkan hasil yang lebih mendekati garis ideal (`y = x`).
- **Random Forest** menunjukkan sebaran yang lebih acak, menandakan underfitting.

### Penjelasan Metrik Evaluasi
- **MAE**: Rata-rata absolut dari selisih nilai aktual dan prediksi. MAE = ∑|y - ŷ| / n
- **RMSE**: Akar dari rata-rata kuadrat selisih nilai aktual dan prediksi.
- **R² Score**: Mengukur proporsi variansi target yang bisa dijelaskan oleh model.

---

### Analisis Feature Importance - Random Forest

Berdasarkan `.feature_importances_`, fitur paling berpengaruh adalah:

1. `lunch_standard`  
2. `test preparation course_none`  
3. `gender_male`  
4. `parental level of education_high school`  
5. `race/ethnicity_group B`  

Interpretasi:
- Akses terhadap makanan standar dan kursus persiapan memiliki pengaruh besar terhadap performa siswa.
- Gender dan latar belakang pendidikan orang tua juga relevan dalam memengaruhi hasil akhir.

---

## Kesimpulan

### Menjawab Problem Statement:
1. Prediksi nilai akhir siswa berhasil dilakukan dengan **Linear Regression** sebagai model terbaik saat ini.
2. Fitur paling berpengaruh terhadap performa siswa berdasarkan Random Forest:
   - `lunch_standard`
   - `test preparation course_none`
   - `gender_male`

### Rekomendasi:
- Fokus intervensi pada program makan dan kursus persiapan.
- Tambahkan fitur tambahan seperti jam belajar, absensi, dan motivasi.
- Coba model lain seperti Gradient Boosting atau XGBoost, serta tuning hyperparameter untuk meningkatkan akurasi.

---
