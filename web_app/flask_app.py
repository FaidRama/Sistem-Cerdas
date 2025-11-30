from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg') # Penting: Mode non-GUI untuk server
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# ================= KELAS DARI KODEMU (Sedikit Modifikasi) =================

class KNNClassifier:
    def __init__(self, k=9):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, x):
        # Jarak Euclidean
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = distances.argsort()[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def get_nearest_info(self, x, inv_classes, n_neighbors=5):
        # Fungsi ini dimodifikasi untuk mengembalikan teks string untuk ditampilkan di web
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        sorted_idx = np.argsort(distances)[:n_neighbors]
        
        info = []
        for i, idx in enumerate(sorted_idx):
            label_name = inv_classes[self.y_train[idx]]
            dist_val = distances[idx]
            info.append(f"{i+1}. Jarak: {dist_val:.2f} | Label: {label_name}")
        return info

class visualisasi_pca:
    def __init__(self, X, y, inv_classes, n_components=2):
        self.X = X
        self.y = y
        self.inv_classes = inv_classes
        self.n_components = n_components
        self.X_pca, self.components, self.eigenvalues, self.mean_X = self._fit_pca()

    def _fit_pca(self):
        mean_X = np.mean(self.X, axis=0)
        X_centered = self.X - mean_X
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        components = eigenvectors[:, :self.n_components]
        X_reduced = np.dot(X_centered, components)
        return X_reduced, components, eigenvalues, mean_X
    
    def get_plot_url(self, p):
        # Modifikasi: Return base64 string gambar, BUKAN plt.show()
        input_centered = p - self.mean_X
        input_reduced = np.dot(input_centered, self.components)
        
        plt.figure(figsize=(8,6))
        # Plot data latih
        unique_labels = np.unique(self.y)
        for label in unique_labels:
            plt.scatter(self.X_pca[self.y == label, 0],
                        self.X_pca[self.y == label, 1],
                        label=self.inv_classes[label], alpha=0.6)

        # Plot input baru
        plt.scatter(input_reduced[0], input_reduced[1],
                    color='black', marker='*', s=250, label='Input Kamu', edgecolors='white')
        
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("Visualisasi Posisi Data Kamu (PCA)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Simpan ke buffer memory
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close() # Bersihkan plot agar memori tidak penuh
        return plot_url

# ================= LOAD DATA & TRAIN SAAT SERVER START =================

# Setup Global Variables
knn = None
viz = None
inv_classes = {}

def init_model():
    global knn, viz, inv_classes
    
    # Gunakan path relatif agar aman di server
    csv_path = os.path.join(os.path.dirname(__file__), 'weather_data_balanced.csv')
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("ERROR: File CSV tidak ditemukan! Pastikan sudah diupload.")
        return

    # Preprocessing sama seperti kodemu
    df = df[df["Weather_Condition"].isin(["Sunny", "Rainy", "Stormy", "Cloudy"])]
    X = df[["Temperature", "Humidity", "Wind_Speed", "Air_Pressure"]].values
    y_raw = df["Weather_Condition"].values

    classes = {label: idx for idx, label in enumerate(np.unique(y_raw))}
    inv_classes = {v: k for k, v in classes.items()}
    y = np.array([classes[label] for label in y_raw])

    # Kita pakai semua data untuk training di Web App (biar makin pintar)
    # Atau kamu bisa tetap split kalau mau konsisten
    knn = KNNClassifier(k=9)
    knn.fit(X, y)

    # Siapkan PCA
    viz = visualisasi_pca(X, y, inv_classes, n_components=2)
    print("Model KNN dan PCA berhasil dimuat!")

# Jalankan init sekali saat aplikasi jalan
init_model()

# ================= FLASK ROUTES =================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if knn is None:
        return "Error: Model belum siap. Cek apakah CSV sudah ada."

    try:
        # Ambil data dari form HTML
        temp = float(request.form['temp'])
        hum = float(request.form['hum'])
        wind = float(request.form['wind'])
        press = float(request.form['press'])
        
        sample = np.array([temp, hum, wind, press])
        
        # 1. Prediksi
        pred_idx = knn.predict_one(sample)
        hasil_cuaca = inv_classes[pred_idx]
        
        # 2. Visualisasi
        plot_image = viz.get_plot_url(sample)
        
        # 3. Info Tetangga Terdekat
        neighbors_info = knn.get_nearest_info(sample, inv_classes)

        return render_template('result.html', 
                               prediction=hasil_cuaca, 
                               plot_url=plot_image,
                               neighbors=neighbors_info,
                               input_data=sample)
                               
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

if __name__ == '__main__':
    app.run(debug=True)