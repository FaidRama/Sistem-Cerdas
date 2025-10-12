import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ===== KNN From Scratch =====
class KNNClassifier:
    def __init__(self, k=9):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, x):
        # Hitung jarak Euclidean ke semua data latih
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        #manhattan
        #distances = np.sum(np.abs(self.X_train - x), axis=1)


        # Ambil k tetangga terdekat
        k_indices = distances.argsort()[:self.k]
        k_labels = self.y_train[k_indices]

        # Voting mayoritas
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    def show_nearest(self, x, n_neighbors=5):
        # Hitung jarak (pakai manhattan biar konsisten dengan kode kamu)
        distances = np.sum(np.abs(self.X_train - x), axis=1)

        # Urutkan dari jarak terkecil
        sorted_idx = np.argsort(distances)[:n_neighbors]

        print(f"\n{n_neighbors} data cuaca paling dekat dengan input:")
        print("=" * 60)
        for i, idx in enumerate(sorted_idx):
            print(f"{i+1}. Jarak = {distances[idx]:.3f} | Label = {inv_classes[self.y_train[idx]]}")
            print(f"   Fitur: {self.X_train[idx]}")
        print("=" * 60)

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

class visualisasi_pca:
    def __init__(self, X, y, inv_classes, n_components=2):
        self.X = X
        self.y = y
        self.n_components = n_components
        self.inv_classes = inv_classes
        self.X_pca, self.components, self.eigenvalues, self.mean_X = self._fit_pca()

    def _fit_pca(self):
        # 1. Mean centering
        mean_X = np.mean(self.X, axis=0)
        X_centered = self.X - mean_X

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Urutkan dari eigenvalue terbesar ke kecil
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        eigenvalues = eigenvalues[sorted_idx]

        # 5. Ambil n_components
        components = eigenvectors[:, :self.n_components]

        # 6. Proyeksikan data
        X_reduced = np.dot(X_centered, components)

        return X_reduced, components, eigenvalues, mean_X
    
    def visualisasi_input(self, p):
        input_centered = p - self.mean_X
        input_reduced = np.dot(input_centered, self.components)
        # Plot
        plt.figure(2, figsize=(8,6))
        for label in np.unique(y_train):
            plt.scatter(self.X_pca[self.y == label, 0],
                self.X_pca[self.y == label, 1],
                label=self.inv_classes[label], alpha=0.6)

        plt.scatter(input_reduced[0], input_reduced[1],
            color='black', marker='*', s=250, label='Sample (input baru)')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA (4 fitur → 2D) Weather Dataset (Scaled)")
        plt.legend()
        plt.show()




# ========================== Load Dataset ===============================================
df = pd.read_csv(r"python\AI\KNN prediction\Weather Prediction\weather_data_balanced.csv")

# Ambil hanya 3 kelas utama
df = df[df["Weather_Condition"].isin(["Sunny", "Rainy", "Stormy", "Cloudy"])]
X = df[["Temperature", "Humidity", "Wind_Speed", "Air_Pressure"]].values
y = df["Weather_Condition"].values

# Encode label manual
classes = {label: idx for idx, label in enumerate(np.unique(y))}
inv_classes = {v: k for k, v in classes.items()}
y = np.array([classes[label] for label in y])

# ===== Split train/test manual (80/20) =====
np.random.seed(1)
indices = np.arange(len(X))
np.random.shuffle(indices)

#split = int(0.8 * len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# =================================== Training & Evaluasi =============================
knn = KNNClassifier(k=9)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Hitung akurasi manual
accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
print("Akurasi (KNN From Scratch):", accuracy, " %")

# Confusion Matrix manual
conf_matrix = pd.crosstab(
    [inv_classes[i] for i in y_test],
    [inv_classes[i] for i in y_pred],
    rownames=["Asli"],
    colnames=["Prediksi"],
    margins=True
)

# ===== Prediksi Input Baru =====
# Format fitur: [Temp_C, Rel Hum_%, Wind Speed_km/h, Press_kPa]
Sample = np.array([26.42, 28, 2, 1010.2])  # contoh input

pred_label = knn.predict_one(Sample)
print("\nPrediksi cuaca untuk input", Sample, "adalah:", inv_classes[pred_label])

# ===== Visualisasi Confusion Matrix =====
cm = conf_matrix.iloc[:-1, :-1].values  # ambil tanpa total 'All'
labels = conf_matrix.columns[:-1]       # nama kolom prediksi

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')

# Tampilkan nilai dalam setiap kotak
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

# Label sumbu
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(conf_matrix.index[:-1])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title('Confusion Matrix')
plt.colorbar(im)
plt.show(block=False)



# ===== Standarisasi fitur sebelum PCA =====

# Jalankan PCA
viz = visualisasi_pca(X_train, y_train, inv_classes, n_components=2)
viz.visualisasi_input(Sample)

# ======== Input dari pengguna ========
try:
    temp = float(input("Masukkan suhu (°C): "))
    hum = float(input("Masukkan kelembapan (%): "))
    wind = float(input("Masukkan kecepatan angin (km/h): "))
    press = float(input("Masukkan tekanan udara (kPa): "))
except:
    print("input tidak valid, harap masukan angka")
    exit()

# ======== Bentuk jadi array numpy ========
sample = np.array([temp, hum, wind, press])

# ======== Prediksi menggunakan model KNN ========
pred_label = knn.predict_one(sample)

# ======== Tampilkan hasil prediksi ========
print("\nPrediksi cuaca untuk input", sample, "adalah:", inv_classes[pred_label])
viz.visualisasi_input(sample)
knn.show_nearest(sample, n_neighbors=9)
