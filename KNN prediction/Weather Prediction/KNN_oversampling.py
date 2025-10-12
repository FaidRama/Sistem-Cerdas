import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def oversample(X, y):
    # Hitung jumlah sample tiap kelas
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_resampled = []
    y_resampled = []
    
    for cls, count in zip(unique, counts):
        # Ambil semua data kelas ini
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        # Tentukan berapa kali perlu duplikasi
        n_repeat = max_count // count
        n_extra = max_count % count
        
        # Duplikasi penuh
        X_dup = np.tile(X_cls, (n_repeat, 1))
        y_dup = np.tile(y_cls, n_repeat)
        np.random.seed(1)
        # Tambah extra sample random
        if n_extra > 0:
            idx = np.random.choice(len(X_cls), n_extra, replace=True)
            X_extra = X_cls[idx]
            y_extra = y_cls[idx]
            X_dup = np.vstack([X_dup, X_extra])
            y_dup = np.hstack([y_dup, y_extra])
        
        X_resampled.append(X_dup)
        y_resampled.append(y_dup)
    
    # Gabungkan semua kelas
    X_balanced = np.vstack(X_resampled)
    y_balanced = np.hstack(y_resampled)
    
    # Acak supaya campur
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    
    return X_balanced[indices], y_balanced[indices]


# ===== Load Dataset =====
df = pd.read_csv(r"D:\coding\python\AI\weather_dataset.csv")

# Ambil hanya 3 kelas utama
df = df[df["Weather"].isin(["Clear", "Cloudy", "Rain"])]

X = df[["Temp_C", "Rel Hum_%", "Wind Speed_km/h", "Press_kPa"]].values
y = df["Weather"].values

# Encode label manual
classes = {label: idx for idx, label in enumerate(np.unique(y))}
inv_classes = {v: k for k, v in classes.items()}
y = np.array([classes[label] for label in y])


X_bal, y_bal = oversample(X, y)

print("Jumlah sebelum oversampling:", np.unique(y, return_counts=True))
print("Jumlah sesudah oversampling:", np.unique(y_bal, return_counts=True))

# ===== Split train/test manual (80/20) =====
np.random.seed(1)
indices = np.arange(len(X))
np.random.shuffle(indices)

#split = int(0.8 * len(X))
split = int(0.8 * len(X_bal))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X_bal[:split], X_bal[split:]
y_train, y_test = y_bal[:split], y_bal[split:]
#X_train, X_test = X[train_idx], X[test_idx]
#y_train, y_test = y[train_idx], y[test_idx]



# ===== KNN From Scratch =====
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict_one(self, x):
        # Hitung jarak Euclidean ke semua data latih
        #distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        #manhattan
        distances = np.sum(np.abs(self.X_train - x), axis=1)

        
        # Ambil k tetangga terdekat
        k_indices = distances.argsort()[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Voting mayoritas
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

# ===== Training & Evaluasi =====
knn = KNNClassifier(k=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Hitung akurasi manual
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Akurasi (KNN From Scratch):", accuracy)

# Confusion Matrix manual
conf_matrix = pd.crosstab(
    [inv_classes[i] for i in y_test],
    [inv_classes[i] for i in y_pred],
    rownames=["Asli"],
    colnames=["Prediksi"],
    margins=True
)
print("\nConfusion Matrix:\n", conf_matrix)

# ===== Prediksi Input Baru =====
# Format fitur: [Temp_C, Rel Hum_%, Wind Speed_km/h, Press_kPa]
sample = np.array([26.42, 28, 2, 1010.2])  # contoh input

pred_label = knn.predict_one(sample)
print("\nPrediksi cuaca untuk input", sample, "adalah:", inv_classes[pred_label])
