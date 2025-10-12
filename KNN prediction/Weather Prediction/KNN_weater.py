import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ===== Load Dataset =====
df = pd.read_csv(r"D:\coding\python\AI\weather_data_balanced.csv")

# Ambil hanya 3 kelas utama
df = df[df["Weather_Condition"].isin(["Sunny", "Cloudy", "Rainy", "Stormy"])]

X = df[["Temperature","Humidity","Air_Pressure","Wind_Speed"]].values
y = df["Weather_Condition"].values

# Encode label manual
classes = {label: idx for idx, label in enumerate(np.unique(y))}
inv_classes = {v: k for k, v in classes.items()}
y = np.array([classes[label] for label in y])


# ===== Split train/test manual (80/20) =====
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ===== KNN From Scratch =====
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict_one(self, x):
        # Hitung jarak Euclidean ke semua data latih
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        #manhattan
        #distances = np.sum(np.abs(self.X_train - x), axis=1)
        #print(self.X_train)
        #print(x)
        print(distances)

        
        # Ambil k tetangga terdekat
        k_indices = distances.argsort()[:self.k]
        k_labels = self.y_train[k_indices]
       
        
        # Voting mayoritas
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

# ===== Training & Evaluasi =====
knn = KNNClassifier(k=5)
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
sample = np.array([25, 70, 10, 1010.2])  # contoh input

pred_label = knn.predict_one(sample)
print("\nPrediksi cuaca untuk input", sample, "adalah:", inv_classes[pred_label])


