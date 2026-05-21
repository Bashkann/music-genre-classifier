"""
train_model.py
--------------
features.csv'yi okur, 3 farklı model eğitir,
en iyisini seçer ve model.pkl olarak kaydeder.

Nasıl çalışır:
  python train_model.py
Çıktı:
  model.pkl      — kaydedilen model
  scaler.pkl     — normalizasyon parametreleri (Streamlit'te de lazım)
  Ekrana confusion matrix ve F1-score tablosu
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ------------------------------------------------------------------
# 1) VERİYİ YÜKLE
# ------------------------------------------------------------------
print("Veri yükleniyor...")
df = pd.read_csv("features.csv")

print(f"  Toplam satır (şarkı): {len(df)}")
print(f"  Toplam sütun (özellik + label): {len(df.columns)}")
print(f"  Sınıf dağılımı:\n{df['label'].value_counts()}\n")

# ------------------------------------------------------------------
# 2) X ve y'yi AYIR
# ------------------------------------------------------------------
X = df.drop(columns=["label"])   # özellik matrisi
y = df["label"]                  # hedef etiket

# Label'ları sayıya çevir: blues→0, classical→1, ...
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------------------------------------------
# 3) NORMALİZASYON (StandardScaler)
#
# Neden gerekli?
#   MFCC değerleri -100 ile +100 arasında olabilirken
#   tempo değerleri 60-200 arasında.
#   Ölçek farkı SVM ve KNN'i olumsuz etkiler.
#   StandardScaler her sütunu mean=0, std=1 yapacak şekilde ölçekler.
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 4) TRAIN / TEST AYRIMI
#
# stratify=y_encoded → her sınıftan eşit oranda test'e gider
# Bu GTZAN'ın veri sızıntısı sorununu kısmen önler
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)
print(f"Eğitim seti: {len(X_train)} örnek")
print(f"Test seti  : {len(X_test)} örnek\n")

# ------------------------------------------------------------------
# 5) MODELLERİ TANIMLA
# ------------------------------------------------------------------
models = {
    # Random Forest:
    #   Çok sayıda karar ağacı oluşturur, sonuçları oylar.
    #   Overfitting'e karşı dirençlidir, yorumlanması kolaydır.
    "Random Forest": RandomForestClassifier(
        n_estimators=200,   # kaç ağaç?
        max_depth=None,     # ağaçlar sınırsız derinleşebilir
        random_state=42,
        n_jobs=-1           # tüm CPU çekirdeklerini kullan
    ),

    # SVM (Support Vector Machine):
    #   Yüksek boyutlu uzaylarda güçlüdür.
    #   kernel="rbf" → doğrusal olmayan sınırlar çizebilir
    #   C=10 → hata toleransı (büyük C = daha sıkı fit)
    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True   # güven skoru için
    ),

    # KNN (K-Nearest Neighbors):
    #   Test örneğine en yakın k komşunun etiketine bakarak karar verir.
    #   Basit ama eğitim verisi büyüdükçe yavaşlar.
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ------------------------------------------------------------------
# 6) EĞİT VE KARŞILAŞTIR (Cross-Validation)
#
# StratifiedKFold → her fold'da sınıf oranı korunur
# cv=5 → 5 farklı train/val bölümlemesi dene, ortalamasını al
# ------------------------------------------------------------------
print("=" * 55)
print(f"{'Model':<20} {'CV Accuracy':>12} {'Std Dev':>10}")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"{name:<20} {scores.mean():.4f}       ±{scores.std():.4f}")

print("=" * 55)

# ------------------------------------------------------------------
# 7) EN İYİ MODELİ SEÇ ve TEST SETİNDE DEĞERLENDİR
# ------------------------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 En iyi model: {best_model_name} (CV accuracy: {results[best_model_name]:.4f})")
print("Test seti üzerinde eğitiliyor ve değerlendiriliyor...\n")

# Tüm eğitim verisiyle yeniden eğit
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Test accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)\n")

# Sınıf bazlı rapor
print("Sınıf bazlı rapor:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# ------------------------------------------------------------------
# 8) CONFUSION MATRIX GÖRSELİ
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — {best_model_name} (Test Accuracy: {test_acc:.2%})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("📊 Confusion matrix 'confusion_matrix.png' olarak kaydedildi.")

# ------------------------------------------------------------------
# 9) MODELİ ve SCALER'I KAYDET
#
# Streamlit uygulaması bu dosyaları yükleyecek.
# İkisini de kaydetmeyi UNUTMA — scaler olmadan tahmin yanlış olur!
# ------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ model.pkl, scaler.pkl, label_encoder.pkl kaydedildi.")
print("Sonraki adım: streamlit run app.py")
