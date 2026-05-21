"""
extract_features.py
-------------------
GTZAN veri setindeki her .wav dosyasını okuyup
akustik özelliklerini çıkarır ve features.csv olarak kaydeder.

Nasıl çalışır:
  python extract_features.py
Çıktı:
  features.csv  (her satır = 1 şarkı, son sütun = label/tür)
"""

import os
import csv
import librosa
import numpy as np

# ------------------------------------------------------------------
# AYARLAR
# ------------------------------------------------------------------
DATA_DIR   = "data/genres_original"   # GTZAN klasörünün yolu
OUTPUT_CSV = "features.csv"
SR         = 22050   # örnekleme hızı (Hz) — librosa varsayılanı
DURATION   = 30      # her klibin kaç saniyesini kullanalım

# ------------------------------------------------------------------
# BAŞLIK SATIRI
# Her özellik için bir sütun adı tanımlıyoruz.
# MFCC: 20 katsayı → mean + std = 40 sütun
# Spectral centroid, ZCR, chroma, rolloff, contrast: her biri mean+std
# Tempo: tek değer
# ------------------------------------------------------------------
header = []

# 20 MFCC için mean ve standart sapma
for i in range(1, 21):
    header += [f"mfcc{i}_mean", f"mfcc{i}_std"]

# Diğer özellikler
header += [
    "spectral_centroid_mean", "spectral_centroid_std",
    "zero_crossing_rate_mean", "zero_crossing_rate_std",
    "chroma_mean", "chroma_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_contrast_mean", "spectral_contrast_std",
    "tempo",
    "label"   # hedef değişken (müzik türü)
]


def extract_features(file_path):
    """
    Bir .wav dosyasından özellik vektörü çıkarır.

    Parametreler
    ------------
    file_path : str  — ses dosyasının tam yolu

    Döndürür
    --------
    list — sayısal özellikler listesi (label HARİÇ)
    """
    # 1) Ses dosyasını yükle
    #    y  = ses sinyali (numpy array)
    #    sr = örnekleme hızı (saniyede kaç sample)
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)

    features = []

    # 2) MFCC — Mel-Frequency Cepstral Coefficients
    #    İnsan kulağının frekans algısını taklit eder.
    #    20 katsayı → her katsayının ortalama ve std'si
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfccs:
        features += [np.mean(coef), np.std(coef)]

    # 3) Spectral Centroid — "ağırlık merkezi" frekansı
    #    Parlak sesler (metal, pop) yüksek, koyu sesler (blues) düşük değer alır
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features += [np.mean(sc), np.std(sc)]

    # 4) Zero Crossing Rate — sesin kaç kez sıfırı geçtiği
    #    Gürültülü ve perküsif sesler yüksek ZCR üretir
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features += [np.mean(zcr), np.std(zcr)]

    # 5) Chroma Features — 12 nota sınıfının enerjisi
    #    Harmonik içerik hakkında bilgi verir (tonal müzik ↔ perkusif)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [np.mean(chroma), np.std(chroma)]

    # 6) Spectral Rolloff — enerjinin %85'inin altında kaldığı frekans
    #    Yüksek değer → zengin harmonik içerik (metal, klasik)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features += [np.mean(rolloff), np.std(rolloff)]

    # 7) Spectral Contrast — frekans bantları arasındaki enerji farkı
    #    Müzik türleri arasında ayırt edici güçlü bir özelliktir
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features += [np.mean(contrast), np.std(contrast)]

    # 8) Tempo — BPM (beats per minute) tahmini
    #    Tek bir skaler değer
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa 0.10+ bazen array döndürüyor, float'a çevir
    features.append(float(np.atleast_1d(tempo)[0]))

    return features


# ------------------------------------------------------------------
# ANA DÖNGÜ
# ------------------------------------------------------------------
print("Özellik çıkarımı başladı...\n")

with open(OUTPUT_CSV, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)   # başlık satırını yaz

    genres = sorted(os.listdir(DATA_DIR))

    for genre in genres:
        genre_dir = os.path.join(DATA_DIR, genre)

        if not os.path.isdir(genre_dir):
            continue   # dosya varsa atla

        wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        print(f"  [{genre}] → {len(wav_files)} dosya işleniyor...")

        for idx, file_name in enumerate(wav_files):
            file_path = os.path.join(genre_dir, file_name)
            try:
                feats = extract_features(file_path)
                writer.writerow(feats + [genre])   # label'ı sona ekle
            except Exception as e:
                print(f"    ⚠ Hata ({file_name}): {e}")

        print(f"  [{genre}] ✓ tamamlandı")

print(f"\n✅ Tüm özellikler '{OUTPUT_CSV}' dosyasına kaydedildi.")
print("Sonraki adım: python train_model.py")
