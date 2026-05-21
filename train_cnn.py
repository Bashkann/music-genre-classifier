"""
Müzik Türü Sınıflandırma — CNN (PyTorch)
Mel-Spectrogram görüntüleri üzerinde eğitim
"""

import os
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─── Ayarlar ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/genres_original"
IMG_SIZE    = 128          # 128×128 piksel
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# MPS = Apple Silicon GPU. Varsa otomatik kullanır, yoksa CPU'ya düşer.

print(f"Kullanılan cihaz: {DEVICE}")

# ─── 1. Mel-Spectrogram Üretici ──────────────────────────────────────────────
def wav_to_melspec(file_path, img_size=IMG_SIZE):
    """
    .wav dosyasını IMG_SIZE×IMG_SIZE numpy dizisine çevirir.
    librosa → mel-spectrogram → dB ölçeği → 0-255 normalize → PIL Image
    """
    y, sr = librosa.load(file_path, duration=30, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)       # dB ölçeğine çevir
    # 0-255 arasına normalize et (görüntü gibi)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    img = Image.fromarray(mel_norm).resize((img_size, img_size))
    return np.array(img)

# ─── 2. Veri Yükleme ─────────────────────────────────────────────────────────
print("\nMel-spectrogramlar üretiliyor (bu 5-10 dakika sürebilir)...")

X, y_labels = [], []
genres = sorted(os.listdir(DATA_PATH))
genres = [g for g in genres if os.path.isdir(os.path.join(DATA_PATH, g))]

for genre in genres:
    genre_path = os.path.join(DATA_PATH, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    print(f"  [{genre}] → {len(files)} dosya işleniyor...")
    for fname in files:
        fpath = os.path.join(genre_path, fname)
        try:
            img_array = wav_to_melspec(fpath)
            X.append(img_array)
            y_labels.append(genre)
        except Exception as e:
            print(f"    ⚠ Atlandı: {fname} ({e})")
    print(f"  [{genre}] ✓")

X = np.array(X)  # (N, 128, 128)
print(f"\nToplam örnek: {len(X)}")

# Label encode
le = LabelEncoder()
y = le.fit_transform(y_labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Eğitim: {len(X_train)} | Test: {len(X_test)}")

# ─── 3. PyTorch Dataset ───────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # (N, H, W) → float32 tensor, normalize
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        # unsqueeze(1): kanal boyutu ekle → (N, 1, H, W)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Data augmentation (sadece train'e)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),   # zaman ekseninde flip
    transforms.RandomErasing(p=0.2),           # küçük parçaları sil (dropout gibi)
])

train_dataset = MelDataset(X_train, y_train, transform=train_transform)
test_dataset  = MelDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ─── 4. CNN Modeli ────────────────────────────────────────────────────────────
class MusicCNN(nn.Module):
    """
    3 katmanlı konvolüsyon + tam bağlantılı kafası olan CNN.
    Girdi: (batch, 1, 128, 128) gri-tonlama mel-spectrogram
    Çıktı: (batch, 10) sınıf skoru
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 1→32 kanal
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1,128,128)→(32,128,128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # →(32,64,64)

            # Blok 2: 32→64 kanal
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # →(64,32,32)

            # Blok 3: 64→128 kanal
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # →(128,16,16)

            # Blok 4: 128→256 kanal
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                 # →(256,4,4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # →(256*4*4 = 4096)
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),              # overfitting önleme
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # 10 tür
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MusicCNN(num_classes=len(le.classes_)).to(DEVICE)
print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters()):,}")

# ─── 5. Eğitim ───────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
# weight_decay: L2 regularization, overfitting'i azaltır

# Learning rate scheduler: her 10 epoch'ta lr'yi 0.5'e indir
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses, train_accs = [], []
test_accs = []

print(f"\nEğitim başlıyor ({EPOCHS} epoch, cihaz: {DEVICE})...")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    # — Eğitim —
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # — Değerlendirme —
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    test_accs.append(test_acc)
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

print("=" * 55)

# ─── 6. Sonuç Raporu ─────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

final_acc = accuracy_score(all_labels, all_preds)
print(f"\n🏆 CNN Test Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
print("\nSınıf bazlı rapor:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ─── 7. Grafik ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train Loss")
ax1.set_title("Eğitim Kaybı")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(train_accs, label="Train Acc")
ax2.plot(test_accs,  label="Test Acc")
ax2.set_title("Doğruluk")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("cnn_training.png", dpi=150)
print("\n📊 Eğitim grafiği 'cnn_training.png' olarak kaydedildi.")

# ─── 8. Model Kaydet ─────────────────────────────────────────────────────────
torch.save(model.state_dict(), "model_cnn.pth")
with open("label_encoder.pkl", "rb") as f:
    pass  # mevcut label_encoder zaten var, üzerine yazmıyoruz

print("\n✅ model_cnn.pth kaydedildi.")
print(f"\n{'='*40}")
print(f"KARŞILAŞTIRMA:")
print(f"  SVM (ML)  → %73.5 test accuracy")
print(f"  CNN       → %{final_acc*100:.1f} test accuracy")
print(f"{'='*40}")
print("\nSonraki adım: streamlit run app.py  (CNN'i de gösterecek şekilde güncelleyeceğiz)")
