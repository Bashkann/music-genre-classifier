"""
app.py — Streamlit Web Arayüzü (Premium UX - Sidebar Fix 2)
SVM ve CNN model seçimi destekli versiyon
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use("Agg")

# ------------------------------------------------------------------
# SAYFA AYARLARI VE ÖZEL CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Müzik Türü Sınıflandırıcı",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded" # Menü varsayılan olarak açık başlayacak
)

# Premium Tasarım için CSS
st.markdown("""
    <style>
    /* Sadece Deploy butonunu gizle, header'ı görünür bırak ki menü butonu (> ikon) kaybolmasın */
    .stAppDeployButton {display:none;}
    
    /* Özel Tasarım Sınıfları */
    .main-title {
        font-weight: 900;
        color: #1DB954; /* Ana Vurgu Rengi */
        margin-bottom: 0px;
        padding-top: 10px;
    }
    .sub-title {
        color: #B3B3B3;
        margin-bottom: 30px;
        font-size: 16px;
    }
    .pred-box {
        background: linear-gradient(135deg, #1DB954 0%, #121212 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(29, 185, 84, 0.2);
        margin-bottom: 25px;
        border: 1px solid rgba(29, 185, 84, 0.3);
    }
    .pred-text {
        font-size: 60px;
        font-weight: 900;
        color: white;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .pred-label {
        color: #e0e0e0;
        font-size: 16px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MODEL YÜKLEME FONKSİYONLARI
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_svm():
    """SVM modelini, ölçekleyiciyi ve etiket kodlayıcıyı yükler."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, scaler, le

@st.cache_resource(show_spinner=False)
def load_cnn():
    """Eğitilmiş CNN modelini (PyTorch) yükler."""
    import torch
    import torch.nn as nn

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    class MusicCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn = MusicCNN(num_classes=len(le.classes_)).to(device)
    cnn.load_state_dict(torch.load("model_cnn.pth", map_location=device))
    cnn.eval()
    return cnn, le, device

# ------------------------------------------------------------------
# ÖZELLİK ÇIKARIM FONKSİYONLARI
# ------------------------------------------------------------------
def extract_features_svm(y, sr):
    """Gelen ses sinyalinden matematiksel akustik özellikleri (MFCC vb.) çıkarır."""
    features = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfccs:
        features += [np.mean(coef), np.std(coef)]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features += [np.mean(sc), np.std(sc)]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features += [np.mean(zcr), np.std(zcr)]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [np.mean(chroma), np.std(chroma)]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features += [np.mean(rolloff), np.std(rolloff)]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features += [np.mean(contrast), np.std(contrast)]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.atleast_1d(tempo)[0]))
    return np.array(features)

def wav_to_tensor(y, sr):
    """Ses sinyalini CNN modelinin okuyabileceği Mel-Spektrogram tensorüne dönüştürür."""
    import torch
    from PIL import Image
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    img = np.array(Image.fromarray(mel_norm).resize((128, 128)))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

# ------------------------------------------------------------------
# YAN MENÜ (SIDEBAR) TASARIMI
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 Müzik AI")
    st.markdown("---")
    st.markdown("### ⚙️ Yapay Zeka Ayarları")
    
    model_choice = st.radio(
        "Analiz Motorunu Seçin:",
        ["SVM (Klasik Makine Öğrenimi)", "CNN (Derin Öğrenme Ağları)"]
    )
    use_cnn = model_choice.startswith("CNN")
    
    st.markdown("---")
    st.markdown("### ℹ️ Proje Hakkında")
    with st.expander("Modeller Nasıl Çalışır?"):
        st.write("**SVM:** Sesi matematiksel değerlere (Tempo, Melodi) böler. Küçük veri setlerinde çok başarılıdır (%73.5 Doğruluk).")
        st.write("**CNN:** Sesi bir frekans haritasına (görüntüye) çevirir ve sanki bir fotoğrafı tanıyormuş gibi öğrenir (%68.5 Doğruluk).")
        
    st.markdown("---")
    st.caption("👨‍💻 Geliştirici: Kaan\n\n📊 Veri Seti: GTZAN")

# ------------------------------------------------------------------
# ANA EKRAN (MAIN CONTENT)
# ------------------------------------------------------------------
col_text, col_upload = st.columns([1.5, 1], gap="large")

with col_text:
    st.markdown("<h1 class='main-title'>🎵 Müzik Türü Sınıflandırıcı</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Dijital ses sinyallerinin DNA'sını yapay zeka ile analiz edin.</p>", unsafe_allow_html=True)

with col_upload:
    uploaded_file = st.file_uploader(
        "🎧 Analiz edilecek müziği buraya sürükleyin (.wav)",
        type=["wav"]
    )

# ------------------------------------------------------------------
# ANALİZ VE SONUÇ EKRANI
# ------------------------------------------------------------------
if uploaded_file is not None:
    st.markdown("---")
    
    # Çok şık bir container içinde ses oynatıcı
    with st.container():
        st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🎧 Yapay Zeka müziğin akustik özelliklerini ayrıştırıyor..."):
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=30, mono=True)

        if use_cnn:
            import torch
            import torch.nn.functional as F
            try:
                cnn, le, device = load_cnn()
                tensor = wav_to_tensor(y, sr).to(device)
                with torch.no_grad():
                    logits = cnn(tensor)
                    proba_tensor = F.softmax(logits, dim=1)[0]
                proba = proba_tensor.cpu().numpy()
                pred_idx = int(np.argmax(proba))
                pred_label = le.classes_[pred_idx]
                confidence = proba[pred_idx]
            except FileNotFoundError:
                st.error("❌ model_cnn.pth bulunamadı.")
                st.stop()
        else:
            try:
                model, scaler, le = load_svm()
                features = extract_features_svm(y, sr).reshape(1, -1)
                features_scaled = scaler.transform(features)
                pred_idx = model.predict(features_scaled)[0]
                pred_label = le.inverse_transform([pred_idx])[0]
                proba = model.predict_proba(features_scaled)[0]
                confidence = proba[pred_idx]
            except FileNotFoundError:
                st.error("❌ model.pkl bulunamadı.")
                st.stop()

        # Olasılıkları DataFrame'e çevirip en yüksekten düşüğe sıralama
        proba_df = pd.DataFrame({
            "Tür": [cls.capitalize() for cls in le.classes_],
            "Olasılık": proba
        }).sort_values("Olasılık", ascending=False)

        top_3 = proba_df.head(3)

        # SEKMELER (TABS)
        tab_sonuc, tab_grafik = st.tabs(["🎯 Tahmin Raporu", "🌊 Spektral Dalga Analizi"])

        with tab_sonuc:
            col_kutu, col_detay = st.columns([1, 1], gap="large")
            
            with col_kutu:
                # Zirve Tahmin Kutusu
                st.markdown(f"""
                    <div class="pred-box">
                        <p class="pred-label">En Yüksek İhtimal</p>
                        <p class="pred-text">{pred_label}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_detay:
                st.markdown("### 📊 En Güçlü 3 Tahmin")
                # Top 3 için şık ilerleme çubukları
                for index, row in top_3.iterrows():
                    st.write(f"**{row['Tür']}** - %{row['Olasılık']*100:.1f}")
                    st.progress(float(row['Olasılık']))

        with tab_grafik:
            st.markdown("### Modelin Gördüğü Spektral Veriler")
            st.caption("Aşağıdaki grafikler, yapay zekanın duyduğu sesi nasıl sayılara ve haritalara döktüğünü gösterir.")
            
            plt.style.use('dark_background')
            fig, axes = plt.subplots(1, 2, figsize=(15, 4))
            fig.patch.set_facecolor("#121212")
            for ax in axes:
                ax.set_facecolor("#121212")

            # Dalga formu (Sol Grafik)
            librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#1DB954")
            axes[0].set_title("Zaman Serisi (Dalga Formu)", color="white")
            axes[0].tick_params(colors="white")

            # Melspectrogram (Sağ Grafik)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                            ax=axes[1], cmap="magma")
            axes[1].set_title("Frekans Haritası (Mel-Spektrogram)", color="white")
            axes[1].tick_params(colors="white")

            plt.tight_layout()
            st.pyplot(fig)
