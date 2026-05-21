# Müzik Türü Sınıflandırma Sistemi

Bu proje, yüklenen bir ses dosyasının (.wav) hangi müzik türüne ait olduğunu tahmin eden bir web uygulamasıdır. Projede makine öğrenimi ve derin öğrenme yaklaşımları karşılaştırılmıştır.

## Kullanılan Veri Seti
Model eğitiminde Kaggle'da bulunan **GTZAN Music Genre Dataset** kullanılmıştır. Veri setinde 10 farklı müzik türü ve her tür için 100 adet 30 saniyelik ses dosyası bulunmaktadır. Veri sızıntısını (data leakage) önlemek için train/test ayrımı stratify parametresi ile yapılmıştır.

## Kullanılan Modeller ve Sonuçlar
Projede iki farklı model yaklaşımı test edilmiştir:

1. **SVM Modeli (%73.5 Doğruluk):** Ses dosyalarından `librosa` kütüphanesi ile MFCC, spectral centroid, zero crossing rate ve tempo gibi matematiksel özellikler çıkarılarak eğitilmiştir.
2. **CNN Modeli (%68.5 Doğruluk):** Ses dosyaları PyTorch kullanılarak 128x128 boyutlarında Mel-Spektrogram (görüntü) formatına dönüştürülmüş ve evrişimli sinir ağı ile eğitilmiştir.

**Not:** Veri seti boyutu kısıtlı olduğu için (sınıf başı sadece 100 örnek), geleneksel bir makine öğrenimi algoritması olan SVM, veri açlığı çeken CNN modeline göre daha iyi genelleme yapmış ve daha yüksek doğruluk oranına ulaşmıştır. 
## Proje Demo Videosu

[![Müzik Türü Sınıflandırma Demo](https://img.youtube.com/vi/8weYqSHiI00/maxresdefault.jpg)](https://youtu.be/8weYqSHiI00)

*(Videoyu izlemek için yukarıdaki görsele tıklayabilirsiniz.)*

## Projeyi Çalıştırma Adımları

Projeyi çalıştırmak için öncelikle bir sanal ortam (virtual environment) kullanmanız önerilir.

1. Sanal ortamı aktif edin (Mac/Linux için):
source venv/bin/activate

2. Gerekli kütüphaneleri yükleyin:
pip install librosa numpy pandas scikit-learn torch torchvision streamlit matplotlib

3. Arayüzü başlatın:
streamlit run app.py
