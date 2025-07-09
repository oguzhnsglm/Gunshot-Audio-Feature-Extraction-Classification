# 🔫 Gunshot Audio Feature Extraction & Classification

Bu proje, **farklı silah seslerini** sınıflandırmak için geliştirilmiş bir **ses tabanlı makine öğrenmesi sistemi**dir.  
Sistem, ham ses dosyalarından (WAV/MP3) çeşitli özellikler (MFCC, ZCR, Spectral özellikler, Chroma vb.) çıkarır ve bunları kullanarak bir **sinir ağı (Dense Neural Network)** yardımıyla hangi silaha ait olduğunu tahmin eder.

---

## 🎯 Projenin Teorisi ve Çalışma Mantığı

- **Ses Sinyalleri:** 
  - Silah sesleri, frekans spektrumunda farklı enerji dağılımlarına sahiptir.
  - Örneğin bir tabanca ile tüfeğin patlama karakteristiği farklıdır.

- **Özellik Çıkarımı (Feature Extraction):**
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Spectral Centroid, Roll-Off, Bandwidth, Flatness
  - Chroma STFT
  - Zero Crossing Rate
  - RMS Energy
  - Poly Features
- Bu özellikler sayesinde her ses dosyası **128-200 boyutlu** bir vektöre dönüştürülür.

- **Özellik Seçimi (Feature Selection):**
  - Çıkarılan çok sayıda özelliğin en anlamlılarını seçmek için çeşitli algoritmalar (Information Gain, ANOVA F-Value, Random Forest Importance, RFE vb.) kullanılır.
  - Bu sayede boyut azaltma (dimensionality reduction) yapılır ve model daha hızlı, daha az overfitting eğilimi ile çalışır.
  - Örneğin en iyi **128 özellik** seçilerek `selected_features.csv` dosyası oluşturulur.

- **Model:**
  - Seçilen özellikler bir **Dense Neural Network (MLP)**’ye verilerek sınıflandırma yapılır.
  - Activation: `ReLU`, Çıkış: `Softmax`
  - Kayıp fonksiyonu: `Categorical Crossentropy`

---

## 📂 Veri Seti ve Split Yapısı

- Dataset toplamda **8 farklı silah türü** ve her birinden **~100 örnek** olmak üzere yaklaşık **800 ses dosyası** içerir.
- Split oranı:
  - **%80 Train**
  - **%20 Validation**
  
Bu sayede model:
- Eğitim sırasında `train` datasıyla öğrenir,
- `validation` ile overfitting kontrol edilir.

> 🔍 Projede ayrıca `feature_selection.py` çalıştırılarak en iyi 128 öznitelik belirlenip 
`selected_features.csv` dosyası oluşturulur ve eğitim/test bu dosya üzerinden yapılır.

---

## 🔧 Kurulum & Ortam Gereksinimleri

Bu proje Python üzerinde çalışır ve temel kütüphaneleri şunlardır:

- numpy, pandas
- librosa
- scikit-learn
- tensorflow / keras
- matplotlib

### 🐍 Örnek pip kurulumu
```bash
pip install numpy pandas librosa scikit-learn tensorflow matplotlib
