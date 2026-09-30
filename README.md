# 🚀 Ripleytia AI Automated Cover (v1.0.0) - Evrensel Kararlı Sürüm

<p align="center">
  <img src="https://raw.githubusercontent.com/ripleytia/ripleytia-ai-cover/main/assets/logo.jpg" width="250" alt="Ripleytia AI Cover Logo">
</p>

Ripleytia'nın **Gothic Purple** siberpunk estetiğine sahip profesyonel ses stüdyosu artık kullanıma hazır! Meta Demucs v4 ve RVC motorlarının gücüyle, evrensel yapay zeka cover'larınızı espor stüdyosu kalitesinde (sıfır detone, sıfır cızırtı) üretin.

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-blue?style=for-the-badge&logo=windows" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blueviolet?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/UI-PyQt6-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sürüm-v1.0.0%20(Stable)-purple?style=for-the-badge" />
</p>

## 🌟 Öne Çıkan Özellikler (v1.0.0 Stable)

### 1. 🎵 Gelişmiş Yapay Zeka Motorları
* **Meta Demucs v4 Ayrıştırma:** Şarkıları (MP3/WAV) kayıpsız olarak Vokal ve Altyapı (Instrumental) olarak otomatik böler.
* **RVC v2 Inference (RMVPE Algoritması):** Pürüzsüz ve sıfır detone hedef vokal dönüşümü yapar.
* **Device Guard:** Otomatik NVIDIA CUDA donanım hızlandırma tespiti yapar, yoksa güvenli şekilde CPU fallback moduna düşer.

### 2. 🎛️ Pydub Stüdyo Masteringi
* **Temporal Alignment:** Ses dosyalarındaki faz kaymalarını (başlangıç sessizliklerini) budayarak milisaniyelik senkronizasyon sağlar.
* **RMS Auto-Gain:** Vokal ve enstrümantal seslerin ortalama desibel enerjilerini (RMS) hesaplayıp vokali daima müziğin **+3 dB** üstünde (altın oran) tutar.
* **Anti-Clipping / Limitör:** Dijital ses patlamalarını önlemek için stüdyo tipi dinamik kompresör ve `-1.0 dBFS True Peak` limitörü uygular.

### 3. ⚡ Process-Reactive VFX Arayüzü (PyQt6)
* **Animasyonlu Splash Screen:** Uygulama açılırken çerçevesiz ve saydam, merkezden dışa doğru yayılan şimşek emisyonlarına sahip giriş ekranı.
* **Dinamik Şimşek Motoru (Midpoint Displacement):** Yapay zeka ses işlerken ekranda çakan, hızlanan ve ritmik nefes alan Neon Mor / Siber Pembe prosedürel şimşekler.
* **Entegre Oynatıcı:** Dönüştürülen Cover şarkıyı anında dinlemek için native `QtMultimedia` oynatıcısı.

## 📦 Kurulum ve Kullanım (Evrensel Dağıtım)
Bilgisayarınızda Python, PyTorch veya FFmpeg kurulu olmasına gerek yoktur! Gömülü FFmpeg ve PyInstaller altyapısı sayesinde **Tak-Çalıştır (Portable)** mimariye sahiptir.

1. `Releases` sekmesinden **`Ripleytia_AI_Cover_v1.0.0_Setup.zip`** dosyasını indirin.
2. ZIP dosyasını dilediğiniz bir klasöre çıkartın.
3. Klasör içindeki `Ripleytia_AI_Cover.exe` dosyasına çift tıklayarak uygulamayı başlatın.
*(Not: Kendi modellerinizi `/models` klasörüne, icon vs gibi kaynakları `/assets` klasörüne ekleyebilirsiniz.)*

## 🛠️ Geliştiriciler İçin
Projeyi kaynak kodundan çalıştırmak isterseniz:
```bash
# Gereksinimleri Yükle (CUDA destekli evrensel kilidi içerir)
pip install -r requirements.txt

# Uygulamayı Başlat
python main.py

# Tek Tıkla PyInstaller (Dağıtım) Derlemesi
python build_deploy.py
```
