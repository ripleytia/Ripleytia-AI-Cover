<div align="center">
<img src="logo.jpg" width="200">
<br>
<a href="https://www.patreon.com/Ripleytia"><img src="https://img.shields.io/badge/Patreon-Support_Me-F96854?style=for-the-badge&logo=patreon" alt="Patreon"></a>
</div>

# 🚀 Ripleytia AI Sound Studio (AI Ses Stüdyosu) - Cover, Otomatik Dataset ve Ses Modelleme Merkezi

**Ripleytia AI Studio**, artık yalnızca bir AI Cover aracı değil; **AI Cover Üretimi**, **Otomatik Dataset Hazırlayıcı** ve **Profesyonel Ses Modelleme (RVC Eğitim Stüdyosu)** özelliklerini tek bir çatı altında birleştiren kapsamlı bir **AI Ses Stüdyosu (AI Sound Studio)** haline geldi!

---

## 🔥 Büyük Güncelleme ile Gelen Devrimsel Yenilikler

### 1. 🧠 Profesyonel Ses Modelleme (Eğitim Stüdyosu)
Applio'nun güçlü eğitim altyapısı, karmaşık ayarlarından arındırılarak çok daha kolay, anlaşılır ve gelişmiş özelliklerle doğrudan uygulamamıza entegre edildi:
*   **🎯 Tek Tıkla Hazır Eğitim Presetleri:**
    *   **Dengeli (Önerilen - Stüdyo Standart):** 40kHz HiFi-GAN mimarisi ile hem AI Cover hem de W-Okada canlı ses değiştirici için en dengeli stüdyo kalitesi.
    *   **Maksimum Kalite (Ultra Detay & Nefes):** 48kHz Ultra HD örnekleme ve **TITAN-Medium** ön eğitim modeli ile en ince nefes ve tiz detaylarını yakalar.
    *   **Hızlı Render (Pratik & Hızlı Eğitim):** 32kHz örnekleme, **FCPE** hızlı pitch çıkarıcı ve **Ov2Super** ön eğitim modeli ile çok kısa sürede eğitim.
    *   **W-Okada / Canlı Konuşma & Discord (Özel):** Gerçek zamanlı ses değiştiriciler için mikrofonda dip ses/nefes patlamasını önleyen (`include_mutes=4`) özel profil ve **Snowie V3.1** ön eğitim modeli.
*   **🔬 Applio Ötesi Ekstra Kalite ve Ön Eğitim Modelleri (Otomatik İndirmeli):**
    *   **Özel Pretrain Kütüphanesi:** `Standart Applio`, `TITAN-Medium`, `Snowie V3.1`, `Ov2Super`, `KLM 4.9 HFG`, `RIN_E3`, `SingerPreTrain`.
    *   **Gelişmiş Vocoder & Embedder Seçenekleri:** `HiFi-GAN` ve `RefineGAN` sentezleyiciler; `contentvec`, `spin-v2`, `spin`, `chinese/japanese/korean-hubert-base` ses kimliği modelleri; `rmvpe`, `fcpe`, `crepe`, `crepe-tiny` pitch algoritmaları.
*   **🧮 Akıllı Epoch Hesaplayıcı:** Datasetinizin net süresini saniyesi saniyesine analiz ederek Overfitting (ses bozulması) olmadan gereken en ideal Epoch sayısını otomatik hesaplar.
*   **📥 Otomatik Stüdyo Kurulumu:** Eğitim tamamlandığı anda üretilen `.pth` ve `.index` dosyaları otomatik olarak `rvc_models` klasörüne kurulur ve **AI Cover Stüdyosu** sekmesinde anında seçilebilir hale gelir!

---

### 2. 📦 Otomatik Dataset Hazırlayıcı (Auto Dataset Builder)
Ses modeli eğitmek için saatlerce Audacity veya UVR5 ile uğraşma devri bitti!
*   **Çoklu Yerel Dosya & Çoklu YouTube Linki Desteği:** Bilgisayarınızdan istediğiniz kadar ses/video dosyası (`MP3, WAV, FLAC, MP4, MKV`) seçin veya birden fazla YouTube video linkini alt alta yapıştırın.
*   **Otomatik Arka Plan, Geri Vokal ve Yankı Temizleme:** Seçilen tüm kaynakları sırayla indirir, **UVR5 Special 2** (Mel-Band Roformer) ile arka plan müziğini siler, **Karaoke Roformer** ile geri vokalleri/diğer konuşmacıları ayıklar ve **Anvuew De-Reverb (SDR 19.17)** ile oda yankısını tamamen kurutur.
*   **Otomatik Boşluk Kesme, Ses Dengeleme ve Birleştirme:** 75Hz Highpass filtre uygular, konuşma/şarkı arasındaki ölü sessizlikleri otomatik keser, tüm parçaların ses seviyesini (Normalization) eşitler ve temiz vokalleri birleştirerek toplam kaç dakika net vokal çıktığını (Örn: *24 dk 40 sn*) raporlar.
*   **Gece Bırakma Modu (Tam Otomasyon):** İsterseniz *"Dataset Hazırlanır Hazırlanmaz Tam Otomatik Model Eğitimini de Başlat"* seçeneğini işaretleyerek YouTube linklerini yapıştırıp sabaha hazır eğitilmiş ses modelinizi alabilirsiniz!

---

### 3. 🎙️ UVR5 Special 2 (4 Aşamalı Ultra Roformer Vokal Ayırma Hattı)
AI Cover ve Dataset hazırlama için dünyadaki en güncel 4 Roformer modeli tek bir zincirde birleştirildi:
1.  **Mel-Band Roformer Vocals (Becruily):** Kusursuz vokal ayrıştırma.
2.  **Mel-Band Roformer Instrumental (Becruily):** Sıfır vokal sızıntılı enstrümantal ayrıştırma.
3.  **Mel-Band Roformer Karaoke (Aufr33 & Viperx - SDR 10.19):** Ana vokali geri vokallerden (harmoni/koro) bıçak gibi ayırır.
4.  **Mel-Band Roformer De-Reverb (Anvuew - SDR 19.17):** En yankılı vokalleri bile stüdyo mikrofonu kuruluğuna kavuşturur.

---

## 🛠️ Çalıştırma

Uygulamayı başlatmak için proje klasöründeki `Start-SuperGUI.bat` dosyasına çift tıklamanız veya terminalden aşağıdaki komutu çalıştırmanız yeterlidir:

```bash
.\venv\Scripts\python.exe src\super_gui.py
```

---

## ❤️ Projeye Destek Olun
Bu proje tamamen ücretsiz ve açık kaynaklıdır. Geliştirmelerin devam etmesi ve bana destek olmak için [Patreon sayfamı](https://www.patreon.com/Ripleytia) ziyaret edebilirsiniz!
