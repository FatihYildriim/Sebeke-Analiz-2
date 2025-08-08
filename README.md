# Şebeke Analizi ve Kompanzasyon Çözümleri (Streamlit)

Bu uygulama IEC 61000-4-30 Class A ölçüm yaklaşımına uygun şekilde şebeke analizi yapar,
harmonik/THD değerlendirmesi ve kompanzasyon önerileri üretir. Streamlit tabanlıdır.

## İçerik
- `12.py` – Uygulamanın ana dosyası
- `requirements.txt` – Gerekli Python paketleri (Streamlit Cloud uyumlu sürümler)
- `README.md` – Bu dosya

## Lokal Çalıştırma
```bash
pip install -r requirements.txt
streamlit run 12.py
```

## Streamlit Cloud Üzerinde Yayınlama
1. Bu klasörü GitHub'da yeni bir repoya **kök dizin** olarak yükleyin (ör. `sebeke-analiz`).
2. https://share.streamlit.io → **New app** → repo ve branch’i seçin.
3. **Main file path** alanına `12.py` yazın.
4. **Deploy**’a tıklayın. İlk build 1–3 dk sürebilir.
5. Gerekirse **Manage app → Settings → Advanced → Clear cache and restart** ile yeniden build alın.

## Notlar
- PDF çıktılarında Windows fontu bulunamazsa otomatik olarak DejaVu/Helvetica kullanılır.
- Matplotlib/ReportLab gibi paketler için sürümler sabitlenmiştir; Cloud’da sorunsuz kurulum içindir.
