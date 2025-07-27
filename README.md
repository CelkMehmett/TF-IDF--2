# 🇹🇷 Turkish Text Classifier with TF‑IDF and Scikit‑learn

Bu proje, Türkçe kısa cümleleri üç sınıfa ayırmak üzere geliştirilmiş, **TF‑IDF** temelli bir metin sınıflandırma sistemidir.  
Kapsamlı bir **doğal dil işleme (NLP)** süreci içerir ve **Naive Bayes**, **Logistic Regression**, **Karar Ağacı**, **GridSearchCV** gibi güçlü modeller ve araçlar içerir.  
Ayrıca, kullanıcı dostu bir arayüz ile **Streamlit** üzerinden etkileşimli hale getirilmiştir.

---

## 🎯 Hedeflenen Sınıflar

- 🧠 **Kişisel Gelişim** (`kişisel gelişim`)  
- 🏡 **Günlük Yaşam** (`günlük yaşam`)  
- 📢 **Talep / Şikayet** (`talep/şikayet`)  

---

## 🚀 Özellikler

✅ Türkçe’ye özel stopword listesi  
✅ TF‑IDF + N‑gram desteği (unigram + bigram)  
✅ Naive Bayes, Lojistik Regresyon ve Karar Ağacı ile model eğitimi  
✅ GridSearchCV ile hiperparametre optimizasyonu  
✅ `.pkl` uzantılı model ve encoder dosyaları ile model kaydı  
✅ Streamlit üzerinden web tabanlı arayüz

---

## 🧠 Veri Seti

El yapımı kısa Türkçe cümlelerden oluşur ve 3 farklı kategoriye etiketlenmiştir.  
Veri seti doğrudan notebook içinde ya da `data.py` dosyasının içinde bulunmaktadır.

---

## 🛠️ Kurulum

```bash
git clone https://github.com/CelkMehmett/TF-IDF--2.git
cd TF-IDF--2
pip install -r requirements.txt
```

Streamlit arayüzünü başlatmak için:

```bash
streamlit run streamlit_app.py
```

---

## 💻 Uygulama Arayüzü

Aşağıdaki örnek arayüz ile bir cümleyi yazıp tahmini sınıfı görebilirsiniz.

<div align="center">
  <img src="demo.png" width="800"/>
</div>

---

## 🧠 Model Kullanımı (Python içinden)

```python
import joblib

# Model ve LabelEncoder yükle
model = joblib.load("tfidf_tr_model.pkl")
le = joblib.load("label_encoder.pkl")

# Tahmin yap
text = ["Bugün çok yoruldum ama hedeflerimi unutmadım."]
pred = model.predict(text)
print("Tahmin edilen sınıf:", le.inverse_transform(pred)[0])
```

---

## 📁 Dosya Yapısı

```
TF-IDF--2/
│
├── TF_IDF_ile_Metin_Sınıflandırıcı_Yazmak__.ipynb   # Notebook dosyası
├── streamlit_app.py                                  # Streamlit arayüzü
├── tfidf_tr_model.pkl                                # Eğitimli model
├── label_encoder.pkl                                 # Etiket çözücü
├── demo.png                                          # Arayüz ekran görüntüsü
├── README.md
└── requirements.txt
```

---

## 📌 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

## 🌐 Demo

Görüntüle → [GitHub Proje Sayfası](https://github.com/CelkMehmett/TF-IDF--2)

---

## 🤝 Katkıda Bulun

Katkı sağlamak isterseniz `fork` edip `pull request` gönderebilir veya hata bildirimi açabilirsiniz.
