import streamlit as st
import joblib

# Sayfa ayarları
st.set_page_config(
    page_title="Türkçe Metin Sınıflandırıcı",
    page_icon="📦",
    layout="centered"
)

# Başlık
st.title("📦 Türkçe Metin Sınıflandırıcı")
st.markdown("""
Bu uygulama, kısa Türkçe cümleleri üç sınıftan birine otomatik olarak sınıflandırır:

- 🧠 **Kişisel Gelişim**  
- 🏡 **Günlük Yaşam**  
- 📢 **Talep / Şikayet**

Aşağıya bir cümle yazın ve hangi kategoriye ait olduğunu görün.
""")

# Model ve LabelEncoder'ı yükleyelim
@st.cache_resource
def load_model():
    model = joblib.load("tfidf_tr_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# Metin girişi
user_input = st.text_area("✍️ Cümlenizi buraya yazın:", height=150)

# Tahmin butonu
if st.button("🔍 Sınıfı Tahmin Et"):
    if not user_input.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        prediction = model.predict([user_input])
        predicted_label = encoder.inverse_transform(prediction)[0]

        # Sınıfa göre emoji
        emoji_map = {
            "kişisel gelişim": "🧠",
            "günlük yaşam": "🏡",
            "talep/şikayet": "📢"
        }
        emoji = emoji_map.get(predicted_label, "🔎")

        st.success(f"{emoji} Tahmin Edilen Sınıf: **{predicted_label}**")
