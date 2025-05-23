import streamlit as st
import tensorflow as tf
import numpy as np
import re
import pickle
import speech_recognition as sr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from st_audiorec import st_audiorec
from pydub import AudioSegment
import tempfile
import os

# Model ve tokenizer yükle
model = tf.keras.models.load_model("model.keras")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Sabitler
MAX_LEN = 30
labels = {0: "Mutlu 😊", 1: "Üzgün 😔", 2: "Stresli 😵"}

# Playlist bağlantıları (senin verdiğin linkler)
playlists = {
    0: "https://music.youtube.com/playlist?list=PLzR4WEbCMbKl9c54eUXFRKMQgy-7AmFp5",  # Mutlu
    1: "https://music.youtube.com/playlist?list=PLzR4WEbCMbKkPS-Y4IaCKS77oRU2T2Mdm",  # Üzgün
    2: "https://music.youtube.com/playlist?list=PLzR4WEbCMbKkrTmpz2Ff6Pm9PXEbWawds"   # Stresli
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü0-9 ]', '', text)
    return text

st.title("🎤 MoodTune - Anlık Ses Kaydı ile Duygu Tahmini")

# Ses kaydını al
st.write("🎙️ Kaydetmek için aşağıdaki butona tıklayın:")
audio_bytes = st_audiorec()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.info("Ses kaydı alındı, metne dönüştürülüyor...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="tr-TR")

            st.subheader("🗣️ Çözümlenen Metin:")
            st.write(text)

            # Tahmin yap
            cleaned = clean_text(text)
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
            prediction = model.predict(padded)
            predicted_class = np.argmax(prediction)

            st.success(f"💡 Tahmin: {labels[predicted_class]}")

            # Playlist Gösterme Butonu
            if st.button("🎵 Playlist Getir"):
                playlist_url = playlists[predicted_class]
                st.markdown(f"[🎧 Bu duyguya uygun şarkı listesine gitmek için buraya tıkla]({playlist_url})")

    except Exception as e:
        st.error(f"❌ Ses çözümlenemedi: {e}")
    finally:
        os.unlink(temp_audio_path)
