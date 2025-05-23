Duygu Temelli Müzik Öneri Sistemi

Bu derin öğrenme projesi, kullanıcının **anlık ses kaydından** duygu durumunu tahmin ederek, tahmin edilen duyguya uygun bir **müzik listesi** önermektedir. Proje, veri üretimi, model eğitimi ve web arayüzü ile tam entegre bir yapıda geliştirilmiştir.


**Özellikler**

 🎤 Anlık ses kaydı alarak duygu tahmini yapar.
 🤖 Derin öğrenme modeli ile 3 temel duygudan birini sınıflandırır (örneğin: mutlu, üzgün, stresli).
 🎵 Tahmin edilen duyguya uygun bir müzik listesi önerisinde bulunur.
 🌐 Streamlit ile kullanıcı dostu **web arayüzü** sunar.

**Kullanılan Yöntemler ve Teknolojiler**

- Python 
- Jupyter Notebook: Veri oluşturma ve model eğitimi
- TensorFlow / Keras: Derin öğrenme modeli
- Librosa: Ses işleme ve öznitelik çıkarımı
- Streamlit: Web tabanlı kullanıcı arayüzü
- Pickle / .keras: Model kaydetme ve yükleme


#Kullanılan Model Yapısı
Bu projede kullanılan model, **derin ve çift yönlü (bidirectional) LSTM** tabanlı çok sınıflı duygu sınıflandırma modelidir.
