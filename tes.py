from models.preprocessing_model import TextPreprocessor
import joblib

# model = TextPreprocessor()
# joblib.dump(model, 'model.joblib')

model = joblib.load('model.joblib')
kalimat = "Petani padi mengikuti sosialisasi tahap VII yang diadakan oleh mahasiswa jurusan pertanian."
hasil = model.preprocess(kalimat)
print(hasil)