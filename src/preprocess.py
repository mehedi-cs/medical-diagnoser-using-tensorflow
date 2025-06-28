import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(url, num_words=5000):
    data = pd.read_csv(url)
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['Patient_Problem'])
    sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])
    max_length = max(len(x) for x in sequences)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')

    disease_encoder = LabelEncoder()
    prescription_encoder = LabelEncoder()
    disease_labels = disease_encoder.fit_transform(data['Disease'])
    prescription_labels = prescription_encoder.fit_transform(data['Prescription'])

    disease_cat = to_categorical(disease_labels)
    prescription_cat = to_categorical(prescription_labels)

    y = np.hstack((disease_cat, prescription_cat))
    
    return padded, disease_cat, prescription_cat, tokenizer, max_length, disease_encoder, prescription_encoder
