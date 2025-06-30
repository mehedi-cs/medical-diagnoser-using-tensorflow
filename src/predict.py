import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def make_prediction(input_problem, model, tokenizer, max_length, disease_encoder, prescription_encoder):
    sequences = tokenizer.texts_to_sequences([input_problem])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded)

    disease = disease_encoder.inverse_transform([np.argmax(predictions[0])])[0]
    prescription = prescription_encoder.inverse_transform([np.argmax(predictions[1])])[0]
    return disease, prescription
