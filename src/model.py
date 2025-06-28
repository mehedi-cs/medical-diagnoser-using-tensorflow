from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def build_model(max_length, num_words, disease_classes, prescription_classes):
    input_layer = Input(shape=(max_length,))
    embedding = Embedding(input_dim=num_words, output_dim=64)(input_layer)
    lstm = LSTM(64)(embedding)

    disease_output = Dense(disease_classes, activation='softmax', name='disease_output')(lstm)
    prescription_output = Dense(prescription_classes, activation='softmax', name='prescription_output')(lstm)

    model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])
    model.compile(
        loss={'disease_output': 'categorical_crossentropy', 'prescription_output': 'categorical_crossentropy'},
        optimizer='adam',
        metrics={'disease_output': 'accuracy', 'prescription_output': 'accuracy'}
    )
    return model
