import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from kinect_learning import *
import time

DATA_DIR = 'data'


# Training function
def train_model(data_file_name, epochs, noise, num_classes=5):
    # 1. Load Data
    file_path = os.path.join(DATA_DIR, data_file_name)
    data_collection = joints_collection(data_file_name.rstrip('.csv'))
    data = load_data_multiple_dimension(file_path, data_collection, noise)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = create_datasets(data['positions'], data['labels'])

    # Input layers
    query_input = keras.Input(shape=train_x[0].shape, dtype='float64')
    value_input = keras.Input(shape=train_x[0].shape, dtype='float64')

    # Convolution layers
    convolution_layer = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same')
    convolved_query = convolution_layer(query_input)
    convolved_value = convolution_layer(value_input)

    # Batch normalization
    convolved_query = keras.layers.BatchNormalization()(convolved_query)
    convolved_value = keras.layers.BatchNormalization()(convolved_value)

    # Second convolution to extract more features
    convolved_query = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same', activation='relu')(convolved_query)
    convolved_value = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same', activation='relu')(convolved_value)

    # Attention layer
    attention_output = keras.layers.Attention()([convolved_query, convolved_value])

    # Flatten and Dense layers
    flatten_layer = keras.layers.Flatten()(attention_output)

    # Dropout to avoid overfitting
    dropout_layer_1 = keras.layers.Dropout(0.5)(flatten_layer)

    # Dense layer with L2 regularization and 'relu' activation
    dense_layer_1 = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(
        dropout_layer_1)

    # Dropout to avoid overfitting
    dropout_layer_2 = keras.layers.Dropout(0.3)(dense_layer_1)

    # Final Dense layer with softmax activation for classification
    dense_layer_2 = keras.layers.Dense(num_classes, activation='softmax')(dropout_layer_2)

    # Build the model
    model = keras.Model(inputs=[query_input, value_input], outputs=[dense_layer_2])

    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Callbacks
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        (train_x, train_x), train_y,
        validation_data=((val_x, val_x), val_y),
        epochs=epochs,
        batch_size=64,
        callbacks=[lr_schedule, early_stopping]
    )

    # Save the trained model to disk for later inference
    model.save(f'models/model_{data_file_name.rstrip(".csv")}.h5')

    return model, history, test_x, test_y


# Create datasets with train, validation, and test sets
def create_datasets(x, y, valid_size=0.2, test_size=0.2):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # Shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = np.take(x, indices, axis=0)
    y = np.take(y, indices)

    # Split into train, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=valid_size + test_size)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=test_size / (valid_size + test_size))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


# Function to convert model to TensorFlow Lite with quantization
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_model_path = f'models/quantized_model_.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model converted to TFLite format with quantization.")
    return tflite_model_path


# Inference function for TFLite model
def run_inference_tflite(tflite_model_path, test_x):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess test data
    test_x = np.asarray(test_x, dtype=np.float32)  # Adjust type if necessary

    # Time the inference step
    start_time = time.time()

    predicted_classes = []

    # Run inference for all test samples
    for i in range(test_x.shape[0]):
        # Set input tensor for each test sample
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(test_x[i], axis=0))  # Add batch dimension
        interpreter.invoke()

        # Get the output for this sample
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Convert probability to class (for binary, use threshold 0.5; for multi-class, use argmax)
        predicted_class = np.argmax(output_data, axis=-1)
        predicted_classes.append(predicted_class)

    # Measure time taken for inference
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference completed in {inference_time:.4f} seconds")
    print(f"Predicted classes: {len(predicted_classes)}")
    print("---------------------------------------------------------------------")
    return predicted_classes, inference_time


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--noise', default=True, action=argparse.BooleanOptionalAction)
    FILE_NAMES = list(map(os.path.basename, glob.glob('./data/*.csv')))
    print('Training on files: {}'.format(FILE_NAMES))
    ARGS = PARSER.parse_args()
    print(f'Noise: {ARGS.noise}')

    TRAINING_ATTEMPTS = 5
    EPOCHS = 100
    RESULT = {
        file_name: [train_model(file_name, epochs=EPOCHS, noise=ARGS.noise) for _ in range(TRAINING_ATTEMPTS)]
        for file_name in FILE_NAMES
    }
    BEST_RESULTS = {
        file_name: max(trained_models, key=lambda model: model[1].history['val_accuracy'])
        for file_name, trained_models in RESULT.items()
    }

    for DATA_FILE, BEST_RESULT in BEST_RESULTS.items():
        model, history, test_x, test_y = BEST_RESULT
        print(DATA_FILE, history.history['val_accuracy'][-1])

        # Convert model to TFLite with quantization
        tflite_model_path = convert_to_tflite(model)
        #print(tflite_model_path)

        # Run inference on the actual test set
        predicted_classes, inference_time = run_inference_tflite(tflite_model_path, test_x)
        print(inference_time)
