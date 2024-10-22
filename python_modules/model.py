import keras
import pandas
import numpy
import constant
import sys
import node
import time 

# helper functions to translate data_graph into keras models

def model_from_data_graph(data_graph):
    input_data_node, output_data_node = data_graph
    keras_model = keras.Model(inputs=input_data_node.inducted_layer, outputs=output_data_node.inducted_layer)
    return keras_model
    
def keras_inputs_and_labels_from_data_frame(data_frame, input_column_name='image',label_column_name='emotion'):
    get_all_inputs_as_array_from_matrix_series = lambda matrix_series: numpy.stack([ matrix for matrix in matrix_series ])
    inputs_as_array = get_all_inputs_as_array_from_matrix_series(data_frame[input_column_name])
    labels_as_one_hot_encoded = keras.utils.to_categorical(data_frame[label_column_name].to_numpy())
    return inputs_as_array, labels_as_one_hot_encoded

def path_data_graph_from_data_nodes(datas):
    first_data_node = node.DataNode(data=datas[0] ,parents=None)
    last_data_node = first_data_node
    for data in datas[1:-1]:
        last_data_node = node.DataNode(data=data).add_parent(last_data_node)
    last_data_node = node.DataNode(data=datas[-1], children=None).add_parent(last_data_node)
    return first_data_node, last_data_node

def scores_by_metric_from_path_data_graph(
    last_data_node,
    input_shape,
    first_data_node,
    train_inputs_as_array,
    train_labels_as_one_hot_encoded,
    validate_inputs_as_array,
    validate_labels_as_one_hot_encoded,
    # test_inputs_as_array=None,
    # test_labels_as_one_hot_encoded=None,
    epochs,
    verbose=1,
    batch_size=16,
    callbacks=[],
):
    assert input_shape == train_inputs_as_array.shape[1:] == validate_inputs_as_array.shape[1:], f"{input_shape = } {train_inputs_as_array.shape[1:] = } {validate_inputs_as_array.shape[1:] = }"
    keras_model = node.path_keras_model_from_last_data_node(
        last_data_node=last_data_node,
        input_shape=input_shape,
        first_data_node=first_data_node,
    )
    keras_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'],
    )
    keras_model.summary()
    assert (batch_size & (batch_size-1) == 0) and batch_size != 0 ,f"internet says that batch size: {batch_size} should be a power of two"
    assert train_inputs_as_array.shape == ( *train_labels_as_one_hot_encoded.shape[:1] , *input_shape ), f"shape missmatch"
    history = keras_model.fit(
        x=train_inputs_as_array,
        y=train_labels_as_one_hot_encoded,
        validation_data=(validate_inputs_as_array,validate_labels_as_one_hot_encoded),
        epochs=epochs,
        batch_size=batch_size,
        validation_batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    score_by_metric = { metric: values[-1] for metric, values in history.history.items() }
    # if test_inputs_as_array and test_labels_as_one_hot_encoded:
    #     metric, acc = model.evaluate(test_inputs_as_array, test_labels_as_one_hot_encoded, batch_size=batch_size)
    #     print(f"output of evaluate method: {metric = }, {acc =} ")
    return score_by_metric

def image_and_label_generator_from_data_frame(
    data_frame, 
    image_file_path_column_name, 
    label_column_name,
    color_mode,
    class_mode,
    rescale,
):
    train_datagen = keras.src.legacy.preprocessing.image.ImageDataGenerator(
        rescale=rescale,
        vertical_flip=True,
        rotation_range=3,
        # theta=3,
    )
    input_generator = train_datagen.flow_from_dataframe(
        dataframe=data_frame,
        x_col=image_file_path_column_name,
        y_col=label_column_name,
        target_size=(48, 48),
        color_mode=color_mode,
        class_mode=class_mode
    )
    return input_generator

def fixing_fer_parameters_of_image_and_label_generator_from_data_frame_from_fer_constants(
    image_file_path_column_name='image',
    label_column_name='emotion',
    target_size=(48, 48),
    color_mode='grayscale',
    rescale=1./255,
):
    def fixed_fer_parameters_image_and_label_generator_from_data_frame(data_frame):
        class_mode = 'categorical' if data_frame[label_column_name].dtype == pandas.StringDtype() else 'raw'
         # 'raw' if label col is numerical dtype
            # class_mode='raw',
         # 'categorical' if label col is string dtype
            # class_mode='categorical',
        return image_and_label_generator_from_data_frame(
            data_frame=data_frame,
            image_file_path_column_name=image_file_path_column_name,
            label_column_name=label_column_name,
            color_mode=color_mode,
            class_mode=class_mode,
            rescale=rescale,
        )
    return fixed_fer_parameters_image_and_label_generator_from_data_frame

fer_image_and_label_generator_from_data_frame = (
    fixing_fer_parameters_of_image_and_label_generator_from_data_frame_from_fer_constants()
)

def load_keras_model_from_file_path(model_file_path):
    return keras.saving.load_model(model_file_path, custom_objects=None, compile=True, safe_mode=True)

def create_emotion_likelinesses_from_face_function_from_model(keras_model):
    def emotion_likelinesses_from_face(face_image):
        emotion_likelinesses = keras_model.predict(
            x=numpy.array( [face_image,] ),
            verbose=0,
            batch_size=1,
        )[0]
        return emotion_likelinesses 
    return emotion_likelinesses_from_face

def emotion_from_emotion_likelinesses(emotion_likelinesses, emotion_format='name'):
    emotion_as_number = numpy.argmax(emotion_likelinesses)
    if emotion_format == 'number':
        return emotion_as_number
    if emotion_format == 'name':
        return constant.emotion_names[emotion_as_number]
    raise Excpetion(f"probably a wrong {emotion_format = }")

def emotion_and_likeliness_from_emotion_likelinesses(emotion_likelinesses, emotion_format='name'):
    emotion_as_number = numpy.argmax(emotion_likelinesses)
    emotion_likeliness = int(100*emotion_likelinesses[emotion_as_number])/100
    if emotion_format == 'number':
        emotion = emotion_as_number
    if emotion_format == 'name':
        emotion = constant.emotion_names[emotion_as_number]
    return emotion, emotion_likeliness

def create_emotion_and_likeliness_from_face_function_from_model_file_path(model_file_path, emotion_format='name'):
    keras_model = load_keras_model_from_file_path(model_file_path)
    emotion_likelinesses_from_face = create_emotion_likelinesses_from_face_function_from_model(keras_model)
    def emotion_and_likeliness_from_face(face):
        emotion_likelinesses = emotion_likelinesses_from_face(face)
        emotion_and_likeliness = emotion_and_likeliness_from_emotion_likelinesses(emotion_likelinesses)
        return emotion_and_likeliness 
    return emotion_and_likeliness_from_face

def compiled_model_from_datas_graph(
    data_graph,
    input_shape=(48,48),
):
    last_data_node = data_graph[-1]
    first_data_node = data_graph[0]
    keras_model = node.path_keras_model_from_last_data_node(
        last_data_node=last_data_node,
        input_shape=input_shape,
        first_data_node=first_data_node,
    )
    keras_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'],
    )
    keras_model.summary()
    return keras_model


def trained_model_from_compiled_model(
    compiled_model,
    train_inputs_as_array,
    train_labels_as_one_hot_encoded,
    validate_inputs_as_array,
    validate_labels_as_one_hot_encoded,
    epochs,
    verbose=1,
    batch_size=32,
    callbacks=[],
):
    history = compiled_model.fit(
        x=train_inputs_as_array,
        y=train_labels_as_one_hot_encoded,
        validation_data=(validate_inputs_as_array,validate_labels_as_one_hot_encoded),
        epochs=epochs,
        batch_size=batch_size,
        validation_batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return compiled_model


def test_accuracy_from_model_and_test_set(
    trained_model, 
    test_inputs_as_array,
    test_labels_as_one_hot_encoded,
    batch_size=32,
):
    test_loss, test_accuracy = trained_model.evaluate(test_inputs_as_array, test_labels_as_one_hot_encoded, batch_size=batch_size)
    print(f"output of evaluate method: {test_loss = }, {test_accuracy =} ")
    return test_accuracy


def print_emotions_from_images(
    images,
    faces_from_image,
    emotion_from_face,
):
    previous_emotions = None
    for image in images:
        faces = faces_from_image(image)
        new_emotions = [emotion_from_face(face) for face in faces]
        if new_emotions != previous_emotions:
            sys.stdout.write(f"\rdetected emotions: {new_emotions}                        ")
            sys.stdout.flush()
            previous_emotions= new_emotions

            
def saved_trained_model_file_path_from_data_graph(
    data_graph,
    train_inputs_as_array,
    train_labels_as_one_hot_encoded,
    validate_inputs_as_array,
    validate_labels_as_one_hot_encoded,
    test_inputs_as_array,
    test_labels_as_one_hot_encoded,
    destination_folder_path=constant.model_folder_path,
    file_name=constant.emotion_recognition_model_file_name,
    epochs=70, # 15 for testing for example
    timeout=120, # minutes, less for simple testing
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=False, monitor='accuracy', min_delta=0.005, verbose=1),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=False, monitor='val_accuracy', min_delta=0.005, verbose=1),
    ],
):
    if timeout > 0:
        class TimeOut(keras.callbacks.Callback):
            def __init__(self, t0, timeout):
                super().__init__()
                self.t0 = t0
                self.timeout = timeout  # time in minutes
            def on_train_batch_end(self, batch, logs=None):
                if time.time() - self.t0 > self.timeout * 60:  # 58 minutes
                    print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
                    self.model.stop_training = True
        callabcks = callbacks.append(TimeOut(t0=time.time(), timeout=timeout))

    compiled_model = compiled_model_from_datas_graph( data_graph, input_shape=(48,48), )
    trained_model = trained_model_from_compiled_model(
        compiled_model=compiled_model,
        train_inputs_as_array=train_inputs_as_array,
        train_labels_as_one_hot_encoded=train_labels_as_one_hot_encoded,
        validate_inputs_as_array=validate_inputs_as_array,
        validate_labels_as_one_hot_encoded=validate_labels_as_one_hot_encoded,
        epochs=epochs,
        verbose=1,
        batch_size=32,
        callbacks=callbacks,
    )
    print(
        test_accuracy_from_model_and_test_set(
            trained_model,
            test_inputs_as_array,
            test_labels_as_one_hot_encoded,
        )
    )
    keras.saving.save_model(
        model=trained_model, 
        filepath=constant.emotion_recognition_model_file_path, 
        overwrite=True, 
        zipped=None,
    )
    return constant.emotion_recognition_model_file_path
