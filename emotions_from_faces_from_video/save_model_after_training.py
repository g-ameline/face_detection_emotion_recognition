import constant
import model
import fer
import utils
import node

print(f"get train data set from {constant.train_data_csv_file_path}")

train_data_frame = fer.get_data_from_path(
    data_path=constant.train_data_csv_file_path,
    # data_path='../data_csv/train/data.csv',
    input_image_format='numerals',
    output_image_format='matrix',
    sampling_quantity=None,
)
train_inputs_as_array, train_labels_as_one_hot_encoded = model.keras_inputs_and_labels_from_data_frame( train_data_frame )

print(f"get validate data set from {constant.validate_data_csv_file_path}")

validate_data_frame = fer.get_data_from_path(
    data_path=constant.validate_data_csv_file_path,
    input_image_format='numerals',
    output_image_format='matrix',
    sampling_quantity=None,
)
validate_inputs_as_array, validate_labels_as_one_hot_encoded = model.keras_inputs_and_labels_from_data_frame( validate_data_frame )

print(f"get test data set from {constant.test_data_csv_file_path}")

test_data_frame = fer.get_data_from_path(
    data_path=constant.test_data_csv_file_path,
    input_image_format='numerals',
    output_image_format='matrix',
    sampling_quantity=None,
)
test_inputs_as_array, test_labels_as_one_hot_encoded = model.keras_inputs_and_labels_from_data_frame( test_data_frame )

print(f"get architecture  from {constant.architecture_file_path}")

architecture_as_datas = utils.object_from_pickled_file_path(constant.architecture_file_path)
architecture_as_data_graph = node.path_data_graph_from_datas( architecture_as_datas )

print(f"start making model and training it")

model.saved_trained_model_file_path_from_data_graph(
    data_graph= architecture_as_data_graph,
    train_inputs_as_array=train_inputs_as_array,
    train_labels_as_one_hot_encoded=train_labels_as_one_hot_encoded,
    validate_inputs_as_array=validate_inputs_as_array,
    validate_labels_as_one_hot_encoded=validate_labels_as_one_hot_encoded,
    test_inputs_as_array=test_inputs_as_array,
    test_labels_as_one_hot_encoded=test_labels_as_one_hot_encoded,
    destination_folder_path=constant.model_folder_path,
    file_name=constant.emotion_recognition_model_file_name,
    epochs=15,
)

print(f"model should be saved at: {constant.model_folder_path}{constant.emotion_recognition_model_file_name}")
