data_url = 'https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip'
data_folder_path = '../data/'
model_folder_path= '../model/'

emotion_names=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

downloaded_data_file_name = 'downloaded_data.zip'
all_data_file_name='icml_face_data.csv'
unzipped_data_folder_name = 'unzipped_data/'
legacy_data_set_names = ['Training', 'PublicTest', 'PrivateTest']
data_set_names = ['train', 'validate', 'test']

data_csv_file_name = 'data.csv'

train_data_csv_file_path = f"{data_folder_path}{data_set_names[0]}/{data_csv_file_name}"
validate_data_csv_file_path = f"{data_folder_path}{data_set_names[1]}/{data_csv_file_name}"
test_data_csv_file_path = f"{data_folder_path}{data_set_names[2]}/{data_csv_file_name}"

architecture_folder_path= '../architecture/'
architecture_file_name = 'architecture.pickle'
architecture_file_path = architecture_folder_path + architecture_file_name

emotion_recognition_model_file_name = 'emotion_recognition.keras'
emotion_recognition_model_file_path = model_folder_path + emotion_recognition_model_file_name 

yunet_model_url = 'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
yunet_model_file_name = 'yunet.onnx'
yunet_model_file_path = model_folder_path + yunet_model_file_name 

emotion_from_face_function_file_name = 'emotion_from_face'
emotion_from_face_function_file_path = emotion_from_face_function_file_name + emotion_from_face_function_file_name

face_image_shape = (48,48)

sample_data_folder_path = '../sample/'
exctrated_faces_folder_name = 'twenty_seconds_of_face_extraction/'
exctrated_faces_folder_path = sample_data_folder_path + exctrated_faces_folder_name 
video_of_human_faces_file_url = 'https://www.youtube.com/watch?v=MTWkfpa-jJw'
video_of_human_faces_file_name = 'facial_emotion_video_from_youtube.mp4'
video_of_human_faces_file_path= sample_data_folder_path + video_of_human_faces_file_name 
labelized_faces_folder_name = 'labelized_face_images/'
labelized_faces_folder_path = sample_data_folder_path + labelized_faces_folder_name 

