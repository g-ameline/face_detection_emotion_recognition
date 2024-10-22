import cv2
import utils
import constant
import video

yunet_model_file_path = utils.fetch_file_stream(
    url=constant.yunet_model_url, 
    destination_folder_path=constant.model_folder_path, 
    data_file_name=constant.yunet_model_file_name,
    redownload=False,
)
utils.show_folders_and_files(constant.model_folder_path)

def yunet_model_with_proper_input_shape_from_yunet_model_file_path(yunet_model_file_path, video_path=None):
    yunet_model = cv2.FaceDetectorYN_create(
        yunet_model_file_path, 
        "", # no idea what it is 
        (0,0), # same
    )
    if video_path is None:
        video_path = video.find_webcam_index()
    assert isinstance(video_path, str) or isinstance(video_path, int), f"video_path must be path to file or device or index: {video_path = }"
    yunet_model.setInputSize(video.find_video_shape(video_path)) 
    # will try any available device and return frame shape of the first one that ca be opened
    return yunet_model 

def create_functions_from_model(yunet_model, video_path, face_image_shape=constant.face_image_shape):    
    video_path = video.find_webcam_index() if video_path is None else video_path
    assert isinstance(video_path, str) or isinstance(video_path, int), f"video_path must be path to file or device or index: {video_path = }"
    assert len(face_image_shape)==2
    def faces_and_rectangles_from_image(image_array):
        _truc, faces = yunet_model.detect(image_array)
        extracted_faces = []
        rectangles = []
        if faces is None:
            pass
        if faces is not None:
            for face in faces:
                x, y, w, h = list( int(v) for v in face[:4] )
                face_rectangle = image_array[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_rectangle, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, face_image_shape)
                extracted_faces.append(face_resized)
                rectangles.append(top_left_and_bot_right:=((x, y), (x+w, y+h)))
        return extracted_faces, rectangles
        
    def faces_from_image(image_array):
        _truc, faces = yunet_model.detect(image_array)
        extracted_faces = []
        if faces is None:
            pass
        if faces is not None:
            for face in faces:
                x, y, w, h = list( int(v) for v in face[:4] )
                face_rectangle = image_array[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_rectangle, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, face_image_shape)
                assert face_resized.shape == face_image_shape, f"{face_resized.shape = } not compatible with {face_image_shape}"
                extracted_faces.append(face_resized)
        return extracted_faces 
        
    def face_rectangles_from_image(image_array):
        _truc, faces = yunet_model.detect(image_array)
        rectangles = []
        if faces is None:
            pass
        if faces is not None:
            for face in faces:
                x, y, w, h = list( int(v) for v in face[:4] )
                face_rectangle = image_array[y:y+h, x:x+w]
                rectangles.append(top_left_and_bot_right:=((x, y), (x+w, y+h)))
        return rectangles

    return faces_and_rectangles_from_image, faces_from_image, face_rectangles_from_image

def create_functions_from_yunet_model_file_path(yunet_model_file_path, video_path=None):
    yunet_model = yunet_model_with_proper_input_shape_from_yunet_model_file_path(yunet_model_file_path)
    return create_functions_from_model(yunet_model, video_path)

def create_faces_and_rectangles_from_image_fnction_from_yunet_model_file_path(yunet_model_file_path, video_path=None):
    yunet_model = yunet_model_with_proper_input_shape_from_yunet_model_file_path(yunet_model_file_path, video_path)
    return create_functions_from_model(yunet_model, video_path)[0]

def create_faces_from_image_function_from_yunet_model_file_path(yunet_model_file_path, video_path=None):
    yunet_model = yunet_model_with_proper_input_shape_from_yunet_model_file_path(yunet_model_file_path, video_path)
    return create_functions_from_model(yunet_model, video_path)[1]

def create_face_rectangles_from_image_function_from_yunet_model_file_path(yunet_model_file_path, video_path=None):
    yunet_model = yunet_model_with_proper_input_shape_from_yunet_model_file_path(yunet_model_file_path, video_path)
    return create_functions_from_model(yunet_model, video_path)[2]
