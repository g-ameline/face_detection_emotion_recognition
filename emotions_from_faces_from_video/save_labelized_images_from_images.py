import constant
import utils
import model

# print(utils.file_paths_from_folder_path(folder_path=constant.exctrated_faces_folder_path, files_limit=30))

def save_iamge_under_emotion_label_folder(face_images_folder_path=constant.exctrated_faces_folder_path):
    saved_labelized_face_file_path = []
    print(f"loading facial recognition model function from {constant.emotion_recognition_model_file_path = }")
    emotion_and_likeliness_from_face = (
        model.create_emotion_and_likeliness_from_face_function_from_model_file_path(
            constant.emotion_recognition_model_file_path,
        )
    )
    counter = 0
    print(f"reaching face images in the folder: {face_images_folder_path = }")
    # print(f"{utils.file_paths_from_folder_path(folder_path=face_images_folder_path, files_limit=30)}")
    for image_file_path in utils.file_paths_from_folder_path(folder_path=face_images_folder_path, files_limit=30):
        assert image_file_path[-4:]=='.png', f"should only be .png here: {face_images_folder_path} and we got {image_file_path = }"
        print(f"\ntreating the following image: {image_file_path = }")
        face_image_as_matrix = utils.image_as_matrix_from_path(image_file_path)
        emotion, likeliness = emotion_and_likeliness_from_face(face_image_as_matrix)
        print(f"{emotion = } and {likeliness = }")
        counter+=1
        labelized_picture_name= f"{counter}_{emotion}_{int(likeliness*100)}.png"
        # labelized_image_file_path= f"{constant.labelized_faces_folder_path}{labelized_picture_name}"
        labelized_image_file_path = utils.save_image_to_png(
            image_matrix=face_image_as_matrix, 
            destination_folder_path=f"{constant.labelized_faces_folder_path}{emotion}/",
            image_file_name=labelized_picture_name,
        )
        saved_labelized_face_file_path.append(labelized_image_file_path) 
        print(f"labelized image saved at: {labelized_image_file_path}")
    # utilsimage_as_matrix_from_path(image_file_path)
    return saved_labelized_face_file_path
save_iamge_under_emotion_label_folder()
