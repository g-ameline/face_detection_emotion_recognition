import yunet
import constant
import video
import utils

def saved_faces_file_path_from_video(video_path, yunet_model_file_path, fps=30):
    print(f"extractng faces from video at: {video_path = }")
    saved_face_frame_file_paths = []
    images = video.images_from_video(video_path)

    faces_from_image = yunet.create_faces_from_image_function_from_yunet_model_file_path(
        yunet_model_file_path=yunet_model_file_path, 
        video_path=video_path,
    )
    print(faces_from_image)
    frame_start = 0
    frame_end = frame_start+20*fps # seconds
    frame_last_save = frame_start
    id = 0
    frame = 0
    for image in images:
        if frame > frame_end:
            break
        frame += 1
        if frame - frame_last_save > fps:
            try:
                faces = faces_from_image(image)
            except Exception as error:
                print(f"{error = }")
            if len(faces) < 1:
                continue
            for face in faces:
                print(f"saving_face image at to {constant.sample_data_folder_path+constant.exctrated_faces_folder_name}{f"extracted_face_{str(id)}.png"}")
                id += 1
                saved_frame_path = utils.save_image_to_png(
                    face, 
                    constant.sample_data_folder_path+constant.exctrated_faces_folder_name,
                    f"extracted_face_{str(id)}.png",
                )
                saved_face_frame_file_paths.append(saved_frame_path)
            frame_last_save = frame 
    return saved_face_frame_file_paths
            
saved_faces_file_path_from_video(
    video_path=constant.video_of_human_faces_file_path, 
    yunet_model_file_path=constant.yunet_model_file_path,
)
print(f"end of faces exctraction")
