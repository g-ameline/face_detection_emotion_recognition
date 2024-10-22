import os
import yt_dlp
import constant

from yt_dlp import YoutubeDL

def downloaded_video_file_path_from_video_url(video_url, destination_folder_path, video_file_name):
    print(f"dowloading video from {video_url}")
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    destination_filepath = destination_folder_path+video_file_name
    ydl_opts = {
        'format': 'worstvideo/worst',  
        'outtmpl': destination_filepath,
        'no_warnings': True,           
        'quiet': True                  
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return destination_filepath

constant.video_of_human_faces_file_url

video_file_path = downloaded_video_file_path_from_video_url(
    video_url=constant.video_of_human_faces_file_url, 
    destination_folder_path=constant.sample_data_folder_path,
    video_file_name=constant.video_of_human_faces_file_name,
)
print(f"video idownloaded at: {video_file_path}")
