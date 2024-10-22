import cv2
import time

def find_webcam_index():
    for device_index in range(0,200):
        video_capture = cv2.VideoCapture(device_index)
        if video_capture.isOpened():
            ok, frame = video_capture.read()
            if not ok:
                continue
            return device_index
        if not video_capture.isOpened():
            video_capture.release()
            continue
        # except:
        #     print(f"did not work {device_index = }")
        #     video_capture.release()
    raise Exception("could not find a working webcam device, try : `!cameractrls -L` if you are on systemd linux distro")

def find_webcam_index_and_shape():
    for device_index in range(200):
        try:
            video_capture = cv2.VideoCapture(device_index)
            if video_capture.isOpened():
                device_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_capture.release()
                return device_index, device_shape
            video_capture.release()
        except:
            video_capture.release()
    raise Exception("could not find a working webcam device, try : `!cameractrls -L` if you are on systemd linux distro")

def find_video_shape(video_index_or_path=None):
    video_index_or_path= find_webcam_index() if video_index_or_path is None else video_index_or_path
    video_capture = cv2.VideoCapture(video_index_or_path)
    if video_capture.isOpened():
        device_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()
        return device_shape
    raise Exception(f"failed to open the device {device_index =}")
         
def images_from_video(video_index_or_path=None, duration=10):
    video_index_or_path= find_webcam_index() if video_index_or_path is None else video_index_or_path
    print(f"{video_index_or_path = }")
    video_capture = cv2.VideoCapture(video_index_or_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open camera device with index {video_index_or_path}")
    start_time = time.time()
    end_time = start_time + duration
    while time.time() < end_time:
        ok, frame = video_capture.read()
        if not ok:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        yield frame
    video_capture.release()
