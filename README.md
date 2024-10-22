# Image Detection and Recognition

## Content

The repository contains:
  - A small exercise on MNIST digit recognition.
  - A main task about recognizing human expressions in a video, split into different steps:
    - Extracting faces from an image 
      - Pretrained YuNet model.
    - Recognizing emotions on a facial image
      - Deep convolutional neural network trained on a dataset (Keras functional API).



* to be able to run it from your machine check <./how_to_run_it.md>

## Testing Scripts

Go to <./emotions_from_faces_from_video> and run:
  - `save_sample_video.py`
  - `save_faces_from_video.py`
  - `save_emotion_recognition_data.py`
  - `save_architecture.py`
  - `save_model_after_training.py`
  - `save_labeled_images_from_images.py`
  
## Testing Notebooks

The project is divided into different folders containing notebooks corresponding to different stages of the ML process:

0) Get the data
<./data_from_internet/>
  - Exploration and validation of the facial emotion dataset.
  - Download video sample.

1) Set up face extractor model (YuNet)
<./faces_from_images/>
  - Extract human faces and rectangular frames from an image.

2) Define neural architecture and train the emotion recognition model
<./emotion_from_face/>
  - Deep convolutional neural network.
  - Compiling, training, and saving of the network.
  - Lightweight version for smaller hardware.

3) Put everything together and predict facial emotions from a video
<./emotion_from_face/>
  - Save video frames in a specific folder under labeled folders.
  - Or simply run a webcam and see the detected emotion for each face directly.

## Author
g-ameline

