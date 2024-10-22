import os
import requests
from PIL import Image
from IPython.display import display
import zipfile
import tarfile
import shutil
import numpy
import pickle

def show_folders_and_files(start_path='./', folder_limit=20, files_limit=5):
# ├ └  ─ │
    limit = 0
    last_root = None
    counter=0
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep) + 1
        if (limit:=limit+1) > folder_limit:
            print(f"{' ' * level+'.../'}")
            break
        if root == start_path:
            print(os.path.normpath(root)+'/')
        if root != start_path:
            print(f"{' ' * level}{os.path.basename(root)}/")
        if len(files) > files_limit:
            sample_number = 3 if files_limit > 3 else files_limit
            for f in files[:sample_number]:
                print(f"{' ' * (level+1)+'├ '}{f}")
            print(f"{' ' * (level+1)}├ ... {len(files)-sample_number-1} files")
        if len(files) < files_limit:
            for f in files[:-1]:
                print(f"{' ' * (level+1)+'├ '}{f}")
        if (fs:=files[-1:]):
            print(f"{' ' * (level+1)+'└ '}{fs[0]}")       
            
def show_folders(start_path, folder_limit=20):
    limit = 0
    for root, _, _ in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep) + 1
        if (limit:=limit+1) > folder_limit:
            print(f"{' ' * level+'.../'}")
            break
        if root == start_path:
            print(os.path.normpath(root)+'/')
        if root != start_path:
            print(f"{' ' * level}{os.path.basename(root)}/")

def show_files(start_path, files_limit=10):
    print(os.path.normpath(start_path)+'/')
    files = list(f for f in os.listdir(start_path) if os.path.isfile(os.path.join(start_path, f)))
    if len(files) > files_limit:
        sample_number = 3 if files_limit > 3 else files_limit
        for f in files[:sample_number]:
            print(f" ├ {f}")
        print(f" ├ ... {len(files)-sample_number} files")
    if len(files) < files_limit:
        for f in files[:-1]:
            print(f" ├ {f}")
    if (fs:=files[-1:]):
        print(f" └ {fs[0]}")       

def file_names_from_folder_path(folder_path, files_limit=30):
    files_path = list(f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
    return files_path

def file_paths_from_folder_path(folder_path, files_limit=30):
    files_path = list(os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
    return files_path

def delete_everything_inside_folder(folder, file_name_to_keep=None):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if item == file_name_to_keep:
            continue
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def fetch_file_stream(url, destination_folder_path, data_file_name, redownload=False):
    downloaded_file_path = destination_folder_path + data_file_name
    if not redownload:
        if os.path.exists(downloaded_file_path):
            return downloaded_file_path 
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    if not os.path.exists(downloaded_file_path):
        response = requests.get(url, stream=True)
        with open(downloaded_file_path, 'wb') as downloading_file:
            for chunk in response.iter_content(chunk_size=128):
                downloading_file.write(chunk)
    return downloaded_file_path 
                
def unzip_files(zipped_file_path, destination_folder_path, check_path_already_exist=None):
    if check_path_already_exist:
        if os.path.exists(destination_folder_path):
            return
    try:
        with zipfile.ZipFile(zipped_file_path,"r") as zip_file:
            zip_file.extractall(destination_folder_path)
        return
    except:
        with tarfile.open(zipped_file_path, "r:gz") as gzip_file:
            gzip_file.extractall()
        return
        
def show_image_from_path(image_file_path):
    image = Image.open(image_file_path)
    display(image)

def image_as_matrix_from_path(image_file_path):
    return numpy.asarray(Image.open(image_file_path))

def save_image_to_png(image_matrix, destination_folder_path, image_file_name):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    image_png = Image.fromarray(image_matrix, 'L')
    image_png.save(image_file_path:=f"{destination_folder_path}{image_file_name}")
    return image_file_path
    

def pickled_file_path_from_object(the_object, destination_folder_path, file_name):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    destination_file_path = f"{destination_folder_path}{file_name}"
    with open(destination_file_path, 'wb') as handle:
        pickle.dump(the_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination_file_path 

def object_from_pickled_file_path(destination_file_path):
    with open(destination_file_path, 'rb') as handle:
        loaded_data = pickle.load(handle)
    return loaded_data

# def dilled_file_path_from_object(the_object, destination_folder_path, file_name):
#     if not os.path.exists(destination_folder_path):
#         os.makedirs(destination_folder_path)
#     destination_file_path = f"{destination_folder_path}{file_name}"
#     with open(destination_file_path, 'wb') as handle:
#         dill.dump(the_object, handle, protocol=dill.HIGHEST_PROTOCOL)
#     return destination_file_path 

# def object_from_dille_file_path(destination_file_path):
#     with open(destination_file_path, 'rb') as handle:
#         loaded_data = dill.load(handle)
#     return loaded_data
