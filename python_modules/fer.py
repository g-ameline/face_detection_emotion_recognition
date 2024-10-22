import utils
import pandas
import numpy
import os
import constant

def emotion_name_from_emotion_number(emotion_number):
    return constant.emotion_names[emotion_number]

def emotion_number_from_emotion_name(emotion_name):
    return constant.emotion_names.index(emotion_name)

def trim_column_names(data_frame):
    return data_frame.rename(
        columns={column: column.strip() for column in data_frame.columns}
    )

def get_data_from_path(
    data_path, 
    input_image_format='numerals',# 'bytes', 'numerals' # filepath | numerals | bytes 
    output_image_format='matrix',# 'numerals', 'bytes', 
    sampling_quantity=None,
):
    data_frame = pandas.read_csv(
        data_path, 
        index_col=0,
        dtype={
            # 'emotion':numpy.uint8, 'image':'string'}
            'emotion':pandas.StringDtype(), 'image':pandas.StringDtype()}
    )

    # data_frame['emotion'] = reformatted_emotion_series(data_frame['emotion'])
    data_frame['emotion'] = data_frame['emotion'].astype(pandas.CategoricalDtype()) 
    # data_frame['emotion'] = data_frame['emotion'].astype(numpy.uint8) 
    #seems that keras want perticular combo of dtype and class_mode
    # try either  .astype(uint8) * class_mode='raw' | .astype(string) * class_mode='categorical'

    if sampling_quantity:
        data_frame=sampled_data_frame(
            data_frame=data_frame,
            sampling_quantity=sampling_quantity,
            to_be_preserved_column_names=['emotion'],
        )
    if input_image_format == 'filepath': # nothing to do
        assert output_image_format == 'filepath'
        return data_frame
    if input_image_format == 'bytes': # hopefully we already have bytes as bytes
        assert output_image_format == 'bytes'
        data_frame['image'] = bytes_series_from_numerals_string_series(data_frame['image'])
        return data_frame
    if input_image_format == 'numerals': # hopefully we have numeral string
        if output_image_format == 'numerals':
            return data_frame
        if output_image_format == 'integers':
            data_frame['image'] = integers_series_from_numerals_string_series(data_frame['image'])
            return data_frame
        if output_image_format == 'matrix':
            data_frame['image'] = uint8_matrix_series_from_numerals_string_series(data_frame['image'])
            return data_frame
    raise Exception(f"failed to understand what the image is {nput_image_format = } and what it should be {output_image_format = }")

def sampled_data_frame(
    data_frame, 
    sampling_quantity, 
    to_be_preserved_column_names=['set','emotion'],
):
    unique_valuess = list(
        data_frame[column_name].unique()
        for column_name in to_be_preserved_column_names
    )
    import itertools
    products_of_values_to_preserve = itertools.product(*unique_valuess)
    result = pandas.DataFrame()   
    for one_set_of_unique_values in products_of_values_to_preserve:
        condition = pandas.Series([True] * len(data_frame), index=data_frame.index)
        for column_name, unique_value in zip(to_be_preserved_column_names, one_set_of_unique_values):
            condition = condition & (data_frame[column_name]==unique_value)
        if type(sampling_quantity) == int:
            number_of_samples = sampling_quantity if sampling_quantity<condition.sum() else condition.sum()
            result = pandas.concat(
                [result, data_frame[condition].sample(n=number_of_samples)]
            )
        if type(sampling_quantity) == float:
            sample_proportion = sampling_quantity
            result = pandas.concat(
                [result, data_frame[condition].sample(frac=sample_proportion)]
            )
    return result

def assert_emotion_series_represent_a_number_from_0_to_6(emotion_number_label_series):
    list(
        number if 0<=number<=6 else 1/0
        for number in emotion_number_label_series
    )

def assert_string_series_represent_48x48_uint8_series(string_series):
    list(
        list(
            int(numerale) if 0<=int(numerale)<=255 else 1/0
            for numerale in numerals.split() 
        ) if len(numerals.split())==48*48 else 1/0
        for numerals in string_series
    )

def reformatted_set_series(
    set_series,
    old_set_names=['Training', 'PublicTest', 'PrivateTest'],
    new_set_names=['train', 'validate', 'test'],
):
    return (
        set_series.astype('category').cat.rename_categories(
                {old_name: new_name for old_name,new_name in zip(old_set_names, new_set_names)}
    ) )

def reformatted_emotion_series(emotion_series):
    # return keras.utils.to_categorical(emotion_series)
    return emotion_series.astype(numpy.uint8).astype('category')

def integers_series_from_numerals_string_series(numbers_string_series, number_type=numpy.uint8):
    return pandas.Series(
        data=(
            numpy.array( [number_type(numerale) for numerale in numerals.split()] ) 
            for numerals in numbers_string_series
        ),
        index=numbers_string_series.index,
    )

def bytes_series_from_uint8s_series(uint8s_series):
    return pandas.Series(
        data=(uint8s.tobytes() for uint8s in uint8s_series)
        ,
        index=uint8s_series.index
    )

def matrix_series_from_uint8s_series(integers_series, width=None, height=None):
    return pandas.Series(
        data=(
            numpy.array(
                [pixel for pixel in pixels]
            ).reshape(width,height).astype('float32')
            for pixels in integers_series
        ),
        index=integers_series.index
    )

def uint8_matrix_series_from_numerals_string_series(numerals_string_series):
    uint8s_series = integers_series_from_numerals_string_series(
        numerals_string_series,
        number_type=numpy.uint8,
    )
    uint8s_matrix_series = matrix_series_from_uint8s_series(
        uint8s_series,
        width=48, height=48,
    )
    return uint8s_matrix_series

def bytes_series_from_numerals_string_series(numerals_string_series):
    uint8s_series = integers_series_from_numerals_string_series(
        numerals_string_series,
        number_type=numpy.uint8,
    )
    bytes_series = bytes_series_from_uint8s_series(
        uint8s_series,
    )
    return bytes_series


def saved_image_file_path_from_image_matrix(
    image_matrix,
    data_folder_path,
    set_name,
    emotion_name,
    index,
):
    image_folder_path = f"{data_folder_path}{set_name}/images/{emotion_name}/"
    print(f'{image_folder_path = }')
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
    utils.save_image_to_png(image_matrix, f"{image_folder_path}{index}.png")
    return image_file_path
    
def saved_image_file_path_series_from_data_frame(
    set_series,
    emotion_series,
    image_matrix_series,
    data_folder_path=constant.data_folder_path,
    emotion_names=constant.emotion_names,
    folder_structure='dataset/emotion_number/index.png',
    # folder_structure='dataset/images/emotion_name/index.png'
):
    # determine the file path where we gonna save each image from its set and emotion label(name)
    if folder_structure == 'dataset/emotion_number/index.png':
        print(f"saving {data_folder_path}set_name/emotion_number/index.png")
        image_folder_paths_and_image_file_names = (
            (
                f"{data_folder_path}{set_name}/{emotion_number}/",
                f"{index}.png",
            )
            for set_name, emotion_number, index 
            in zip(set_series, emotion_series, image_matrix_series.index) 
        )

    if folder_structure == 'dataset/images/emotion_name/index.png':
        print(f"saving {data_folder_path}set_name/images/emotion_name/index.png")
        image_folder_paths_and_image_file_names = (
            (
                f"{data_folder_path}{set_name}/images/{emotion_names[emotion_number]}/",
                f"{index}.png",
            ) 
            for set_name, emotion_number, index 
            in zip(set_series, emotion_series, image_matrix_series.index) 
        )
    # save images as png
    image_file_paths = list( 
        utils.save_image_to_png(image_matrix, destination_folder_path, image_file_name)
        for image_matrix, (destination_folder_path, image_file_name)
        in zip(image_matrix_series, image_folder_paths_and_image_file_names)
    )
    print(f"image file paths: {image_file_paths[:5] = }")
    # return saved file paths as series
    saved_paths = pandas.Series(
        data=image_file_paths,
        index=image_matrix_series.index,
    )
    print(f"{saved_paths= }")
    return saved_paths

def display_sample_of_each_set_of_each_emotion(
    data_frame,
    data_folder_path, 
    emotion_names=constant.emotion_names,
):
    for set_name in data_frame['set'].unique():
        for emotion_index in data_frame['emotion'].unique():
            emotion_name = constant.emotion_names[emotion_index]
            data_of_the_set = data_frame[ data_frame['set']==set_name ] 
            data_of_the_set_and_the_emotion = data_of_the_set[ data_of_the_set['emotion']==emotion_index ]
            print(f"{data_folder_path}{set_name}/images/{emotion_name}/")
            n_rows = len(data_of_the_set_and_the_emotion)
            if n_rows > 0:
                n_samples = n_rows if n_rows < 3 else 3
                indices = data_of_the_set_and_the_emotion.sample(n=n_samples).index
                for index in indices:
                    image_path = f"{data_folder_path}{set_name}/images/{emotion_name}/{index}.png"
                    utils.show_image_from_path(image_path)

def saved_data_set_paths_from_dataframe(data_frame, data_folder_path, set_names):

    def saved_data_set_frame_path(
        data_set_frame, 
        saving_file_path, 
        to_be_saved_column_names=['emotion','image']
    ):
        data_set_frame.to_csv(
            saving_file_path,
            columns=to_be_saved_column_names,
            index=True,#seems like autogluon wants the indexes
        )
        return saving_file_path

    for set_name in set_names:
        set_folder_path = f"{data_folder_path}{set_name}/"
        if not os.path.exists(set_folder_path):
            os.makedirs(set_folder_path)
        data_file_name = 'data.csv'

    (
        saved_data_set_frame_path(
            data_set_frame=data_frame[data_frame['set'] == set_name],
            saving_file_path=f"{set_folder_path}{data_file_name }",
            to_be_saved_column_names=['emotion','image'],
        ) 
        for set_name in set_names
    )

    return {
        saved_data_set_frame_path(
            data_frame[data_frame['set'] == set_name],
            f"{data_folder_path}{set_name}/data.csv",
        ) 
        for set_name in set_names
    }

def saved_facial_emotion_recognition_data_sets_from_url(
    download_url=constant.data_url, 
    data_folder_path=constant.data_folder_path,
    downloaded_data_file_name=constant.downloaded_data_file_name,
    unzipped_data_folder_name=constant.unzipped_data_folder_name,
    all_data_file_name=constant.all_data_file_name,
    old_and_new_set_column_names=('Usage', 'set'),
    old_and_new_set_names=(constant.legacy_data_set_names, constant.data_set_names),
    image_saving_format='numerals', # numerals | filepath | bytes
    # 'dataset/class/index.png',
    # 'dataset/dataframe.csv and dataset/images/class/index.png'
    sampling=None,
    merge_training_and_validate=False,
):
    if not os.path.exists(data_folder_path):
        print(f"\ncreating data_folder: {data_folder_path}")
        os.makedirs(data_folder_path)

    print(f"\ndownloading raw data from: {download_url}")
    print(f"at: {data_folder_path}{downloaded_data_file_name}")
    utils.fetch_file_stream(
        url=download_url,
        destination_folder_path=data_folder_path,
        data_file_name=downloaded_data_file_name,
    )
    print(f"\nunizpping raw data from: {data_folder_path}{downloaded_data_file_name}")
    print(f"into: {data_folder_path}{unzipped_data_folder_name}")
    utils.unzip_files(
        zipped_file_path=f"{data_folder_path}{downloaded_data_file_name}",
        destination_folder_path=f"{data_folder_path}{unzipped_data_folder_name}",
    )

    print(f"\nreading data from: {data_folder_path}{unzipped_data_folder_name}data/{all_data_file_name}")
    print(f"as data frame")
    all_data_path = f"{data_folder_path}{unzipped_data_folder_name}data/{all_data_file_name}"
    all_data_frame = pandas.read_csv(all_data_path)
    
    print(f"\ndelete zipped and unzipped data")
    utils.delete_everything_inside_folder(folder=data_folder_path, file_name_to_keep=all_data_file_name)

    all_data_frame = trim_column_names(all_data_frame)
    
    old_set_column_name, new_set_column_name = old_and_new_set_column_names
    old_set_names, new_set_names = old_and_new_set_names
    print(f"\nchecking data are conform")
    assert len(all_data_frame) == 35887
    assert len(all_data_frame[all_data_frame[old_set_column_name]=='Training']) == 28709
    assert len(all_data_frame[all_data_frame[old_set_column_name]=='PublicTest']) == 3589

    if sampling:
        all_data_frame=sampled_data_frame(
            data_frame=all_data_frame,
            sampling_quantity=sampling,
            to_be_preserved_column_names=[old_set_column_name,'emotion'],
        )
    
    assert_emotion_series_represent_a_number_from_0_to_6(all_data_frame['emotion'])
    assert_string_series_represent_48x48_uint8_series(all_data_frame['pixels'])

    old_set_column_name, new_set_column_name = old_and_new_set_column_names
    old_set_names, new_set_names = old_and_new_set_names
    all_data_frame[new_set_column_name] = reformatted_set_series(all_data_frame[old_set_column_name])
    all_data_frame = all_data_frame.drop(columns=old_set_column_name)
    
    all_data_frame['emotion'] = reformatted_emotion_series(all_data_frame['emotion'])

    if image_saving_format == 'bytes':
        print(f"\nunder folder for each data set")
        print(f"saving data as csv")
        print(f"with images as bytes")
        all_data_frame['image'] = bytes_series_from_numerals_string_series(all_data_frame['pixels'])

    if image_saving_format == 'dataset/data.csv and dataset/images/emotion_name/index.png':
        print(f"\nunder folder for each data set")
        print(f"saving data as csv")
        print(f"with images as filepath to subfolder with emotion name")
        images_as_matrix_series = uint8_matrix_series_from_numerals_string_series(all_data_frame['pixels'])
        all_data_frame['image'] = saved_image_file_path_series_from_data_frame(
            set_series=all_data_frame['set'],
            emotion_series=all_data_frame['emotion'],
            image_matrix_series=images_as_matrix_series,
            data_folder_path=data_folder_path,
            folder_structure='dataset/images/emotion_name/index.png',
        )
    if image_saving_format == 'dataset/class/index.png':
        print(f"\nunder folder for each data set")
        print(f"saving images as .png under emotion number folder")
        images_as_matrix_series = uint8_matrix_series_from_numerals_string_series(all_data_frame['pixels'])
        all_data_frame['image'] = saved_image_file_path_series_from_data_frame(
            set_series=all_data_frame['set'],
            emotion_series=all_data_frame['emotion'],
            image_matrix_series=images_as_matrix_series,
            data_folder_path=data_folder_path,
            folder_structure='dataset/emotion_number/index.png',
        )
        print("done !")
        return None

    if image_saving_format == 'numerals':
        print(f"\nunder folder for each data set")
        print(f"saving data as csv")
        print(f"with images as uint8s (as is actually)")
        all_data_frame['image'] = all_data_frame['pixels']

    # if image_saving_format == 'matrix':
    #     print(f"\nunder folder for each data set")
    #     print(f"saving data as csv")
    #     print(f"with images as [48][48]uint8 matrix")
    #     all_data_frame['image'] = uint8_matrix_series_from_numerals_string_series(all_data_frame['pixels'])

    if merge_training_and_validate:
        all_data_frame['set'] = all_data_frame['set'].replace(new_set_names[1],new_set_names[0])
        new_set_names = [new_set_names[0], new_set_names[2]]
        
    all_data_frame = all_data_frame.drop(columns=['pixels'])

    saved_data_frame_paths = saved_data_set_paths_from_dataframe(
        all_data_frame, 
        data_folder_path,
        new_set_names,
    )

    print(f"done {saved_data_frame_paths = }")
    return saved_data_frame_paths

class Plotting:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pyplot.tight_layout()
        pyplot.show()

def prepare_image_plotting_from_matrices(image_matrices, labels=None):
    image = Image.open(image_file_path)
    display(image)

def prepare_image_plotting_from_matrix(image_matrix, label=None):
    ax = axes[i, j]
    ax.imshow(train_images[idx], cmap='gray')
    if label:
        ax.set_title(f"{label}")
    ax.axis("off")

