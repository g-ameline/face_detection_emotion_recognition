import utils
import data 
import constant

augmentation_datas = [
    data.RandomRotationData(factor=0.005 ),
    data.RandomTranslationData(height_factor=0.01, width_factor=0.01 ),
    data.RandomFlipData(mode='horizontal' ),
]

preprocessing_datas = [
    data.ReshapeData( target_shape=(48, 48, 1) ),
]

dcnn_datas = [ #reference
        data.ConvolutionData(shape=(5,5),features=64,activation='elu'),data.BatchNormalizationData(),
        data.ConvolutionData(shape=(5,5),features=64,activation='elu'),data.BatchNormalizationData(),
        data.MaxPoolingData(shape=(2,2)), data.DropoutData(rate=0.4),
        data.ConvolutionData(shape=(3,3),features=128,activation='elu'),data.BatchNormalizationData(),
        data.ConvolutionData(shape=(3,3),features=128,activation='elu'),data.BatchNormalizationData(),
        data.MaxPoolingData(shape=(2,2)), data.DropoutData(rate=0.4),
        data.ConvolutionData(shape=(3,3),features=256,activation='elu'),data.BatchNormalizationData(),
        data.ConvolutionData(shape=(3,3),features=256,activation='elu'),data.BatchNormalizationData(),
        data.MaxPoolingData(shape=(2,2)), data.DropoutData(rate=0.5),
        data.FlattenData(),
        data.DenseData(units=128, activation='elu'),
        data.BatchNormalizationData(),
        data.DropoutData(rate=0.6),
        data.DenseData(units=7, activation='softmax'),
    ]
reference_datas = augmentation_datas + preprocessing_datas +  dcnn_datas

print(f"architecture datas: {reference_datas }")

architecture_file_path = utils.pickled_file_path_from_object(
    the_object=reference_datas,
    destination_folder_path=constant.architecture_folder_path,
    file_name=constant.architecture_file_name,
)

print(f"architecture datas saved at: {architecture_file_path}")

utils.show_folders_and_files('../architecture')
