import keras
import random

def is_natural(x):
    return x%1==0 and x>=0 and isinstance(x,int)
def is_counting(x):
    return x%1==0 and x>0 and isinstance(x,int)

class ConvolutionData():
    def __new__(cls, features=1, shape=(3,3), activation='relu'):
        assert is_counting
        assert is_counting(shape[0]) and is_counting(shape[1]) and len(shape) == 2
        instance = object().__new__(cls)
        instance.__dict__['features'] = features
        instance.__dict__['shape'] = shape
        instance.__dict__['activation'] = activation
        return instance
                
    def __init__(the_convolution_data, features=1, shape=(1,1), activation='relu'):
        pass

    def doubled_features(the_convolution_data):
        new_features = the_convolution_data.features+1
        return ConvolutionData(new_features, the_convolution_data.shape)

    def halved_features(the_convolution_data):
        new_features = max(1,the_convolution_data.features-1)
        return ConvolutionData(new_features, the_convolution_data.shape)

    def widened_kernel(the_convolution_data):
        width, height = the_convolution_data.shape
        return ConvolutionData(
            the_convolution_data.features, 
            (width+1,height),
        )
        
    def unwidened_kernel(the_convolution_data):
        width, height = the_convolution_data.shape
        return ConvolutionData(
            the_convolution_data.features, 
            (max(1,width-1),height),
        )
        
    def heightened_kernel(the_convolution_data):
        width, height = the_convolution_data.shape
        return ConvolutionData(
            the_convolution_data.features,
            (width,height+1),
        )
    def unheightened_kernel(the_convolution_data):
        width, height = the_convolution_data.shape
        return ConvolutionData(
            the_convolution_data.features,
            (width,max(1,height-1)),
        )
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_convolution_data):
        return f"convolution data of {the_convolution_data.shape} kernel with {the_convolution_data.features} features"
    def __repr__(the_convolution_data):
        return the_convolution_data.__str__()
    def keras_layer(the_data):
        return keras.layers.Conv2D(
                filters=the_data.features,
                kernel_size=the_data.shape,
                padding='valid',
                activation=the_data.activation,
                # name=the_data.__str__(),
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return ConvolutionData(**the_data.__dict__)
    def mutated(the_data):
        make_new_data = random.choice([
            the_data.doubled_features,
            the_data.halved_features,
            the_data.heightened_kernel,
            the_data.unheightened_kernel,
            the_data.widened_kernel,
            the_data.unwidened_kernel,
        ])
        return make_new_data()
        
        

if __name__ == "__main__":
    def test():
        cd = ConvolutionData(features=3, shape=(1,1))
        print(cd)
        cd.heightened_kernel()
    test()


class FlattenData():
    def __new__(cls):
        instance = object().__new__(cls)
        return instance                
    def __init__(cls):
        pass
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_dense_data):
        return f"flatten data"
    def __repr__(the_dense_data):
        return the_dense_data.__str__()
    def keras_layer(the_data):
        return keras.layers.Flatten(
            # name=the_data.__str__(),
        )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return FlattenData(**the_data.__dict__)

if __name__ == "__main__":
    def test():
        FlattenData()
    test()

class FlattenAndDenseData():
    def __new__(cls, units, activation='softmax'):
        assert is_counting(units)
        instance = object().__new__(cls)
        instance.__dict__['units'] = units
        instance.__dict__['activation'] = activation
        return instance
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_flatten_and_dense_data):
        return f"{the_flatten_and_dense_data.units} categorical outputs data"
    def __repr__(the_flatten_and_dense_data):
        return the_flatten_and_dense_data.__str__()
    def keras_layer(the_flatten_and_dense_data):
        def inducted_flatten_and_dense_layer_together(parent_layer):
            return keras.layers.Dense(
                units=the_flatten_and_dense_data.units,
                activation=the_flatten_and_dense_data.activation,
                name="flatten_before_neural_categorizing",
            )(
                keras.layers.Flatten(
                    name="neural_categorizing",
                )(parent_layer)
            )
        return inducted_flatten_and_dense_layer_together
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return FlattenAndDenseData(**the_data.__dict__)

class DenseData():
    def __new__(cls, units, activation='relu'):
        assert is_counting(units)
        instance = object().__new__(cls)
        instance.__dict__['units'] = units
        instance.__dict__['activation'] = activation
        return instance
    # def __init__(the_data, units, activation='relu'):
    #     pass
    def augment_units(the_dense_data):
        return DenseData(the_dense_data.units+1)
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_dense_data):
        return f"dense data of {the_dense_data.units} units"
    def __repr__(the_dense_data):
        return the_dense_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.Dense(
                units=the_data.units,
                activation=the_data.activation,
                # name=the_data.__str__(),
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return DenseData(**the_data.__dict__)


if __name__ == "__main__":
    def test():
        d = DenseData(units=34)
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()
    
# data.data_from_keras_layer( keras.layers.RandomRotation, factor=0.005 ),

class RandomRotationData():
    def __new__(cls, factor=0.005):
        instance = object().__new__(cls)
        instance.__dict__['factor'] = factor
        return instance
    # def __init__(the_data, units, activation='relu'):
    #     pass
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"rotation data of factor {the_data.factor}"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.RandomRotation(
                factor=the_data.factor,
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return RandomRotationData(**the_data.__dict__)

# data.data_from_keras_layer( keras.layers.RandomTranslation, height_factor=0.01, width_factor=0.01 ),
class RandomTranslationData():
    def __new__(cls, height_factor=0.01, width_factor=0.01):
        instance = object().__new__(cls)
        instance.__dict__['width_factor'] = width_factor
        instance.__dict__['height_factor'] = height_factor
        return instance
    # def __init__(the_data, units, activation='relu'):
    #     pass
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"translation data of factor {the_data.height_factor} & {the_data.width_factor}"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.RandomTranslation(
                width_factor=the_data.width_factor,
                height_factor=the_data.height_factor,
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return RandomTranslationData(**the_data.__dict__)

# data.data_from_keras_layer( keras.layers.RandomFlip, mode='horizontal' ),

class RandomFlipData():
    def __new__(cls, mode='horizontal'):
        instance = object().__new__(cls)
        instance.__dict__['mode'] = mode
        return instance
    # def __init__(the_data, units, activation='relu'):
    #     pass
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"flipdata of mode {the_data.mode}"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.RandomFlip(
                mode=the_data.mode,
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return RandomFlipData(**the_data.__dict__)

def data_from_keras_layer(keras_layer, **kwargs):
    class NewData():
        def __new__(cls, **kwargs):
            instance = object().__new__(cls)
            for key, value in kwargs.items():
                instance.__dict__[key] = value
            return instance
        def __setattr__(the_data, key, _value):
            raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
        def __str__(the_dense_data):
            return f"new data from {keras_layer} with {kwargs}"
        def __repr__(the_data):
            return the_data.__str__()
        def keras_layer(the_data): 
            return keras_layer(
               **kwargs,
               # name=the_data.__str__(),
            )
        def __getnewargs__(the_data):
            return tuple(the_data.__dict__.values())
        def clone(the_data):
            return NewData(**the_data.__dict__)
    return NewData(**kwargs)

if __name__ == "__main__":
    def test():
        data_from_keras_layer(keras.layers.Normalization, axis=1, invert=True).keras_layer()
    test()

class ReshapeData():
    def __new__(cls, target_shape=(48,48)):
        instance = object().__new__(cls)
        instance.__dict__['target_shape'] = target_shape
        return instance
    # def __init__(the_data, units, activation='relu'):
    #     pass
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"reshape data of target_shape {the_data.target_shape}"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.Reshape(
                target_shape=the_data.target_shape,
            )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return ReshapeData(**the_data.__dict__)

class IdentityData():
    def __new__(cls):
        instance = object().__new__(cls)
        return instance
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_id_data):
        return f"id data"
    def __repr__(the_id_data):
        return the_id_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.Identity()
    def clone(the_data):
        return IdentityData()

if __name__ == "__main__":
    def test():
        d = IdentityData()
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()

class MaxPoolingData():
    def __new__(cls, shape=(2,2) ):
        assert all(is_counting(length) for length in shape)
        instance = object().__new__(cls)
        instance.__dict__['shape'] = shape
        return instance
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_pooling_data):
        return f"max pooling data {the_pooling_data.shape}"
    def __repr__(the_pooling_data):
        return the_pooling_data.__str__()

    def heightened_kernel(the_max_pooling_data):
        width, height = the_max_pooling_data.shape
        new_shape = width, height+1
        return MaxPoolingData(
            shape=new_shape,
        )
    def unheightened_kernel(the_max_pooling_data):
        width, height = the_max_pooling_data.shape
        new_shape = width, max(height-1,1)
        return MaxPoolingData(
            shape=new_shape,
        )
    def widened_kernel(the_max_pooling_data):
        width, height = the_max_pooling_data.shape
        new_shape = width+1, height
        return MaxPoolingData(
            shape=new_shape,
        )
    def unwidened_kernel(the_max_pooling_data):
        width, height = the_max_pooling_data.shape
        new_shape = max(width-1,1), height, 
        return MaxPoolingData(
            shape=new_shape,
        )
    def mutated(the_data):
        make_new_data = random.choice([
            the_data.heightened_kernel,
            the_data.unheightened_kernel,
            the_data.widened_kernel,
            the_data.unwidened_kernel,
        ])
        return make_new_data()
    
    def keras_layer(the_data): 
        return keras.layers.MaxPooling2D(
            pool_size=(2,2),
            # name=the_data.__str__(),
        )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return MaxPoolingData(**the_data.__dict__)
    

if __name__ == "__main__":
    def test():
        d = MaxPoolingData(shape=(2,2))
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()
    
class DropoutData():
    def __new__(cls, rate=0.4):
        instance = object().__new__(cls)
        instance.__dict__['rate'] = rate
        return instance
                                
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_dropout_data):
        return f"max pooling data {the_dropout_data.rate}"
    def __repr__(the_dropout_data):
        return the_dropout_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.Dropout(
            rate=the_data.rate,
            # name=the_data.__str__(),
        )
    def reduced_rate(the_dropout_data):
        rate = the_dropout_data.rate
        new_rate = rate/2
        return DropoutData(
            rate=new_rate,
        )
    def increased_rate(the_dropout_data):
        rate = the_dropout_data.rate
        new_rate = (1-rate)/2 +rate
        return DropoutData(
            rate=new_rate,
        )
    def mutated(the_data):
        make_new_data = random.choice([
            the_data.reduced_rate,
            the_data.increased_rate,
        ])
        return make_new_data()
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return DropoutData(**the_data.__dict__)

if __name__ == "__main__":
    def test():
        d = DropoutData(rate=0.6)
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()

class BatchNormalizationData():
    def __new__(cls):
        instance = object().__new__(cls)
        return instance
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"batch normalization data"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.BatchNormalization(
            # name=the_data.__str__()
        )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return BatchNormalizationData(**the_data.__dict__)

if __name__ == "__main__":
    def test():
        d = BatchNormalizationData()
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()

class LayerNormalizationData():
    def __new__(cls):
        instance = object().__new__(cls)
        return instance
    def __setattr__(the_data, key, _value):
        raise Exception(f"Cannot modify attribute '{key}' of an immutable class {the_data = }")
    def __str__(the_data):
        return f"batch normalization data"
    def __repr__(the_data):
        return the_data.__str__()
    def keras_layer(the_data): 
        return keras.layers.LayerNormalization(
            # name=the_data.__str__()
        )
    def __getnewargs__(the_data):
        return tuple(the_data.__dict__.values())
    def clone(the_data):
        return LayerNormalizationData(**the_data.__dict__)

if __name__ == "__main__":
    def test():
        d = LayerNormalizationData()
        print(d)
        ld = d.keras_layer()
        print(ld)
    test()

