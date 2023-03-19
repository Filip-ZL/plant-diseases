import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, \
                                    Activation, Lambda, Add, Input, \
                                    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model
import numpy as np

class Resnet:

    def __init__(self, 
                 input_shape: tuple = (224, 224, 3), 
                 n: int = 3, 
                 classes: tuple = (None),
                 **kwargs: any) -> None:
        
        self.input_shape = input_shape
        self.n = n
        self.classes = classes

        try:
            self.model = self.__build_model(kwargs)
        except Exception as e:
            print(f"Unable to build model with following error: \n { e }")
            self.model = tf.keras.Model(inputs=self.input_shape,
                                        outputs=Dense(len(self.classes), 
                                        activation="sigmoid")
                                       )
            
    def __call__(self) -> any:
        return self.model
    
    def fit_data(self, data, labels, callbacks=None,
                 batch_size=64, epochs=10, validation_split=.2):
        history = self.model.fit(data,
                                 labels,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_split=validation_split)
        return history

    def residual_block(x, number_of_filters, match_filter_size,  **kwargs):

        x_skip = x

        if match_filter_size:
            x = Conv2D(number_of_filters, 
                       kernel_size=(3, 3), 
                       strides=(2, 2), 
                       kernel_initializer=kwargs.get('initializer'), 
                       padding="same")(x_skip)
        else:
            x = Conv2D(number_of_filters,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       kernel_initalizer=kwargs.get('initializer'),
                       padding="same")(x_skip)
        
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(number_of_filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   kernel_initializer=kwargs.get("initializer"),
                   paddint="same")(x)
        x = BatchNormalization(axis=3)(x)

        if match_filter_size and kwargs.get("shortcut_type") == "identity":
            x_skip = Lambda(lambda x: tf.pad(x[:, ::2, ::2, :], 
                                             tf.constant([[0, 0,], [0, 0], [0, 0,], 
                                                          [number_of_filters // 4, 
                                                           number_of_filters // 4]]), 
                                             mode="CONSTANT"))(x_skip)
        elif match_filter_size and kwargs.get("shortcut_type") == "projection":
            x_skip = Conv2D(number_of_filters,
                            kernel_size=(1, 1),
                            kernel_initializer=kwargs.get("initializer"),
                            strides=(2, 2))(x_skip)

        x = Add()([x, x_skip])
        x = Activation("relu")(x)

        return x

    def residual_blocks(self, x, **kwargs):

        filter_size = kwargs.get("initial_no_filters")
        
        for layer_group in range(3):
            for block in range(kwargs.get("stack_n")):
                if layer_group > 0 and block == 0:
                    filter_size *= 2
                    x = self.residual_block(x, 
                                            number_of_filters=filter_size,
                                            match_filter_size=True)
                
                else:
                    x = self.residual_block(x,
                                            number_of_filters=filter_size)

        return x


    def model(self, **kwargs):        
        inputs = Input(shape=self.input_shape)
        x = Conv2D(kwargs.get("initial_no_filters"),
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   kernel_initializer=kwargs.get("initializer"),
                   padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = self.residual_blocks(x)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        outputs = Dense(len(self.classes))

        return inputs, outputs

    def __build_model(self, **kwargs):
        
        inputs, outputs = self.model(kwargs)

        model = Model(inputs, outputs, name=kwargs.get("name"))
        model.compile(loss=kwargs.get("loss"),
                      optimizer=kwargs.get("optim"),
                      metrics=kwargs.get("optim_additional_metrics"))

        model.summary()

        self.model = model


    

