from tensorflow.keras import layers, models, backend
#from keras_applications import imagenet_utils
from deep_learning import algorithm
import tensorflow as tf

class MLPNet(algorithm.AlgorithmBase):
    TYPE = None

    def __init__(self, input_shape, parameter, output_units):
        self.output_units = output_units
        self.validation = parameter['validation']
        self.check = parameter['check']
        self.informed = parameter['informed']
        self.loss = parameter['loss']


        morphology = ['radius','distance','sigma_radius','omage_distance']
        if not parameter['distribution']:
            morphology = ['radius','distance']
        if parameter['morphology'] != 'all':
            morphology = [key for key in morphology if parameter['morphology'] in key]

        if self.TYPE == "MLP2":
            self.activations = 'lr'
            self.model = self.build_mlp2(input_shape=input_shape, morphology=morphology)
        elif self.TYPE == "MLP3":
            self.model = self.build_mlp3(input_shape=input_shape, morphology=morphology)
        algorithm.AlgorithmBase.__init__(self, model=self.model, parameter=parameter, morphology=morphology, output_units=output_units)

    def build_mlp2(self, input_shape, morphology, hidden=1024):
        #input_shape = imagenet_utils._obtain_input_shape(input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=True)
        image = layers.Input(shape=input_shape,name="image")

        if self.informed: 
            label_input_shape = (260, 380, 1)  # Shape for labels
            label_input = layers.Input(shape=label_input_shape, name="label_input")

        x = layers.Flatten()(image)
        x = layers.Dense(units=hidden, name='fc' + str(hidden))(x)
        x = layers.LeakyReLU(negative_slope=0.03)(x) if self.activations[0]=='l' else layers.ReLU()(x)
        x = layers.Dense(units=hidden//2, name='fc' + str(hidden//2))(x)
        x = layers.LeakyReLU(negative_slope=0.03)(x) if self.activations[1]=='l' else layers.ReLU()(x)

        outputs = []
        if 'radius' in morphology:
            radius = layers.Dense(units=self.output_units['radius'], activation="softmax", name="radius")(x)
            outputs.append(radius)        
        if 'sigma_radius' in morphology:
            sigma_radius = layers.Dense(units=self.output_units['sigma'], activation="softmax", name="sigma_radius")(x)
            outputs.append(sigma_radius)
        if 'distance' in morphology:
            distance = layers.Dense(units=self.output_units['distance'], activation="softmax", name="distance")(x)
            outputs.append(distance)
        if 'omega_distance' in morphology:
            omega_distance = layers.Dense(units=self.output_units['omega'], activation="softmax", name="omega_distance")(x)
            outputs.append(omega_distance)
        # create model
        if self.informed: 

            outputs = distance

            # Expand dimensions to make the tensor 3D: (batch_size, 380, 1)
            expanded_outputs = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(outputs)  # Insert new axis at position 1 -> (batch_size, 1, 380)
            expanded_outputs = layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(expanded_outputs) # Insert new axis at position 3 -> (batch_size, 1, 380, 1)
            # Tile it along the new dimension to match the desired shape: (batch_size, 260, 380, 1)
            tiled_outputs = layers.Lambda(lambda x: tf.tile(x, [1, 260, 1, 1]))(expanded_outputs)  # Tile along axis=1 -> (batch_size, 260, 380, 1)

            # Concatenate the input image with the tiled outputs along the last dimension
            final_output = layers.Concatenate(axis=2)([image, tiled_outputs])  # Concatenate on the last axis
            model = models.Model(inputs=image, outputs=[final_output], name=self.TYPE+"_"+self.activations)

        else:
            model = models.Model(inputs=image, outputs=outputs, name=self.TYPE+"_"+self.activations)


        print(model.summary())
        return model

    def build_mlp2_GRU(self, input_shape, morphology, hidden=1024):
        image = layers.Input(shape=input_shape, name="image")
        x = layers.Flatten()(image)
        x = layers.GRU(units=hidden,shape=input_shape, stateful=True, return_sequences=False, name='gru_layer')(x)
        x = layers.LeakyReLU(negative_slope=0.03)(x) if self.activations[0] == 'l' else layers.ReLU()(x)

        outputs = []
        radius = layers.Dense(units=self.output_units['radius'], activation="softmax", name="radius")(x)
        outputs.append(radius)

        model = models.Model(inputs=image, outputs=outputs, name=self.TYPE + "_" + self.activations)
        return model

class MLP2(MLPNet):
    TYPE = "MLP2"
    def __init__(self, input_shape, parameter, output_units):
        MLPNet.__init__(self, input_shape=input_shape, parameter=parameter, output_units=output_units)
