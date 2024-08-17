from tensorflow.keras import layers, models, backend
#from keras_applications import imagenet_utils
from deep_learning import algorithm

class MLPNet(algorithm.AlgorithmBase):
    TYPE = None

    def __init__(self, input_shape, parameter, output_units):
        self.output_units = output_units
        self.validation = parameter['validation']

        morphology = ['radius','distance','sigma_radius','omage_distance']
        if not parameter['distribution']:
            morphology = ['radius','distance']
        if parameter['morphology'] != 'all':
            morphology = [key for key in morphology if key in parameter['morphology']]

        if self.TYPE == "MLP2":
            self.activations = 'lr'
            self.model = self.build_mlp2(input_shape=input_shape, morphology=morphology)
        elif self.TYPE == "MLP3":
            self.model = self.build_mlp3(input_shape=input_shape, morphology=morphology)
        algorithm.AlgorithmBase.__init__(self, model=self.model, parameter=parameter, morphology=morphology, output_units=output_units)

    def build_mlp2(self, input_shape, morphology, hidden=1024):
        print('input_shape to build_model:', input_shape)
        #input_shape = imagenet_utils._obtain_input_shape(input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=True)
        image = layers.Input(shape=input_shape,name="image")
        x = layers.Flatten()(image)
        x = layers.Dense(units=hidden, name='fc' + str(hidden))(x)
        x = layers.LeakyReLU(alpha=0.03)(x) if self.activations[0]=='l' else layers.ReLU()(x)

        outputs = []
        if 'distance' in morphology:
            distance = layers.Dense(units=self.output_units['distance'], activation="softmax", name="distance")(x)
            outputs.append(distance)
        if 'omega_distance' in morphology:
            omega_distance = layers.Dense(units=self.output_units['omega_distance'], activation="softmax", name="omega_distance")(x)
            outputs.append(omega_distance)
        if 'radius' in morphology:
            radius = layers.Dense(units=self.output_units['radius'], activation="softmax", name="radius")(x)
            outputs.append(radius)
        if 'sigma_radius' in morphology:
            sigma_radius = layers.Dense(units=self.output_units['sigma_radius'], activation="softmax", name="sigma_radius")(x)
            outputs.append(sigma_radius)
        # create model
        model = models.Model(inputs=image, outputs=outputs, name=self.TYPE+"_"+self.activations)
        return model


class MLP2(MLPNet):
    TYPE = "MLP2"
    def __init__(self, input_shape, parameter, output_units):
        MLPNet.__init__(self, input_shape=input_shape, parameter=parameter, output_units=output_units)
