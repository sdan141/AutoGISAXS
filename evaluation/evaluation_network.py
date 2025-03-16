import sys  
path_project = '/home/danshach/pot_store/gpfs_scratch/Xalantir'
sys.path.insert(1, path_project)
import tensorflow as tf
import main, arguments
from base import utilities, data_augmentation, detector, real_dataloader, experiment_dataloader, simulation_dataloader, setup
from tensorflow.keras import models, utils, callbacks, optimizers, losses, metrics
import numpy

def test_random_data():
    random_data = numpy.random.uniform(size=(100,260,220))
    predictions_labels = model.predict(random_data)
    predictions_max, predictions_avg  = algorithm.get_numerical_prediction(predictions_labels)
    print(predictions_avg)
    print(predictions_max)

def test_real_data(images):
    images = algorithm.create_input_tensor(images)
    predictions_labels = model.predict(images)
    predictions_max, predictions_avg  = algorithm.get_numerical_prediction(predictions_labels)
    print(predictions_avg)
    print(predictions_max)

def test_randomized_exp_data(images, targets):
    targets_shuffled = targets.sample(frac = 1)
    images = numpy.array(images)
    images_shuffled = images[targets_shuffled.index]
    print(targets_shuffled.index)
    images_shuffled = algorithm.create_input_tensor(images_shuffled)
    predictions_labels_shuffled = model.predict(images_shuffled)
    _, predictions_avg_shuffled  = algorithm.get_numerical_prediction(predictions_labels_shuffled)
    print(numpy.round(targets['radius'],2))
    print(sorted(numpy.round(predictions_avg_shuffled['radius'],2)))
    images = algorithm.create_input_tensor(images)
    predictions_labels= model.predict(images)
    _, predictions_avg  = algorithm.get_numerical_prediction(predictions_labels)
    print(sorted(numpy.round(predictions_avg['radius'],2)))

model_to_load = '/home/danshach/pot_store/gpfs_scratch/Xalantir/results/radius/mlp2/run_133/round_0/weights_epoch-02.weights.h5'
arg = arguments.parse()

output_units = {'radius': arg.radius_classes, 'distance':arg.distance_classes,
                'omega':arg.omega_classes, 'sigma':arg.sigma_classes}

deep_learning_parameter = {'algorithm': arg.algorithm, 'path': path_project, 'train': arg.train,'validation': arg.validation, 'estimation': arg.estimation,
                        'morphology': arg.morphology, 'distribution': arg.distr, 'beta': arg.beta, 'run':arg.run, 'check': arg.check, 'loss':arg.loss, 'informed': arg.informed} #}
    

# initialize algorithm, load weights, compile
algorithm = main.intialize_algorithm(algorithm=arg.algorithm, input_shape=(260, 220, 1), parameter=deep_learning_parameter, output_units=output_units)
model = algorithm.load_weights_from_path(algorithm.model, path=model_to_load)
model.compile(optimizer=optimizers.Adam(), loss=losses.MeanAbsoluteError(), metrics=[metrics.MeanSquaredError(), metrics.MeanSquaredError()])

DetectorClass = getattr(detector, arg.detector)
detector = DetectorClass(mask=(path_project + '/base/masks/' + arg.maskfile))

experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 
                                                                        'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x),
                                                                        'sample_detector_distance': arg.distance},
                                                                        detector=detector, experiment_maskfile=arg.experiment_maskfile)

data_augmentation = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detector, beta=arg.beta if arg.beta>0 else None)

path_experiment_data = {'path': '/home/danshach/pot_store/beegfs.migration/simple_net/experiment/sputter/',
                            'folders': ['sputter_500K'],
                            'target_value_files': ['tar_500K.csv']}
    
path_real_data = {'path': '/home/danshach/pot_store/beegfs.migration/simple_net/experiment/sputter/sputter_500K/'}

experiment = experiment_dataloader.Experiment(data_path=path_experiment_data, fast_exp=False, test=False)
real = real_dataloader.Experiment(data_path=path_real_data, fast_real=False)
        # fit experiment images

experiment_images, experiment_target_values = data_augmentation.fit_experiment(experiment=experiment)
#images, files = data_augmentation.fit_real(real=real, sample=None)
        # initialize algorithm
        #algorithm = intialize_algorithm(algorithm=arg.algorithm, input_shape=(real_images[0].shape + (1,)), parameter=deep_learning_parameter)
        # predict parameter


#test_random_data()
test_randomized_exp_data(experiment_images, experiment_target_values)
#test_real_data(images=images)
