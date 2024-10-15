# courtesy of E. Almamedov with modifications for our purposes

import arguments
from datetime import date
from base import simulation_dataloader, experiment_dataloader, real_dataloader, detector, setup, data_augmentation, utilities
from deep_learning import MLPnet, CNNnet
from base.simulation_dataloader import training_interval


def intialize_algorithm(algorithm, input_shape, parameter, output_units):
    if algorithm == 'MLP':
        return MLPnet.MLP2(input_shape, parameter, output_units)
    elif algorithm == 'CNN':
        return CNNnet.CNN2(input_shape, parameter, output_units)

if __name__ == '__main__':

    # parse arguments
    arg = arguments.parse()

    # set paths
    path_project = '/home/danshach/pot_store/gpfs_scratch/Xalantir' # necessary when working on the cluster

    path_simulation = '/home/danshach/pot_store/gpfs_scratch/Xalantir/data/simulation/'

    path_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/simple_net/experiment/sputter/',
                            'folders': ['sputter_100K', 'sputter_300K', 'sputter_400K', 'sputter_500K'],
                            'target_value_files': ['tar_100K.csv', 'tar_300K.csv', 'tar_400K.csv', 'tar_500K.csv']}
    
    path_real_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/simple_net/experiment/sputter/sputter_500K/'}
    # path_real_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/experiment/unlabeled/'}

    # path_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/sputter/',
    #                         'folders': ['sputter_100K', 'sputter_300K', 'sputter_400K', 'sputter_500K'],
    #                         'target_value_files': ['tar_100K.csv', 'tar_300K.csv', 'tar_400K.csv', 'tar_00K.csv']}

    # path_unlabeled_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/unlabeled/'}
    path_other_validation = '/home/danshach/pot_store/beegfs_scratch/dependencies/validation_data'
    path_other_validation_data = {'path': path_other_validation, 'files': ['validation_ausi_400K.npy'], 'target_value_files': ['validation_ausi_400K_target_values.csv']}

    # hdf5 databases with simulated form and structure factors
    if arg.sim_source=='factors_h5':
        path_simulation_data = {'path': path_simulation, 'files': ['database_1.h5', 'database_2.h5']}
    # hdf5 databases with simulated GISAXS patterns
    if arg.sim_source=='ready_h5':
        path_simulation_data = {'path': path_simulation, 'files': ['database_ready.h5']}
    # databases on files with simulated form and structure factors ### faster loading
    if arg.sim_source=='factors_file':
        path_simulation_data = {'path': path_simulation, 'files':[{'image_files': ['isGISAXS_database_sim.npz'],
                                                         'target_value_files': ['structure_factor_target_values.csv', 'form_factor_target_values.csv']}]}

    # initialize detector
    DetectorClass = getattr(detector, arg.detector)
    detector = DetectorClass(mask=(path_project + '/base/masks/' + arg.maskfile))

    # initialize experiment setup
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 
                                                                            'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x),
                                                                            'sample_detector_distance': arg.distance},
                                                                            detector=detector, experiment_maskfile=arg.experiment_maskfile)\
    # initialize data augmentation instance
    data_augmentation = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detector)

    # dimension of output unit for the neural network
    output_units = {'radius': arg.radius_classes, 'distance':arg.distance_classes,
                 'omega':arg.omega_classes, 'sigma':arg.sigma_classes}

 
    if arg.check:
        chosen_intervals ={
            'radius': training_interval['radius']['test'],
            'distance':  training_interval['distance']['test'],
            'sigma_radius': training_interval['sigma_radius']['test'],
            'omega_distance':  training_interval['omega_distance']['test']}
    else:
        chosen_intervals ={
            'radius': training_interval['radius']['costume'],
            'distance':  training_interval['distance']['costume'],
            'sigma_radius': training_interval['sigma_radius']['test'],
            'omega_distance':  training_interval['omega_distance']['costume']}   

    # constrains for including simulations
    constrains = {'fast_sim': arg.fast_sim, 'samples': None, 'check': arg.check}

    # create simulation object with constrains
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, sim_source=arg.sim_source,
                                                  constrains=constrains, intervals=chosen_intervals)
    #utilities.record_simulation_parameters
    if arg.test:
        # create experiment object
        experiment = experiment_dataloader.Experiment(data_path=path_experiment_data, fast_exp=arg.fast_exp, test=arg.check)
        # fit experiment images
        if arg.fast_exp:
            # experiment images are already extracted from cbf files
            experiment_images, experiment_target_values = data_augmentation.fit_experiment_ready(experiment=experiment)
        else:
            experiment_images, experiment_target_values = data_augmentation.fit_experiment(experiment=experiment)

    # fit simulation images
    if arg.fast_sim:
        # simulation images are already extracted from database files
        simulation_images, simulation_target_values = data_augmentation.fit_simulation_ready(simulation=simulation)
    else:
        simulation_images, simulation_target_values = data_augmentation.fit_simulation(simulation=simulation)

    # set validation_data
    if arg.validation == 'exp':
        validation_data = experiment.get_type('400K', experiment_images, experiment_target_values, sample=None)
    elif arg.validation == 'exp_reduced':
        validation_data = experiment.get_type('400K', experiment_images, experiment_target_values, sample=25)
    # elif arg.validation == 'sim':
    #     validation_data = 'sim'
    # elif arg.validation == 'other':
    #     validation_data = path_other_validation_data
    # else:
    #     validation_data = None

    deep_learning_parameter = {'algorithm': arg.algorithm, 'path': path_project, 'validation': arg.validation, 'estimation': arg.estimation,
                               'morphology': arg.morphology, 'distribution': arg.distr, 'beta': arg.beta, 'run':arg.run, 'check': arg.check, 'loss':arg.loss}
    algorithm = intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter, output_units=output_units)
    if 'exp' in arg.validation:
        algorithm.train_on_simulations_validate_with_experiment(simulation_images, simulation_target_values, validation_data)
    else:
        algorithm.train_on_simulations_validate_with_simulations(simulation_images, simulation_target_values)
    
    if arg.test:
        algorithm.test_on_experiment_in_batch(experiment_images, experiment_target_values)
        # algorithm.test_on_simulations()
        algorithm.estimate_model(thresh=2)

    if arg.real:
        # create real object
        real = real_dataloader.Experiment(data_path=path_real_experiment_data, fast_real=arg.fast_real)
        # fit experiment images
        real_images, files = data_augmentation.fit_real(real=real)
        # initialize algorithm
        #algorithm = intialize_algorithm(algorithm=arg.algorithm, input_shape=(real_images[0].shape + (1,)), parameter=deep_learning_parameter)
        # predict parameter
        algorithm.test_on_real(images=real_images, files=files)

    utilities.record_run_parameters(run_path=algorithm.model_path, morphology=algorithm.keys, training_scope=chosen_intervals, 
                                    augmentation=data_augmentation, model=algorithm.TYPE, labels=algorithm.label_coder.mode, 
                                    validation=arg.validation, estimation=arg.estimation, n_training=len(simulation_images), 
                                    n_batchs_and_epochs=arg.run, output_units=output_units, loss=arg.loss, learning_rate=algorithm.model.optimizer.learning_rate,
                                    decay=algorithm.model.optimizer.weight_decay)