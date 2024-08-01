# courtesy of E. Almamedov with modifications for our purposes

import arguments
from datetime import date
from base import simulation_dataloader, experiment_dataloader, real_dataloader, detector, setup, data_augmentation
from deep_learning import MLPnet, CNNnet


def initialize_algorithm(algorithm, input_shape, parameter, output_units):
    if algorithm == 'MLP':
        return MLPnet.MLP2(input_shape, parameter, output_units)
    #elif algorithm == 'CNN'

if __name__ == '__main__':

    # parse arguments
    arg = arguments.parse()

    # set paths
    path_project = '/home/danshach/pot_store/beegfs_scratch/Xalantir' # necessary when working on the cluster
    path_simulation = '/home/danshach/pot_store/beegfs_scratch/dependencies/simulation'
    path_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/sputter/',
                            'folders': ['sputter_100K', 'sputter_300K', 'sputter_400K', 'sputter_500K'],
                            'target_value_files': ['tar_100K.csv', 'tar_300K.csv', 'tar_400K.csv', 'tar_00K.csv']}#, 'first_frame': [190, 190]}
    path_unlabeled_experiment_data = {'path': '/home/danshach/pot_store/beegfs_scratch/unlabeled/'}
    path_other_validation = '/home/danshach/pot_store/beegfs_scratch/dependencies/validation_data'
    path_other_validation_data = {'path': path_other_validation, 'files': ['validation_ausi_400K.npy'] 'target_value_files': ['validation_ausi_400K_target_values.csv']}

    # hdf5 databases with simulated form and structure factors
    if arg.sim_source=='factors_h5':
        path_simulation_data = {'path': path_simulation, 'files': ['database_1.h5', 'database_2.h5']}
    # hdf5 databases with simulated GISAXS patterns
    if arg.sim_source=='ready_h5':
        path_simulation_data = {'path': path_simulation, 'files': ['database_ready.h5']}
    # databases on files with simulated form and structure factors
    if arg.sim_source=='factors_file':
        path_simulation_data = {'path': path_simulation, 'files':[{'image_files': ['isGISAXS_database_sim.npz'],
                                                         'target_value_files': ['structure_factor_target_values.csv', 'form_factor_target_values.csv']}]}

    # initialize detector
    DetectorClass = getattr(detector, arg.detector)
    detector = detector.DetectorClass(mask=(path_project + 'base/masks/' + arg.maskfile))

    # initialize experiment setup
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 'incidence_angle': arg.incidence_angle,
                                                                            'direct_beam_position': (arg.db_y, arg.db_x), 'sample_detector_distance': arg.distance},
                                                                            detector=detector, experiment_maskfile=arg.experiment_maskfile)
    # initialize data augmentation instance
    data_augmentation = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detector)

    # dimension of output unit for the neural network
    output_units = {'radius': arg.radius_classes, 'distance':arg.distance_classes,
                 'omega':arg.omega_classes, 'sigma':arg.sigma_classe}

    # intervals for trainning scope
    intervals =
    {
        'radius':
                {'all': {'start': 0.5, 'stop': 7.6, 'step': 0.1}, 'low': {'start': 0.5, 'stop': 4.6, 'step': 0.1}, 'medium': {'start': 1.5, 'stop': 5.6, 'step': 0.1},
                 'high': {'start': 2.5, 'stop': 7.6, 'step': 0.1}, 'stepsize': {'start': 1.0, 'stop': 8.0, 'step': 1.0}, 'costume':{'start': 1.2, 'stop': 7.6, 'step': 0.1}},
        'distance':
                {'all': {'start': 0.5, 'stop': 7.6, 'step': 0.1}, 'low': {'start': 0.5, 'stop': 4.6, 'step': 0.1}, 'medium': {'start': 1.5, 'stop': 5.6, 'step': 0.1},
                 'high': {'start': 2.5, 'stop': 7.6, 'step': 0.1}, 'stepsize': {'start': 1.0, 'stop': 8.0, 'step': 1.0}, 'costume':{'start': 1.2, 'stop': 7.6, 'step': 0.1}},
        'sigma_radius':
                {'all': {'start': 0.1, 'stop': 0.51, 'step': 0.01}, 'low': {'start': 0.1, 'stop': 0.25, 'step': 0.01}, 'medium': {'start': 0.18, 'stop': 0.33, 'step': 0.01},
                 'high': {'start': 0.33, 'stop': 0.51, 'step': 0.01}, 'stepsize': {'start': 0.1, 'stop': 0.51, 'step': 0.1}, 'costume': {'start': 0.15, 'stop': 0.31, 'step': 0.05}},
        'omega_distance':
                {'all': {'start': 0.1, 'stop': 0.51, 'step': 0.01}, 'low': {'start': 0.1, 'stop': 0.25, 'step': 0.01}, 'medium': {'start': 0.18, 'stop': 0.33, 'step': 0.01},
                 'high': {'start': 0.33, 'stop': 0.51, 'step': 0.01}, 'stepsize': {'start': 0.1, 'stop': 0.51, 'step': 0.1}, 'costume': {'start': 0.18, 'stop': 0.28, 'step': 0.01}}

    }
    chosen_intervals ={
        'radius': intervals['radius']['all'],
        'distance':  intervals['distance']['medium'],
        'sigma_radius': intervals['sigma_radius']['costume'],
        'omega_distance':  intervals['omega_distance']['costume']}

    # constrains for including simulations
    constrains = {'fast_sim': arg.fast_sim, 'samples': None, 'check': arg.check}

    # create simulation object with constrains
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, sim_source=arg.sim_source,
                                                  experiment_setup=experiment_setup, detector=detector,
                                                  constrains=constrains, intervals=chosen_intervals)

    if arg.test:
        # create experiment object
        experiment = experiment_dataloader.Experiment(data_path=path_experiment_data, fast_exp=arg.fast_exp)
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
    elif arg.validation == 'sim':
        validation_data = 'sim'
    elif arg.validation == 'other':
        validation_data = path_other_validation_data
    else:
        validation_data = None

    deep_learning_parameter = {'algorithm': arg.algorithm, 'path': path_project, 'validation': arg.validation,
                               'estimation': arg.estimation, 'morphology': arg.morphology, 'distribution': arg.distr}

    algorithm = intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter, output_units=output_units)
    algorithm.train_on_simulations(simulation_images, simulation_target_values)
    # algorithm.train_on_experiments()
    algorithm.test_on_experiment()
    # algorithm.test_on_simulations()
    algorithm.estimate_model()

    if arg.real:
        # create real object
        real = real_dataloader.Experiment(data_path=path_unlabeled_experiment_data, fast_real=arg.fast_real)
        #algorithm.test_real()
        # initialize algorithm
        algorithm = intialize_algorithm(algorithm=arg.algorithm, input_shape=(real_images[0].shape + (1,)), parameter=deep_learning_parameter)
        # predict parameter
        algorithm.real(images=real_images, files=files)

    # if arg.estimation == 'naive':
    #     for i in range(MAX_ROUNDS)
    #         algorithm = algorithm.initialize_algorithm(arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), deep_learning_parameters, , )
    #         algorithm.train_on_simulations(simulation_images, simulation_target_values, validation_data)
    #         algorithm.test


    # maybe just one execution each time!!! arg.morphology ::= 'radius'| 'distance' | 'all' | 'radius_sigma' | 'distance_omega'

    #if arg.network != 'none'
    #    simulation_labels_radius = label_coder.create_labels(key='all', target_values=simulation_target_values)
    #    experiment_labels_radius = label_coder.create_labels(key='all', target_values=experiment_target_values) # comment if real
    # else:
    #    if arg.nn_radius != 'none':
            #simulation_labels_radius = label_coder.create_labels(key='radius', target_values=simulation_target_values.radius)
            #experiment_labels_radius = label_coder.create_labels(key='radius', target_values=experiment_target_values.distance) # comment if real
            # initialize algorithm
