import glob

import numpy
import pandas
import math
import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import arguments, main
from base import utilities, data_augmentation, detector, experiment_dataloader, simulation_dataloader, real_dataloader, setup


from statistics import mean
PATH = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/results/'

            # draw training history plot
            #self.plot_model_accurarcy_and_loss(history=history.history, model_path=self.model_path + "/round_" + str(i))
            
def visualize_data(dataset):
    # Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface
    # for drawing attractive and informative statistical graphics.
    sns.pairplot(dataset[['distance', 'omega_distance', 'radius', 'sigma_radius']], diag_kind='kde')
    plt.show()


def plot_error_metrics(model, filename):
    """
    Plot saved error metrics from CSV file.

    :param model: name of the model (:type str)
    :param filename: name of the CSV file (:type str)
    """
    mse = False
    mae = True
    datatype = '.png'
    # read csv
    history = pandas.read_csv(PATH + model + '/' + filename)

    if mae:
        # distance
        plt.plot(history.epoch, history.distance_mean_absolute_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_distance_mean_absolute_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MAE mittlerer Abstand $D$ [nm]')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_distance_mae' + datatype, bbox_inches='tight')
        plt.show()

        # omega distance
        plt.plot(history.epoch, history.omega_distance_mean_absolute_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_omega_distance_mean_absolute_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MAE Verteilung mittlerer Abstand $\frac{\omega}{D}$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_omega_distance_mae' + datatype, bbox_inches='tight')
        plt.show()

        # radius
        plt.plot(history.epoch, history.radius_mean_absolute_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_radius_mean_absolute_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MAE mittlerer Radius $R$ [nm]')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_radius_mae' + datatype, bbox_inches='tight')
        plt.show()

        # sigma radius
        plt.plot(history.epoch, history.sigma_radius_mean_absolute_error, color='blue', linewidth=2.5, linestyle='-',label='Training')
        plt.plot(history.epoch, history.val_sigma_radius_mean_absolute_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MAE Abweichung mittlerer Radius $\frac{\sigma}{R}$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_sigma_radius_mae' + datatype, bbox_inches='tight')
        plt.show()

    if mse:
        # distance
        plt.plot(history.epoch, history.distance_mean_squared_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_distance_mean_squared_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MSE mittlerer Abstand $D$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_distance_mse' + datatype, bbox_inches='tight')
        plt.show()

        # omega distance
        plt.plot(history.epoch, history.omega_distance_mean_squared_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_omega_distance_mean_squared_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MSE Verteilung mittlerer Abstand $\frac{\omega}{D}$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_omega_distance_mse' + datatype, bbox_inches='tight')
        plt.show()

        # radius
        plt.plot(history.epoch, history.radius_mean_squared_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_radius_mean_squared_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MSE mittlerer Radius $R$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_radius_mse' + datatype, bbox_inches='tight')
        plt.show()

        # sigma radius
        plt.plot(history.epoch, history.sigma_radius_mean_squared_error, color='blue', linewidth=2.5, linestyle='-', label='Training')
        plt.plot(history.epoch, history.val_sigma_radius_mean_squared_error, color="red", linewidth=2.5, linestyle="-", label='Test')
        plt.title(model)
        plt.xlabel(r'Epochen')
        plt.ylabel(r'MSE Abweichung mittlerer Radius $\frac{\sigma}{R}$')
        plt.legend(loc='upper right')
        plt.xlim(0, len(history.epoch))
        plt.savefig(PATH + model + '/' + model + '_sigma_radius_mse' + datatype, bbox_inches='tight')
        plt.show()



def plot_error_metrics_experiment(model, filename):
    """
    Plot saved error metrics from CSV file.

    :param model: name of the model (:type str)
    :param filename: name of the CSV file (:type str)
    """
    datatype = '.png'
    # read csv
    experiment = pandas.read_csv(PATH + 'SP400K.csv')
    # convert 'Frame' from float64 to int64
    experiment['Frame'] = experiment['Frame'].astype(int)
    # keep relevant columns
    experiment = experiment[['Frame', 'Distance', 'Radius']]
    # drop rows with NaN entries
    experiment = experiment.dropna()

    experiment_prediction = pandas.read_csv(PATH + model + '/' + filename, index_col=0)
    experiment_prediction = experiment_prediction[experiment_prediction.meausurement == 'si_au3w_125c_80']
    # distance
    ae_distance = abs(experiment_prediction.distance_pred - experiment_prediction.distance_true)
    mae_distance = abs(experiment_prediction.distance_pred - experiment_prediction.distance_true).mean()
    plt.plot(experiment_prediction.frame, experiment_prediction.distance_true, color='blue', linewidth=1.5, linestyle='-', label='Berechnung')
    plt.plot(experiment_prediction.frame, experiment_prediction.distance_pred, color="red", linewidth=1.5, linestyle='-', label='Prädiktion')
    #plt.plot(experiment_prediction.frame, ae_distance, color="orange", linewidth=1.5, linestyle='-', label='AE')
    #plt.axhline(y=mae_distance, xmin=0.0, xmax=6000.0, color='orange', linestyle='--', label='MAE')
    #plt.axhline(y=15, xmin=0.0, xmax=1710.0, color='green', linestyle='--', label='Simulationsbereich')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'mittlere Distanz $D$ [nm]')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model + '/' + model + '_distance_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

    ae_radius = abs(experiment_prediction.radius_pred - experiment_prediction.radius_true)
    mae_radius = abs(experiment_prediction.radius_pred - experiment_prediction.radius_true).mean()
    plt.plot(experiment_prediction.frame, experiment_prediction.radius_true, color='blue', linewidth=1.5, linestyle='-', label='Berechnung')
    plt.plot(experiment_prediction.frame, experiment_prediction.radius_pred, color="red", linewidth=1.5, linestyle='-', label='Prädiktion')
    #plt.plot(experiment_prediction.frame, ae_radius, color="orange", linewidth=1.5, linestyle='-', label='AE')
    #plt.axhline(y=mae_radius, xmin=0.0, xmax=6000.0, color='orange', linestyle='--', label='MAE')
    #plt.axhline(y=7.5, xmin=0.0, xmax=1710.0, color='green', linestyle='--', label='Simulationsbereich')
    #plt.axvline(x=1710, ymin=0.0, ymax=7.5, color='green', linestyle='--')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'mittlerer Radius $R$ [nm]')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model + '/' + model + '_radius_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

    plt.plot(experiment_prediction.frame, experiment_prediction.omega_distance_pred, color='blue', linewidth=1.5, linestyle='-', label=r'$\frac{\omega}{D}$')
    plt.plot(experiment_prediction.frame, experiment_prediction.sigma_radius_pred, color="red", linewidth=1.5, linestyle='-', label=r'$\frac{\sigma}{R}$')
    #plt.axhline(y=0.5, xmin=0.0, xmax=1710.0, color='green', linestyle='--', label='Simulationsbereich')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'Variation')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model + '/' + model + '_omgea_distance_sigma_radius_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

def plot_error_multi_metrics_experiment(model, filename):
    """
    Plot saved error metrics from CSV file.

    :param model: name of the model (:type str)
    :param filename: name of the CSV file (:type str)
    """
    datatype = '.png'
    # read csv
    experiment = pandas.read_csv(PATH + 'SP400K.csv')
    # convert 'Frame' from float64 to int64
    experiment['Frame'] = experiment['Frame'].astype(int)
    # keep relevant columns
    experiment = experiment[['Frame', 'Distance', 'Radius']]
    # drop rows with NaN entries
    experiment = experiment.dropna()

    experiment_prediction_1 = pandas.read_csv(PATH + model[0] + '/' + filename[0], index_col=0)
    experiment_prediction_1 = experiment_prediction_1[experiment_prediction_1.meausurement == 'si_au3w_125c_80']

    experiment_prediction_2 = pandas.read_csv(PATH + model[1] + '/' + filename[1], index_col=0)
    experiment_prediction_2 = experiment_prediction_2[experiment_prediction_2.meausurement == 'si_au3w_125c_80']

    experiment_prediction_3 = pandas.read_csv(PATH + model[2] + '/' + filename[1], index_col=0)
    experiment_prediction_3 = experiment_prediction_3[experiment_prediction_3.meausurement == 'si_au3w_125c_80']

    # distance
    plt.plot(experiment_prediction_1.frame, experiment_prediction_1.distance_true, color='blue', linewidth=1.5, linestyle='-', label='Berechnung')
    plt.plot(experiment_prediction_1.frame, experiment_prediction_1.distance_pred, color="red", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (50 Epochen)')
    plt.plot(experiment_prediction_2.frame, experiment_prediction_2.distance_pred, color="green", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (100 Epochen)')
    plt.plot(experiment_prediction_3.frame, experiment_prediction_3.distance_pred, color="orange", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (100 Epochen) ohne Summation')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'mittlere Distanz $D$ [nm]')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model[1] + '/' + model[1] + '_distance_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

    plt.plot(experiment_prediction_1.frame, experiment_prediction_1.radius_true, color='blue', linewidth=1.5, linestyle='-', label='Berechnung')
    plt.plot(experiment_prediction_1.frame, experiment_prediction_1.radius_pred, color="red", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (50 Epochen)')
    plt.plot(experiment_prediction_2.frame, experiment_prediction_2.radius_pred, color="green", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (100 Epochen)')
    plt.plot(experiment_prediction_3.frame, experiment_prediction_2.radius_pred, color="orange", linewidth=1.5, linestyle='-', label='Prädiktion DenseNet169 (100 Epochen) ohne Summation')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'mittlerer Radius $R$ [nm]')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model[1] + '/' + model[1] + '_radius_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

# TODO: evaluate histogramm of experimental data

def plot_absolute_error_simulation_histogram(model, filename):
    # define datatype
    datatype = '.png'
    # read csv
    data = pandas.read_csv(PATH + model + '/' + filename)

    distance_absolute_error = numpy.abs(data.distance_pred - data.distance_true)
    plt.hist(distance_absolute_error, weights=[1/len(distance_absolute_error)] * len(distance_absolute_error), color="blue", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.title(model)
    plt.xlabel(r'absoluter Fehler mittlerer Abstand $D$ [nm]')
    plt.ylabel('prozentuale Häufigkeiten')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + model + '/' + model + '_distance_prediction_simulation' + datatype, bbox_inches='tight')
    plt.show()

    omega_distance_absolute_error = numpy.abs(data.omega_distance_pred - data.omega_distance_true)
    plt.hist(omega_distance_absolute_error, weights=[1/len(omega_distance_absolute_error)] * len(omega_distance_absolute_error), color="blue", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.title(model)
    plt.xlabel(r'absoluter Fehler Verteilung mittlerer Abstand $\frac{\omega}{D}$')
    plt.ylabel('prozentuale Häufigkeiten')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + model + '/' + model + '_omega_distance_prediction_simulation'+ datatype, bbox_inches='tight')
    plt.show()

    radius_absolute_error = numpy.abs(data.radius_pred - data.radius_true)
    plt.hist(radius_absolute_error, weights=[1/len(radius_absolute_error)] * len(radius_absolute_error), color="blue", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.title(model)
    plt.xlabel(r'absoluter Fehler mittlerer Radius $R$ [nm]')
    plt.ylabel('prozentuale Häufigkeiten')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + model + '/' + model + '_radius_prediction_simulation' + datatype, bbox_inches='tight')
    plt.show()

    sigma_radius_absolute_error = numpy.abs(data.sigma_radius_pred - data.sigma_radius_true)
    plt.hist(sigma_radius_absolute_error, weights=[1/len(sigma_radius_absolute_error)] * len(sigma_radius_absolute_error), color="blue", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.title(model)
    plt.xlabel(r'absoluter Fehler Abweichung mittlerer Radius $\frac{\sigma}{R}$')
    plt.ylabel('prozentuale Häufigkeiten')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + model + '/' + model + '_sigma_radius_prediction_simulation' + datatype, bbox_inches='tight')
    plt.show()

def calculate_mae(model, filename):
    # read csv
    data = pandas.read_csv(PATH + model + '/' + filename)

    mae_distance = abs((data.distance_pred - data.distance_true).mean())
    mae_omega_distance = abs((data.omega_distance_pred - data.omega_distance_true).mean())
    mae_radius = abs((data.radius_pred - data.radius_true).mean())
    mae_sigma_radius = abs((data.sigma_radius_pred - data.sigma_radius_true).mean())

    print('MAE of', model)
    print('distance: ', mae_distance)
    print('omega_distance: ', mae_omega_distance)
    print('radius: ', mae_radius)
    print('sigma_radius: ', mae_sigma_radius)

def fit_simulation(da, simulation, images, target_values):
    """
    Adapt the simulated scattering images to the experimental scattering images
    :param: simulation  the simulation class.
    :return: list of fitted images (:rtype: list of ndarray)
    """
    fitted_images = []
    # start fitting
    n1, n2, two_theta_f_min, two_theta_f_max, alpha_f_min, alpha_f_max = simulation.get_grid_parameter(key='1')
    two_theta_f, alpha_f, q_y, q_z = da.convert_from_cartesian_to_reciprocal_space()
    two_theta_f_crop_index = int(round((two_theta_f.max()/(two_theta_f_max - two_theta_f_min)) * n2))
    alpha_f_crop_index = int(round((alpha_f.max()/(alpha_f_max - alpha_f_min)) * (n1-1)))
    # crop masks
    detector_mask = da.crop_detector_mask()
    experiment_mask = da.crop_experiment_mask()
    shape_to_bin = (int(math.ceil(0.5 * detector_mask.shape[0])), int(math.ceil(0.5 * detector_mask.shape[1])))
    detector_mask = da.bin_mask(detector_mask, bin_to_shape=shape_to_bin)
    experiment_mask = da.bin_mask(experiment_mask, bin_to_shape=shape_to_bin)
    for image in tqdm.tqdm(images, desc="step: fit simulation data", total=len(images), mininterval=120, miniters=1000):
        # crop
        image = da.crop_simulation(image=image, y=alpha_f_crop_index, x=two_theta_f_crop_index)
        # bin simulation image to experiment image
        image = da.bin_float(to_bin=image, bin_to_shape=shape_to_bin)
         # noise
        image = da.add_gaussian_noise(image=image)
        image = da.add_poisson_shot_noise(image=image)
        # mask
        image = da.mask_image(image=image, mask=experiment_mask)
        # noise
        image = da.add_salt_and_pepper_noise(image=image)
        # mask
        image = da.mask_image(image=image, mask=detector_mask)
        # standardize
        #image = da.standardize(image=image, factor=10)
        image = da.intensity_scale(image=image)
        # append images
        fitted_images.append(image)
    return fitted_images, target_values

def fit_experiment(da, experiment_images, experiment_target_values):
        """
        Adapt the experimental scattering images
        :return: list of fitted images (:rtype: list of ndarray)
        """
        # start fitting
        fitted_experiment_images = []
        # shape to bin
        shape_cropped_experiment = da.crop_experiment(image=experiment_images[0])
        shape_to_bin = (int(math.ceil(0.5 * shape_cropped_experiment.shape[0])), int(math.ceil(0.5 * shape_cropped_experiment.shape[1])))
        for image in tqdm.tqdm(experiment_images, desc="step: fit experiment data", total=len(experiment_images), mininterval=60, miniters=100):
            # crop
            image = da.crop_experiment(image=image)
            # bin
            image = da.bin_float(to_bin=image, bin_to_shape=shape_to_bin)
            # standardize
            #image = da.standardize(image)
            image = da.intensity_scale(image)
            fitted_experiment_images.append(image)
        return fitted_experiment_images, experiment_target_values

def create_attention_maps():
    path_project = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/'
    path_simulation_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/simulation/', 'files': ['database_1.h5']}
    path_experiment_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/experiment/', 'folders': ['si_au3w_125c_80', 'si_au3w_225c_81'], 'target_value_files': ['SP400K.csv', 'SP500K.csv'], 'first_frame': [190, 190]}
    # parser
    arg = arguments.parse()
    # initialize detector
    detec = detector.Pilatus1M(mask=(path_project + 'base/masks/' + arg.maskfile))
    # initialize experiment setup
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x), 'sample_detector_distance': arg.distance}, detector=detector, experiment_maskfile=arg.experiment_maskfile)
    # initialize data augmentation
    da = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detec)
    deep_learning_parameter = {'algorithm': arg.algorithm, 'optimizer': arg.optimizer, 'loss': arg.loss, 'metrics': arg.metrics, 'batch_size': arg.batch_size, 'epochs': arg.epochs, 'test_size': arg.train_test_split, 'run': arg.run, 'path': path_project, 'extend': arg.extend}
    # create simulation object
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, samples=arg.samples)
    # fit simulation images
    simulation_images, simulation_target_values = simulation.load_data()
    simulation_images, simulation_target_values = fit_simulation(da=da, simulation=simulation, images=simulation_images[0:5], target_values=simulation_target_values[0:5])
    # initialize algorithm
    algorithm = main.intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter)
    # load weights
    model = algorithm.load_weights(model=algorithm.model)
    # compile model
    model.compile(optimizer=deep_learning_parameter['optimizer'], loss=deep_learning_parameter['loss'], metrics=deep_learning_parameter['metrics'])
    utilities.plot_saliency_map(model=model, image=simulation_images, target_value=simulation_target_values.distance, variable='distance', image_to_pred=algorithm.create_input_tensor(images=simulation_images))

def compare_experiment_simulation():
    path_project = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/'
    path_simulation_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/simulation/', 'files': ['database_1.h5', 'database_2.h5']}
    path_experiment_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/experiment/','folders': ['si_au3w_125c_80'],  # , 'si_au3w_225c_81'],si_au3w_125c_80
                            'target_value_files': ['SP400K.csv'],
                            'first_frame': [190]}  #, 'SP500K.csv']} # parser
    arg = arguments.parse()
    # initialize detector
    detec = detector.Pilatus1M(mask=(path_project + 'base/masks/' + arg.maskfile))
    # initialize experiment setup
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x), 'sample_detector_distance': arg.distance}, detector=detector, experiment_maskfile=arg.experiment_maskfile)
    # initialize data augmentation
    da = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detec)
    deep_learning_parameter = {'algorithm': arg.algorithm, 'optimizer': arg.optimizer, 'loss': arg.loss, 'metrics': arg.metrics, 'batch_size': arg.batch_size, 'epochs': arg.epochs, 'test_size': arg.train_test_split, 'run': arg.run, 'path': path_project, 'extend': arg.extend}
    # simulation
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, samples=arg.samples)
    # fit simulation images
    simulation_images, simulation_target_values = simulation.load_data()
    sim_images = simulation_images
    sim_target_values = simulation_target_values
    simulation_images, simulation_target_values = fit_simulation(da=da, simulation=simulation, images=simulation_images[0:5], target_values=simulation_target_values[0:5])
    # initialize algorithm
    algorithm = main.intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter)

    # load weights and compile
    model = algorithm.load_weights(algorithm.model)
    model.compile(optimizer=deep_learning_parameter['optimizer'], loss=deep_learning_parameter['loss'], metrics=deep_learning_parameter['metrics'])

    # real, load, fit, predict
    real_experiment = real_dataloader.Experiment(data_path='/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/')
    real_images, real_files = real_experiment.get_images()
    real_images, real_files = da.fit_real(real_experiment)
    real_images_to_predict = algorithm.create_input_tensor(real_images)
    real_prediction = model.predict(real_images_to_predict)

    # experiment, load, fit, predict
    experiment = experiment_dataloader.Experiment(data_path=path_experiment_data)
    experiment_images, experiment_target_values = experiment.load_data()
    experiment_target_values = experiment_target_values.query('Frame == 200  or Frame == 800 or Frame == 1600 or Frame == 3200 or Frame == 5000')
    experiment_images_to_predict = []
    for index, _ in experiment_target_values.iterrows():
        experiment_images_to_predict.append(experiment_images[index])
    experiment_images_to_predict, experiment_target_values = fit_experiment(da, experiment_images_to_predict, experiment_target_values)
    experiment_images = experiment_images_to_predict
    experiment_images_to_predict = algorithm.create_input_tensor(experiment_images_to_predict)
    experiment_prediction = model.predict(experiment_images_to_predict)
    utilities.draw_plot(experiment_images[0], title='200_si_au3w_125c_80', z_limits=[0.00001, 100000.0])
    utilities.draw_plot(experiment_images[1], title='900_si_au3w_125c_80', z_limits=[0.00001, 100000.0])
    utilities.draw_plot(experiment_images[2], title='1600_si_au3w_125c_80', z_limits=[0.00001, 100000.0])
    utilities.draw_plot(experiment_images[3], title='3200_si_au3w_125c_80', z_limits=[0.00001, 100000.0])
    utilities.draw_plot(experiment_images[4], title='5000_si_au3w_125c_80', z_limits=[0.00001, 100000.0])

    print("STOP")

if __name__ == '__main__':
    done = True
    if not done:
        # evaluate densenet121_3550_64
        plot_error_metrics(model='densenet121_3550_64', filename='densenet121_history3550_64.csv')
        plot_absolute_error_simulation_histogram(model='densenet121_3550_64', filename='densenet121_prediction_simulation3550_64.csv')
        calculate_mae(model='densenet121_3550_64', filename='densenet121_prediction_simulation3550_64.csv')

        # evaluate densenet169_3550_64
        plot_error_metrics(model='densenet169_3550_64', filename='densenet169_history3550_64.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64', filename='densenet169_prediction_simulation3550_64.csv')
        calculate_mae(model='densenet169_3550_64', filename='densenet169_prediction_simulation3550_64.csv')

        # evaluate densenet201_3550_64
        plot_error_metrics(model='densenet201_3550_64', filename='densenet201_history3550_64.csv')
        plot_absolute_error_simulation_histogram(model='densenet201_3550_64', filename='densenet201_prediction_simulation3550_64.csv')
        calculate_mae(model='densenet201_3550_64', filename='densenet201_prediction_simulation3550_64.csv')

        # evaluate resnet50_3550_64
        plot_error_metrics(model='resnet50_3550_64', filename='resnet50_history3550_64.csv')
        plot_absolute_error_simulation_histogram(model='resnet50_3550_64', filename='resnet50_prediction_simulation3550_64.csv')
        calculate_mae(model='resnet50_3550_64', filename='resnet50_prediction_simulation3550_64.csv')

        # evaluate densenet169_3550_64_without_noise
        plot_error_metrics(model='densenet169_3550_64_without_noise', filename='densenet169_history3550_64_without_noise.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_without_noise', filename='densenet169_prediction_simulation3550_64_without_noise.csv')
        calculate_mae(model='densenet169_3550_64_without_noise', filename='densenet169_prediction_simulation3550_64_without_noise.csv')

        # evaluate densenet169_3550_64_without_gauss
        plot_error_metrics(model='densenet169_3550_64_without_gauss', filename='densenet169_history3550_64_without_gauss.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_without_gauss', filename='densenet169_prediction_simulation3550_64_without_gauss.csv')
        calculate_mae(model='densenet169_3550_64_without_gauss', filename='densenet169_prediction_simulation3550_64_without_gauss.csv')

        # evaluate densenet169_3550_64_without_poisson
        plot_error_metrics(model='densenet169_3550_64_without_poisson', filename='densenet169_history3550_64_without_poisson.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_without_poisson', filename='densenet169_prediction_simulation3550_64_without_poisson.csv')
        calculate_mae(model='densenet169_3550_64_without_poisson', filename='densenet169_prediction_simulation3550_64_without_poisson.csv')

        # evaluate densenet169_3550_64_without_poisson
        plot_error_metrics(model='densenet169_3550_64_without_salt_pepper', filename='densenet169_history3550_64_without_salt_pepper.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_without_salt_pepper',filename='densenet169_prediction_simulation3550_64_without_salt_pepper.csv')
        calculate_mae(model='densenet169_3550_64_without_salt_pepper', filename='densenet169_prediction_simulation3550_64_without_salt_pepper.csv')

        # evaluate densenet169_3550_64_without_gauss
        plot_error_metrics(model='densenet169_3550_64_without_poisson_salt_pepper', filename='densenet169_history3550_64_without_poisson_salt_pepper.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_without_poisson_salt_pepper', filename='densenet169_prediction_simulation3550_64_without_poisson_salt_pepper.csv')
        calculate_mae(model='densenet169_3550_64_without_poisson_salt_pepper', filename='densenet169_prediction_simulation3550_64_without_poisson_salt_pepper.csv')

        plot_error_metrics_experiment(model='densenet121_3550_64', filename='densenet121_prediction_experiment3550_64.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64', filename='densenet169_prediction_experiment3550_64.csv')
        plot_error_metrics_experiment(model='densenet201_3550_64', filename='densenet201_prediction_experiment3550_64.csv')
        plot_error_metrics_experiment(model='resnet50_3550_64', filename='resnet50_prediction_experiment3550_64.csv')

        plot_error_metrics_experiment(model='densenet169_3550_64_without_noise', filename='densenet169_prediction_experiment3550_64_without_noise.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_without_gauss', filename='densenet169_prediction_experiment3550_64_without_gauss.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_without_poisson', filename='densenet169_prediction_experiment3550_64_without_poisson.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_without_salt_pepper', filename='densenet169_prediction_experiment3550_64_without_salt_pepper.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_without_poisson_salt_pepper', filename='densenet169_prediction_experiment3550_64_without_poisson_salt_pepper.csv')

        # evaluate densenet169_3550_64_100
        plot_error_metrics(model='densenet169_3550_64_100', filename='densenet169_history3550_64_100.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_100', filename='densenet169_prediction_simulation3550_64_100.csv')
        calculate_mae(model='densenet169_3550_64_100', filename='densenet169_prediction_simulation3550_64_100.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_100', filename='densenet169_prediction_experiment3550_64_100.csv')

        # evaluate densenet169_3550_64_intensity
        plot_error_metrics(model='densenet169_3550_64_100_intensity', filename='densenet169_history3550_64_100_intensity.csv')
        plot_absolute_error_simulation_histogram(model='densenet169_3550_64_100_intensity', filename='densenet169_prediction_simulation3550_64_100_intensity.csv')
        calculate_mae(model='densenet169_3550_64_100_intensity', filename='densenet169_prediction_simulation3550_64_100_intensity.csv')
        plot_error_metrics_experiment(model='densenet169_3550_64_100_intensity', filename='densenet169_prediction_experiment3550_64_100_intensity.csv')


        compare_experiment_simulation()

        plot_error_multi_metrics_experiment(model=['densenet169_3550_64_intensity', 'densenet169_3550_64_100_intensity', 'densenet169_3550_64_100_intensity'], filename=['densenet169_prediction_experiment3550_64_intensity.csv', 'densenet169_prediction_experiment3550_64_100_intensity.csv', 'densenet169_prediction_experiment3550_64_100_intensity_no_sum.csv'])
    create_attention_maps()
