import glob
import numpy
import pandas
import math
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorflow.keras import optimizers, losses, metrics
import arguments, main
from base import utilities, data_augmentation, detector, experiment_dataloader, simulation_dataloader, real_dataloader, setup
from deep_learning import algorithm
from statistics import mean

# constants
PATH = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/results/'
DATATYPE = '.png'

# for latex format
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

# TODO:
# saliency plots
# activation maps

# DONE:
# loss function
# histogramms

# 33 % : 33124
# 38 % : 43681
# 42 % : 53361
# 46 % : 64009
# 50 % : 75625


# TODO: plot functions
def plot_model_training_and_validation_loss(folder, model, filename):
    # read csv
    history = pandas.read_csv(PATH + folder + '/' + filename)
    # loss
    figure = plt.figure(dpi=600)
    plt.plot(history.epoch, history.loss, color='black', linewidth=2.5, linestyle='-', label='Trainingsfehler')
    plt.plot(history.epoch, history.val_loss, color='gray', linewidth=2.5, linestyle='-', label='Validierungsfehler')
    plt.xlabel(r'Epochen')
    plt.ylabel(r'Verlustfunktion MAE')
    plt.legend(loc='upper right')
    plt.ylim(0, 8)
    plt.xlim(0, len(history.epoch)-1)
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_loss' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_prediction_simulation_histogram(folder, model, filename):
    # for simulation test data
    # read csv
    data = pandas.read_csv(PATH + folder + '/' + filename)
    # distance histogram
    distance_absolute_error = numpy.abs(data.distance_pred - data.distance_true)
    figure1 = plt.figure(figsize=(5, 4), dpi=600)
    plt.hist(distance_absolute_error, weights=[1/len(distance_absolute_error)] * len(distance_absolute_error), color="gray", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    #plt.title(model)
    plt.xlabel(r'Absoluter Fehler mittlerer Abstand $D$ [nm]')
    plt.ylabel(r'Häufigkeit [\%]')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlim(0, distance_absolute_error.max())
    plt.savefig(PATH + folder + '/' + 'eval_'  + model + '_simulation_histogram_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega distance histogram
    omega_distance_absolute_error = numpy.abs(data.omega_distance_pred - data.omega_distance_true)
    #figure2 = plt.figure(figsize=(5, 4), dpi=600)
    plt.hist(omega_distance_absolute_error, weights=[1/len(omega_distance_absolute_error)] * len(omega_distance_absolute_error), color="gray", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    #plt.title(model)
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Abstand $\omega/D$')
    plt.ylabel(r'Häufigkeit [\%]')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + folder + '/' + 'eval_'  + model + '_simulation_histogram_omega_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius histogram
    radius_absolute_error = numpy.abs(data.radius_pred - data.radius_true)
    #figure3 = plt.figure(figsize=(5, 4), dpi=600)
    plt.hist(radius_absolute_error, weights=[1/len(radius_absolute_error)] * len(radius_absolute_error), color="gray", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.xlabel(r'Absoluter Fehler mittlerer Radius $R$ [nm]')
    plt.ylabel(r'Häufigkeit [%]')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_simulation_histogram_radius' + DATATYPE, bbox_inches='tight')
    plt.show()

    # sigma radius
    sigma_radius_absolute_error = numpy.abs(data.sigma_radius_pred - data.sigma_radius_true)
    #figure4 = plt.figure(figsize=(5, 4), dpi=600)
    plt.hist(sigma_radius_absolute_error, weights=[1/len(sigma_radius_absolute_error)] * len(sigma_radius_absolute_error), color="blue", rwidth=1, bins=30, edgecolor='black', linewidth=1.2)
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Radius $\sigma/R$')
    plt.ylabel(r'Häufigkeit [%]')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_simulation_histogram_sigma_radius' + DATATYPE, bbox_inches='tight')
    plt.show()

def plot_histogramm(folders, models, filenames, labels=['DenseNet-121', 'DenseNet-169', 'DenseNet-201']):
    densenet121 = pandas.read_csv(PATH + folders[0] + '/' + filenames[0])
    densenet169 = pandas.read_csv(PATH + folders[1] + '/' + filenames[1])
    densenet201 = pandas.read_csv(PATH + folders[2] + '/' + filenames[2])

    # distance
    densenet121_distance_absolute_error = numpy.abs(densenet121.distance_pred - densenet121.distance_true)
    densenet169_distance_absolute_error = numpy.abs(densenet169.distance_pred - densenet169.distance_true)
    densenet201_distance_absolute_error = numpy.abs(densenet201.distance_pred - densenet201.distance_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet121_distance_absolute_error, densenet169_distance_absolute_error, densenet201_distance_absolute_error],
             color=['blue', 'red', 'green'],
             #bins=15,
             #histtype='bar', stacked=True, fill=True,
             weights=[[100 / len(densenet121_distance_absolute_error)] * len(densenet121_distance_absolute_error),
                      [100 / len(densenet169_distance_absolute_error)] * len(densenet169_distance_absolute_error),
                      [100 / len(densenet201_distance_absolute_error)] * len(densenet201_distance_absolute_error)])

    #ax1.set_xlim(0, 2)
    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler mittlerer Abstand $D$ [nm]')
    plt.legend(labels=labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_simulation_histogram_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    densenet121_radius_absolute_error = numpy.abs(densenet121.radius_pred - densenet121.radius_true)
    densenet169_radius_absolute_error = numpy.abs(densenet169.radius_pred - densenet169.radius_true)
    densenet201_radius_absolute_error = numpy.abs(densenet201.radius_pred - densenet201.radius_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet121_radius_absolute_error, densenet169_radius_absolute_error, densenet201_radius_absolute_error],
             color=['blue', 'red', 'green'],
             #bins=15,
             #histtype='bar', stacked=True, fill=True,
             weights=[[100 / len(densenet121_radius_absolute_error)] * len(densenet121_radius_absolute_error),
                      [100 / len(densenet169_radius_absolute_error)] * len(densenet169_radius_absolute_error),
                      [100 / len(densenet201_radius_absolute_error)] * len(densenet201_radius_absolute_error)
                      ])

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler mittlerer Radius $R$ [nm]')
    plt.legend(labels=labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_simulation_histogram_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    densenet121_omega_distance_absolute_error = numpy.abs(densenet121.omega_distance_pred - densenet121.omega_distance_true)
    densenet169_omega_distance_absolute_error = numpy.abs(densenet169.omega_distance_pred - densenet169.omega_distance_true)
    densenet201_omega_distance_absolute_error = numpy.abs(densenet201.omega_distance_pred - densenet201.omega_distance_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet121_omega_distance_absolute_error, densenet169_omega_distance_absolute_error, densenet201_omega_distance_absolute_error],
             color=['blue', 'red', 'green'],
             #bins=15,
             #histtype='bar', stacked=True, fill=True,
             weights=[[100 / len(densenet121_omega_distance_absolute_error)] * len(densenet121_omega_distance_absolute_error),
                      [100 / len(densenet169_omega_distance_absolute_error)] * len(densenet169_omega_distance_absolute_error),
                      [100 / len(densenet201_omega_distance_absolute_error)] * len(densenet201_omega_distance_absolute_error)
                      ])

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Abstand $\omega/D$')
    plt.legend(labels=labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_simulation_histogram_omega_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma radius
    densenet121_sigma_radius_absolute_error = numpy.abs(densenet121.sigma_radius_pred - densenet121.sigma_radius_true)
    densenet169_sigma_radius_absolute_error = numpy.abs(densenet169.sigma_radius_pred - densenet169.sigma_radius_true)
    densenet201_sigma_radius_absolute_error = numpy.abs(densenet201.sigma_radius_pred - densenet201.sigma_radius_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet121_sigma_radius_absolute_error, densenet169_sigma_radius_absolute_error, densenet201_sigma_radius_absolute_error],
             color=['blue', 'red', 'green'],
             #bins=15,
             #histtype='bar', stacked=True, fill=True,
             weights=[[100 / len(densenet121_sigma_radius_absolute_error)] * len(densenet121_sigma_radius_absolute_error),
                      [100 / len(densenet169_sigma_radius_absolute_error)] * len(densenet169_sigma_radius_absolute_error),
                      [100 / len(densenet201_sigma_radius_absolute_error)] * len(densenet201_sigma_radius_absolute_error)
                      ])

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Radius $\sigma/R$')
    plt.legend(labels=labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_simulation_histogram_sigma_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_single_histogramm(folder, filename, mode='densenet169_simulation', specifier=''):
    densenet = pandas.read_csv(PATH + folder + '/' + filename)
    # distance
    densenet_distance_absolute_error = numpy.abs(densenet.distance_pred - densenet.distance_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet_distance_absolute_error],
             color=['gray'],
             bins=30,
             #histtype='bar', stacked=True, fill=True,
             edgecolor='black', linewidth=1.0,
             weights=[100 / len(densenet_distance_absolute_error)] * len(densenet_distance_absolute_error))

    #ax1.set_xlim(0, 2)
    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler mittlerer Abstand $D$ [nm]')
    plt.tight_layout()
    plt.savefig(PATH + folder + '/' + 'eval_single_' + mode + '_histogram_distance'+ specifier + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    densenet_radius_absolute_error = numpy.abs(densenet.radius_pred - densenet.radius_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet_radius_absolute_error],
             color=['gray'],
             bins=30,
             #histtype='bar', stacked=True, fill=True,
             edgecolor='black', linewidth=1.0,
             weights=[100 / len(densenet_radius_absolute_error)] * len(densenet_radius_absolute_error))

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler mittlerer Radius $R$ [nm]')
    plt.tight_layout()
    plt.savefig(PATH + folder + '/' + 'eval_single_' + mode + '_histogram_radius'+ specifier + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    densenet_omega_distance_absolute_error = numpy.abs(densenet.omega_distance_pred - densenet.omega_distance_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet_omega_distance_absolute_error],
             color=['gray'],
             bins=30,
             #histtype='bar', stacked=True, fill=True,
             edgecolor='black', linewidth=1.0,
             weights=[100 / len(densenet_omega_distance_absolute_error)] * len(densenet_omega_distance_absolute_error))

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Abstand $\omega/D$')
    plt.tight_layout()
    plt.savefig(PATH + folder + '/' + 'eval_single_' + mode + '_histogram_omega_distance'+ specifier + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma radius
    densenet_sigma_radius_absolute_error = numpy.abs(densenet.sigma_radius_pred - densenet.sigma_radius_true)
    fig, ax1 = plt.subplots(dpi=600)
    ax1.hist([densenet_sigma_radius_absolute_error],
             color=['gray'],
             bins=30,
             #histtype='bar', stacked=True, fill=True,
             edgecolor='black', linewidth=1.0,
             weights=[100 / len(densenet_sigma_radius_absolute_error)] * len(densenet_sigma_radius_absolute_error))

    plt.xlim(left=0)
    plt.ylabel(r'Häufigkeit [\%]')
    plt.xlabel(r'Absoluter Fehler Verteilung mittlerer Radius $\sigma/R$')
    plt.tight_layout()
    plt.savefig(PATH + folder + '/' + 'eval_single_' + mode + '_histogram_sigma_radius'+ specifier + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_prediction_experiment(folders, filenames, labels=['DenseNet-121', 'DenseNet-169', 'DenseNet-201']):
    # densenet121
    prediction_densenet121 = pandas.read_csv(PATH + folders[0] + '/' + filenames[0], index_col=0)
    prediction_densenet121 = prediction_densenet121[prediction_densenet121.meausurement == 'si_au3w_125c_80']
    # densenet169
    prediction_densenet169 = pandas.read_csv(PATH + folders[1] + '/' + filenames[1], index_col=0)
    prediction_densenet169 = prediction_densenet169[prediction_densenet169.meausurement == 'si_au3w_125c_80']
    # densenet201
    prediction_densenet201 = pandas.read_csv(PATH + folders[2] + '/' + filenames[2], index_col=0)
    prediction_densenet201 = prediction_densenet201[prediction_densenet201.meausurement == 'si_au3w_125c_80']
    # distance
    figure_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet121.frame, prediction_densenet121.distance_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet121.frame, prediction_densenet121.distance_pred, color="blue", linewidth=2.5, linestyle='-', label=r'DenseNet-121')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.plot(prediction_densenet201.frame, prediction_densenet201.distance_pred, color="green", linewidth=2.5, linestyle='-', label=r'DenseNet-201')
    plt.fill_between(prediction_densenet201.frame, 8, 10, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen / Datensatz si_au3w_125c_80')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend()
    plt.ylim(2, 14)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    figure_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet121.frame, prediction_densenet121.radius_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet121.frame, prediction_densenet121.radius_pred, color="blue", linewidth=2.5, linestyle='-', label=r'DenseNet-121')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.plot(prediction_densenet201.frame, prediction_densenet201.radius_pred, color="green", linewidth=2.5, linestyle='-', label=r'DenseNet-201')
    plt.fill_between(prediction_densenet201.frame, 3, 4, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend()
    plt.ylim(0, 7)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    figure_omega_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet121.frame, prediction_densenet121.omega_distance_pred, color="blue", linewidth=2.5, linestyle='-', label=r'DenseNet-121')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.omega_distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.plot(prediction_densenet201.frame, prediction_densenet201.omega_distance_pred, color="green", linewidth=2.5, linestyle='-', label=r'DenseNet-201')
    plt.fill_between(prediction_densenet201.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend()
    plt.ylim(-0.75, 0.75)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_omega_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma_radius
    figure_sigma_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet121.frame, prediction_densenet121.sigma_radius_pred, color="blue", linewidth=2.5, linestyle='-', label=r'DenseNet-121')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.sigma_radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.plot(prediction_densenet201.frame, prediction_densenet201.sigma_radius_pred, color="green", linewidth=2.5, linestyle='-', label=r'DenseNet-201')
    plt.fill_between(prediction_densenet201.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Radius $\sigma/R$ ')
    plt.legend()
    plt.ylim(-0.75, 0.75)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_sigma_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

def plot_prediction_single_experiment(folder, filename, frames=[450, 1250]):
    # densenet121
    prediction_densenet = pandas.read_csv(PATH + folder + '/' + filename, index_col=0)
    prediction_densenet = prediction_densenet[prediction_densenet.meausurement == 'si_au3w_125c_80']
    # distance
    figure_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet.frame, prediction_densenet.distance_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet.frame, prediction_densenet.distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet.frame, 8, 12, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen / Datensatz si_au3w_125c_80')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(5, 13)
    plt.xlim(frames[0], frames[1])
    plt.savefig(PATH + folder + '/' + 'eval_prediciton_experiment_distance_extend' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    figure_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet.frame, prediction_densenet.radius_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet.frame, prediction_densenet.radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet.frame, 3, 5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend(loc='upper left')
    plt.ylim(2, 6)
    plt.xlim(frames[0], frames[1])
    plt.savefig(PATH + folder + '/' + 'eval_prediciton_experiment_radius_extend' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    figure_omega_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet.frame, prediction_densenet.omega_distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend(loc='lower right')
    plt.ylim(-0.2, 0.6)
    plt.xlim(frames[0], frames[1])
    plt.savefig(PATH + folder + '/' + 'eval_prediciton_experiment_omega_distance_extend' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma_radius
    figure_sigma_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet.frame, prediction_densenet.sigma_radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Radius $\sigma/R$ ')
    plt.legend(loc='lower right')
    plt.ylim(-0.2, 0.6)
    plt.xlim(frames[0], frames[1])
    plt.savefig(PATH + folder + '/' + 'eval_prediciton_experiment_sigma_radius_extend' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()



def plot_prediction_data_increasing(folders, filenames):
    # 33 % : 33124
    # 38 % : 43681
    # 42 % : 53361
    # 46 % : 64009
    # 50 % : 75625
    mae = [sum(calculate_mae(folders[0], filenames[0]))/4,
           sum(calculate_mae(folders[1], filenames[1]))/4,
           sum(calculate_mae(folders[2], filenames[2]))/4,
           sum(calculate_mae(folders[3], filenames[3]))/4,
           sum(calculate_mae(folders[4], filenames[4]))/4,
          ]
    values = [19874, 26209, 32017, 38405, 45375]
    # distance
    figure_distance = plt.figure(dpi=600)
    plt.plot(values, mae, color="red", linewidth=2.5, linestyle='-', label=r'$D$')
    #plt.plot(values, mae[1], color="blue", linewidth=2.5, linestyle='-', label=r'$R$')
    #plt.plot(values, mae[2], color="green", linewidth=2.5, linestyle='-', label=r'$\omega/D$')
    #plt.plot(values, mae[:][3], color="orange", linewidth=2.5, linestyle='-', label=r'$\sigma/R$')
    #plt.title(r'DenseNet169 / 25 Epochen / Datensatz si_au3w_125c_80')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'MAE')
    plt.legend(loc='upper right')
    #plt.xlim(0, )
    #plt.ylim(0, )
    #plt.xlim(450, 850)
    #plt.savefig(PATH + 'results_final/results_data_increasing' + '/' + 'eval_prediciton_experiment_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_prediction_experiment_data_increasing(folders, filenames):
    # 33 % : 33124
    # 38 % : 43681
    # 42 % : 53361
    # 46 % : 64009
    # 50 % : 75625
    labels = ['19874', '26209', '32017', '38405']
    # data size 1
    prediction_densenet_1 = pandas.read_csv(PATH + folders[0] + '/' + filenames[0], index_col=0)
    prediction_densenet_1 = prediction_densenet_1[prediction_densenet_1.meausurement == 'si_au3w_125c_80']
    # data size 2
    prediction_densenet_2 = pandas.read_csv(PATH + folders[1] + '/' + filenames[1], index_col=0)
    prediction_densenet_2 = prediction_densenet_2[prediction_densenet_2.meausurement == 'si_au3w_125c_80']
    # data size 3
    prediction_densenet_3 = pandas.read_csv(PATH + folders[2] + '/' + filenames[2], index_col=0)
    prediction_densenet_3 = prediction_densenet_3[prediction_densenet_3.meausurement == 'si_au3w_125c_80']
    # data size 4
    prediction_densenet_4 = pandas.read_csv(PATH + folders[3] + '/' + filenames[3], index_col=0)
    prediction_densenet_4 = prediction_densenet_4[prediction_densenet_4.meausurement == 'si_au3w_125c_80']
    # data size 5
    prediction_densenet_5 = pandas.read_csv(PATH + folders[4] + '/' + filenames[4], index_col=0)
    prediction_densenet_5 = prediction_densenet_5[prediction_densenet_5.meausurement == 'si_au3w_125c_80']

    # distance
    figure_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.distance_true, color='black', linewidth=2.5, linestyle='-', label='Berechnung')
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.distance_pred, color="red", linewidth=2.5, linestyle='-', label=labels[0])
    plt.plot(prediction_densenet_2.frame, prediction_densenet_2.distance_pred, color="blue", linewidth=2.5, linestyle='-', label=labels[1])
    plt.plot(prediction_densenet_3.frame, prediction_densenet_3.distance_pred, color="green", linewidth=2.5, linestyle='-', label=labels[2])
    plt.plot(prediction_densenet_4.frame, prediction_densenet_4.distance_pred, color="magenta", linewidth=2.5, linestyle='-', label=labels[3])
    #plt.plot(prediction_densenet_5.frame, prediction_densenet_5.distance_pred, color="cyan", linewidth=2.5, linestyle='-', label='75625')
    plt.fill_between(prediction_densenet_1.frame, 8, 10, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen / Datensatz si_au3w_125c_80')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(2, 14)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/results_data_increasing' + '/' + 'eval_prediciton_experiment_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    figure_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.radius_true, color='black', linewidth=2.5, linestyle='-', label='Berechnung')
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.radius_pred, color="red", linewidth=2.5, linestyle='-', label=labels[0])
    plt.plot(prediction_densenet_2.frame, prediction_densenet_2.radius_pred, color="blue", linewidth=2.5, linestyle='-', label=labels[1])
    plt.plot(prediction_densenet_3.frame, prediction_densenet_3.radius_pred, color="green", linewidth=2.5, linestyle='-', label=labels[2])
    plt.plot(prediction_densenet_4.frame, prediction_densenet_4.radius_pred, color="magenta", linewidth=2.5, linestyle='-', label=labels[3])
    #plt.plot(prediction_densenet_5.frame, prediction_densenet_5.radius_pred, color="cyan", linewidth=2.5, linestyle='-', label='75625')
    plt.fill_between(prediction_densenet_1.frame, 3, 4, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(0, 7)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/results_data_increasing'+ '/' + 'eval_prediciton_experiment_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    figure_omega_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.omega_distance_pred, color="red", linewidth=2.5, linestyle='-', label=labels[0])
    plt.plot(prediction_densenet_2.frame, prediction_densenet_2.omega_distance_pred, color="blue", linewidth=2.5, linestyle='-', label=labels[1])
    plt.plot(prediction_densenet_3.frame, prediction_densenet_3.omega_distance_pred, color="green", linewidth=2.5, linestyle='-', label=labels[2])
    plt.plot(prediction_densenet_4.frame, prediction_densenet_4.omega_distance_pred, color="magenta", linewidth=2.5, linestyle='-', label=labels[3])
    #plt.plot(prediction_densenet_5.frame, prediction_densenet_5.omega_distance_pred, color="cyan", linewidth=2.5, linestyle='-', label='75625')
    plt.fill_between(prediction_densenet_1.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend(loc='lower right')
    plt.ylim(-0.75, 0.75)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/results_data_increasing' + '/' + 'eval_prediciton_experiment_omega_distance' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma_radius
    figure_sigma_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet_1.frame, prediction_densenet_1.sigma_radius_pred, color="red", linewidth=2.5, linestyle='-', label='19875')
    plt.plot(prediction_densenet_2.frame, prediction_densenet_2.sigma_radius_pred, color="blue", linewidth=2.5, linestyle='-', label='43681')
    plt.plot(prediction_densenet_3.frame, prediction_densenet_3.sigma_radius_pred, color="green", linewidth=2.5, linestyle='-', label='53361')
    plt.plot(prediction_densenet_4.frame, prediction_densenet_4.sigma_radius_pred, color="magenta", linewidth=2.5, linestyle='-', label='64009')
    #plt.plot(prediction_densenet_5.frame, prediction_densenet_5.sigma_radius_pred, color="cyan", linewidth=2.5, linestyle='-', label='75625')
    plt.fill_between(prediction_densenet_1.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Abweichung mittlerer Radius $\sigma/R$ ')
    plt.legend(loc='lower right')
    plt.ylim(-0.75, 0.75)
    plt.xlim(450, 850)
    plt.savefig(PATH + 'results_final/results_data_increasing' + '/' + 'eval_prediciton_experiment_sigma_radius' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

def plot_prediction_experiment_out_of_bound(folder, filename):
    # densenet169
    prediction_densenet169 = pandas.read_csv(PATH + folder + '/' + filename, index_col=0)
    prediction_densenet169 = prediction_densenet169[prediction_densenet169.meausurement == 'si_au3w_125c_80']
    # distance
    figure_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet169.frame, prediction_densenet169.distance_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet169.frame, 8, 10, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen / Datensatz si_au3w_125c_80')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(4, 16)
    plt.xlim(200, 1600)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_distance_out_of_bound' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    figure_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet169.frame, prediction_densenet169.radius_true, color='black', linewidth=2.5, linestyle='-', label=r'Berechnung')
    plt.plot(prediction_densenet169.frame, prediction_densenet169.radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet169.frame, 3, 4, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'Nummer GISAXS-Streubild in Messung')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend(loc='upper left')
    plt.ylim(0, 8)
    plt.xlim(200, 1600)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_radius_out_of_bound' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega_distance
    figure_omega_distance = plt.figure(dpi=600)
    plt.plot(prediction_densenet169.frame, prediction_densenet169.omega_distance_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet169.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend()
    plt.ylim(-0.75, 0.75)
    plt.xlim(300, 1700)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_omega_distance_out_of_bound' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma_radius
    figure_sigma_radius = plt.figure(dpi=600)
    plt.plot(prediction_densenet169.frame, prediction_densenet169.sigma_radius_pred, color="red", linewidth=2.5, linestyle='-', label=r'DenseNet-169')
    plt.fill_between(prediction_densenet169.frame, 0, 0.5, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    plt.xlabel(r'GISAXS-Streubild in Messung')
    plt.ylabel(r'Verteilung mittlerer Radius $\sigma/R$ ')
    plt.legend()
    plt.ylim(-0.75, 0.75)
    plt.xlim(300, 1700)
    plt.savefig(PATH + 'results_final/data' + '/' + 'eval_prediciton_experiment_sigma_radius_out_of_bound' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_simulation_prediction_under_interval(folder, model, filename):
    prediction_simulation = pandas.read_csv(PATH + folder + '/' + filename, index_col=0)
    prediction_simulation = prediction_simulation.sample(n=400)
    # distance
    data = prediction_simulation.sort_values(by='distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.distance_true, color='black', linewidth=0.01, linestyle='-', label='Zielwert')
    plt.scatter(range(0, len(data)), data.distance_pred, color="red", linewidth=0.01, linestyle='-', label='DenseNet-169')
    plt.fill_between(range(0, len(data)), 8, 10, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend(loc='upper right')
    plt.ylim(0,16)
    plt.xlim(0, len(data)-1)
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_distance_under_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    data = prediction_simulation.sort_values(by='radius_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.radius_true, color='black', linewidth=0.01, linestyle='-', label='Zielwert')
    plt.scatter(range(0, len(data)), data.radius_pred, color="red", linewidth=0.01, linestyle='-', label='DenseNet-169')
    plt.fill_between(range(0, len(data)), 3, 4, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    #plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend(loc='upper right')
    plt.ylim(0, 8)
    plt.xlim(0, len(prediction_simulation)-1)
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_radius_under_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega distance
    data = prediction_simulation.sort_values(by='omega_distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.omega_distance_true, color="black", linewidth=1.5, linestyle='-', label='Wahrer Wert')
    plt.scatter(range(0, len(data)), data.omega_distance_pred, color="red", linewidth=1.5, label='Vorhersage')
    #plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend(loc='upper right')
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_omega_distance_under_interval_' + DATATYPE, bbox_inches='tight',dpi=600)
    plt.show()

    # sigma radius
    data = prediction_simulation.sort_values(by='omega_distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.sigma_radius_true, color="black", linewidth=1.5, linestyle='-', label='Wahrer Wert')
    plt.scatter(range(0, len(data)), data.sigma_radius_pred, color="red", linewidth=1.5, label='Vorhersage')
    #plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Verteilung mittlerer Radius $\sigma/R$')
    plt.legend(loc='upper right')
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_sigma_radius_under_interval_' + DATATYPE, bbox_inches='tight',dpi=600)
    plt.show()

def plot_simulation_prediction_above_interval(folder, model, filename):
    prediction_simulation = pandas.read_csv(PATH + folder + '/' + filename, index_col=0)
    prediction_simulation = prediction_simulation.sample(n=400)
    # distance
    data = prediction_simulation.sort_values(by='distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.distance_true, color='black', linewidth=0.01, linestyle='-', label='Zielwert')
    plt.scatter(range(0, len(data)), data.distance_pred, color="red", linewidth=0.01, linestyle='-', label='DenseNet-169')
    plt.fill_between(range(0, len(data)), 8, 10, facecolor='gray', alpha=0.5, label='Trainingsbereich')

    # plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Mittlerer Abstand $D$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(0, 16)
    plt.xlim(0, len(data) - 1)
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_distance_above_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # radius
    data = prediction_simulation.sort_values(by='radius_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.radius_true, color='black', linewidth=0.01, linestyle='-', label='Zielwert')
    plt.scatter(range(0, len(data)), data.radius_pred, color="red", linewidth=0.01, linestyle='-', label='DenseNet-169')
    plt.fill_between(range(0, len(data)), 3, 4, facecolor='gray', alpha=0.5, label='Trainingsbereich')
    # plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Mittlerer Radius $R$ [nm]')
    plt.legend(loc='lower right')
    plt.ylim(0, 8)
    plt.xlim(0, len(prediction_simulation) - 1)
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_radius_above_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # omega distance
    data = prediction_simulation.sort_values(by='omega_distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.omega_distance_true, color="black", linewidth=1.5, linestyle='-', label='Wahrer Wert')
    plt.scatter(range(0, len(data)), data.omega_distance_pred, color="red", linewidth=1.5, label='Vorhersage')
    # plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Verteilung mittlerer Abstand $\omega/D$')
    plt.legend(loc='lower right')
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_omega_distance_above_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()

    # sigma radius
    data = prediction_simulation.sort_values(by='omega_distance_true')
    figure = plt.figure(dpi=600)
    plt.scatter(range(0, len(data)), data.sigma_radius_true, color="black", linewidth=1.5, linestyle='-',label='Wahrer Wert')
    plt.scatter(range(0, len(data)), data.sigma_radius_pred, color="red", linewidth=1.5, label='Vorhersage')
    # plt.title(r'DenseNet169 / 25 Epochen')
    plt.xlabel(r'Nummer GISAXS-Streubild')
    plt.ylabel(r'Verteilung mittlerer Radius $\sigma/R$')
    plt.legend(loc='lower right')
    plt.savefig(PATH + folder + '/' + 'eval_' + model + '_sigma_radius_above_interval_' + DATATYPE, bbox_inches='tight', dpi=600)
    plt.show()


def plot_attention_maps():
    path_project = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/'
    path_simulation_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/simulation/', 'files': ['database_1.h5']}
    path_experiment_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/experiment/', 'folders': ['si_au3w_125c_80', 'si_au3w_225c_81'], 'target_value_files': ['SP400K.csv', 'SP500K.csv'], 'first_frame': [190, 190]}
    # parser
    arg = arguments.parse()
    deep_learning_parameter = {'algorithm': arg.algorithm, 'run': arg.run, 'path': path_project, 'extend': arg.extend}
    # initialize detector, experiment setup, data augmentation
    detec = detector.Pilatus1M(mask=(path_project + 'base/masks/' + arg.maskfile))
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x), 'sample_detector_distance': arg.distance}, detector=detector, experiment_maskfile=arg.experiment_maskfile)
    da = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detec)
    # create simulation object
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, samples=arg.samples)
    # fit simulation images
    simulation_images, simulation_target_values = simulation.load_extended_data_for_testcase(samples=0.1)
    simulation_images, simulation_target_values = fit_simulation(da=da, simulation=simulation, images=simulation_images[0:5], target_values=simulation_target_values[0:5])
    # initialize algorithm
    algorithm = main.intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter)
    # load weights
    model = algorithm.load_weights(model=algorithm.model)
    # compile model
    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanAbsoluteError(), metrics=[metrics.MeanSquaredError(), metrics.RootMeanSquaredError()])
    utilities.plot_saliency_map(model=model, image=simulation_images, target_value=simulation_target_values.distance, variable='distance', image_to_pred=algorithm.create_input_tensor(images=simulation_images))




# TODO: helper functions
def fit_simulation(da, simulation, images, target_values):
    fitted_images = []
    # start fitting
    n1, n2, two_theta_f_min, two_theta_f_max, alpha_f_min, alpha_f_max = simulation.get_grid_parameter(key='1')
    two_theta_f, alpha_f, q_y, q_z = da.convert_from_cartesian_to_reciprocal_space()
    two_theta_f_crop_index = int(round((two_theta_f.max() / (two_theta_f_max - two_theta_f_min)) * n2))
    alpha_f_crop_index = int(round((alpha_f.max() / (alpha_f_max - alpha_f_min)) * (n1 - 1)))
    # crop masks
    detector_mask = da.crop_detector_mask()
    experiment_mask = da.crop_experiment_mask()
    shape_to_bin = (int(math.ceil(0.5 * detector_mask.shape[0])), int(math.ceil(0.5 * detector_mask.shape[1])))
    detector_mask = da.crop_window(da.bin_mask(detector_mask, bin_to_shape=shape_to_bin))
    for image in tqdm.tqdm(images, desc="step: fit simulation data", total=len(images), mininterval=120, miniters=1000):
        # crop
        image = da.crop_simulation(image=image, y=alpha_f_crop_index, x=two_theta_f_crop_index)
        # bin simulation image to experiment image
        image = da.bin_int(to_bin=image, bin_to_shape=shape_to_bin)
        # noise
        image = da.add_poisson_shot_noise(image=image)
        # crop window
        image = da.crop_window(image=image)
        # mask
        image = da.mask_image(image=image, mask=detector_mask)
        # normalize
        image = da.normalize(image=image)
        # segment with watershed
        image = da.segment_watershed(image=image, median_smooth_factor=3)
        # append images
        fitted_images.append(image)
    return fitted_images, target_values


def fit_experiment(da, experiment_images, experiment_target_values):
        fitted_experiment_images = []
        # shape to bin
        shape_cropped_experiment = da.crop_experiment(image=experiment_images[0])
        shape_to_bin = (int(math.ceil(0.5 * shape_cropped_experiment.shape[0])), int(math.ceil(0.5 * shape_cropped_experiment.shape[1])))
        detector_mask = da.crop_detector_mask()
        detector_mask = da.crop_window(da.bin_mask(detector_mask, bin_to_shape=shape_to_bin))
        for image in tqdm.tqdm(experiment_images, desc="step: fit experiment data", total=len(experiment_images), mininterval=60, miniters=100):
            # crop
            image = da.crop_experiment(image=image)
            # bin
            image = da.bin_float(to_bin=image, bin_to_shape=shape_to_bin)
            # crop window
            image = da.crop_window(image=image)
            # mask
            image = da.mask_image(image=image, mask=detector_mask)
            # normalize
            image = da.normalize(image=image)
            # segment with watershed
            image = da.segment_watershed(image=image, median_smooth_factor=7)
            fitted_experiment_images.append(image)
        return fitted_experiment_images, experiment_target_values


def predict_simultion_data_out_of_bound(d_interval, r_interval, mode='under'):
    path_project = '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/'
    path_simulation_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/simulation/', 'files': ['database_1.h5', 'database_2.h5']}
    path_experiment_data = {'path': '/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/data/labeled/experiment/','folders': ['si_au3w_125c_80'],  # , 'si_au3w_225c_81'],si_au3w_125c_80
                            'target_value_files': ['SP400K.csv'],
                            'first_frame': [190]}
    arg = arguments.parse()
    deep_learning_parameter = {'algorithm': arg.algorithm, 'run': arg.run, 'path': path_project}
    # initialize detector,  experiment setup, data augmentation
    detec = detector.Pilatus1M(mask=(path_project + 'base/masks/' + arg.maskfile))
    experiment_setup = setup.Experiment(path_project, experiment_parameter={'materials': arg.materials, 'wavelength': arg.wavelength, 'incidence_angle': arg.incidence_angle, 'direct_beam_position': (arg.db_y, arg.db_x), 'sample_detector_distance': arg.distance}, detector=detector, experiment_maskfile=arg.experiment_maskfile)
    # initialize data augmentation
    da = data_augmentation.DataAugmentation(experiment_setup=experiment_setup, detector=detec)
    # simulation
    simulation = simulation_dataloader.Simulation(path=path_simulation_data, samples=arg.samples)
    simulation_images, simulation_target_values = simulation.load_extended_data_for_testcase(samples=100, d_interval=d_interval, r_interval=r_interval)
    # TODO: use unique distances, radius, omega_distance, sigma_radius
    sim_images = simulation_images
    sim_target_values = simulation_target_values
    simulation_images, simulation_target_values = da.fit_simulation_test(simulation, sim_images, sim_target_values)
    # initialize algorithm
    algorithm = main.intialize_algorithm(algorithm=arg.algorithm, input_shape=(simulation_images[0].shape + (1,)), parameter=deep_learning_parameter)
    # load weights and compile
    model = algorithm.load_weights_from_path(algorithm.model, path='/Users/eldaralmamedov/Desktop/Masterarbeit/AutoGiSAXS/deep_learning/models/models_final/models_final_1/densenet169/densenet169_weights_0.33_256_25_sf37_mae.h5')
    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanAbsoluteError(), metrics=[metrics.MeanSquaredError(), metrics.RootMeanSquaredError()])
    simulation_target_values_prediction = model.predict(algorithm.create_input_tensor(simulation_images))
    # pandas dataframe to numpy array
    target_values_true = simulation_target_values.to_numpy()
    # reshape from (number columns,number rows, 1) to (number columns, number rows) and transpose from (number columns, number rows) to (number rows, number columns)
    target_values_prediction = numpy.reshape(simulation_target_values_prediction, (len(simulation_target_values_prediction), len(simulation_target_values_prediction[0])))
    target_values_prediction = target_values_prediction.transpose()
    # concatenate numpy arrays
    results_array = numpy.concatenate((target_values_true, target_values_prediction), axis=1)
    # create dataframe
    results_dataframe = pandas.DataFrame(data=results_array,
                                         columns=['id_sf', 'id_ff', 'key_sf', 'key_ff',
                                                  'distance_true', 'omega_distance_true', 'radius_true',
                                                  'sigma_radius_true',
                                                  'distance_pred', 'omega_distance_pred', 'radius_pred',
                                                  'sigma_radius_pred'
                                                  ])
    results_dataframe.to_csv(path_project + 'results/results_final/' + 'eval_'+ 'densenet169_prediction_simulation_' + mode + '_interval.csv')



def plot_error_metrics_experiment(model, filename, name):
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
    plt.savefig(PATH + model + '/' + name + '_distance_experiment_true_pred' + datatype, bbox_inches='tight')
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
    plt.savefig(PATH + model + '/' + name + '_radius_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

    plt.plot(experiment_prediction.frame, experiment_prediction.omega_distance_pred, color='blue', linewidth=1.5, linestyle='-', label=r'$\frac{\omega}{D}$')
    plt.plot(experiment_prediction.frame, experiment_prediction.sigma_radius_pred, color="red", linewidth=1.5, linestyle='-', label=r'$\frac{\sigma}{R}$')
    #plt.axhline(y=0.5, xmin=0.0, xmax=1710.0, color='green', linestyle='--', label='Simulationsbereich')
    plt.title(r'si_au3w_125c_80')
    plt.xlabel(r'Frames')
    plt.ylabel(r'Variation')
    plt.legend(loc='upper right')
    plt.xlim(0, 5500)
    plt.savefig(PATH + model + '/' + name + '_omgea_distance_sigma_radius_experiment_true_pred' + datatype, bbox_inches='tight')
    plt.show()

def calculate_mae(folder, filename):
    # read csv
    data = pandas.read_csv(PATH + folder + '/' + filename)
    mae_distance = abs((data.distance_pred - data.distance_true).mean())
    mae_omega_distance = abs((data.omega_distance_pred - data.omega_distance_true).mean())
    mae_radius = abs((data.radius_pred - data.radius_true).mean())
    mae_sigma_radius = abs((data.sigma_radius_pred - data.sigma_radius_true).mean())
    return mae_distance, mae_radius, mae_omega_distance, mae_sigma_radius

if __name__ == '__main__':
    # plot loss function of densenets
    done = True
    if not done:
        # sub research question a)
        plot_model_training_and_validation_loss(folder='results_final/data/densenet121', model='densenet121', filename='densenet121_history0.33_256_25_sf37_final.csv')
        plot_model_training_and_validation_loss(folder='results_final/data/densenet169', model='densenet169', filename='densenet169_history0.33_256_25_sf37_mae.csv')
        plot_model_training_and_validation_loss(folder='results_final/data/densenet201', model='densenet201', filename='densenet201_history0.33_256_25_sf37_final.csv')

        # plot test simulation data prediction histogram of densenets
        plot_histogramm(folders=['results_final/data/densenet121', 'results_final/data/densenet169', 'results_final/data/densenet201'],
                        models=['densenet121', 'densenet169', 'densenet201'],
                        filenames=['densenet121_prediction_simulation_test_0.33_256_25_sf37_final.csv', 'densenet169_prediction_simulation_test_0.33_256_25_sf37_mae.csv', 'densenet201_prediction_simulation_test_0.33_256_25_sf37_final.csv'])

        # plot experiment data prediction diagram of densenets
        plot_prediction_experiment(folders=['results_final/data/densenet121', 'results_final/data/densenet169', 'results_final/data/densenet201'],
                                   filenames=['densenet121_prediction_experiment_0.33_256_25_sf37_final.csv', 'densenet169_prediction_experiment_0.33_256_25_sf37_mae.csv', 'densenet201_prediction_experiment_0.33_256_25_sf37_final.csv'])

        # sub research question b): training data as test data
        plot_single_histogramm(folder='results_final/data/densenet169', filename='densenet169_prediction_simulation_train_0.33_256_25_sf37_mae.csv')

        # sub research question b)
        # plot out-of-bound simulation prediction under learned interval
        predict_simultion_data_out_of_bound(d_interval=[1, 8], r_interval=[1, 3], mode='under')
        plot_simulation_prediction_under_interval(folder='results_final', model='densenet169', filename='eval_densenet169_prediction_simulation_under_interval.csv')
        # plot out-of-bound simulation prediction under learned interval
        predict_simultion_data_out_of_bound(d_interval=[1, 8], r_interval=[1, 3], mode='above')
        plot_simulation_prediction_above_interval(folder='results_final', model='densenet169', filename='eval_densenet169_prediction_simulation_above_interval.csv')
        # plot out of bound for experiment
        plot_prediction_experiment_out_of_bound(folder='results_final/data/densenet169', filename='densenet169_prediction_experiment_0.33_256_25_sf37_mae.csv')

        # extend
        # sub research question b): training data as test data
        plot_single_histogramm(folder='results_final/final_4/extend/0.25',
                               filename='densenet169_prediction_simulation_test_0.25_256_25_sf37_final_extend.csv',
                               specifier='_extend')
    plot_prediction_single_experiment(folder='results_final/final_4/extend/0.22',
                                      filename='densenet169_prediction_experiment_0.22_256_25_sf37_final_extend.csv',
                                      frames=[200, 1600])

    # useless
    # check influence of data increasing on experiment
    #plot_prediction_data_increasing(folders=['results_final/data/densenet169', 'results_final/results_data_increasing/0.38', 'results_final/results_data_increasing/0.42', 'results_final/results_data_increasing/0.46', 'results_final/results_data_increasing/0.50'],
    #                                filenames=['densenet169_prediction_simulation_test_0.33_256_25_sf37_mae.csv', 'densenet169_prediction_simulation_test_0.38_256_25_sf37_final.csv', 'densenet169_prediction_simulation_test_0.42_256_25_sf37_final.csv', 'densenet169_prediction_simulation_test_0.46_256_25_sf37_final.csv', 'densenet169_prediction_simulation_test_0.5_256_25_sf37_final.csv' ])

    #plot_prediction_experiment_data_increasing(folders=['results_final/data/densenet169', 'results_final/results_data_increasing/0.38', 'results_final/results_data_increasing/0.42', 'results_final/results_data_increasing/0.46', 'results_final/results_data_increasing/0.50'],
    #                                           filenames=['densenet169_prediction_experiment_0.33_256_25_sf37_mae.csv', 'densenet169_prediction_experiment_0.38_256_25_sf37_final.csv', 'densenet169_prediction_experiment_0.42_256_25_sf37_final.csv', 'densenet169_prediction_experiment_0.46_256_25_sf37_final.csv', 'densenet169_prediction_experiment_0.5_256_25_sf37_final.csv' ])
