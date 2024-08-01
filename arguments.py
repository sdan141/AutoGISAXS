# courtesy of E. Almamedov with modifications for our purposes

import argparse


# mode
TEST = True
REAL = False
RUN = '_256_30'

CHECK = False
SIM = False
PLOT = False

# experiment parameters
MATERIALS = 'AuSi'
WAVELENGTH = 0.09472
INCIDENTE_ANGLE = 0.395
DB_X = 466
DB_Y = 135
DISTANCE = 2357
DETECTOR = 'Pilatus1M'
MASKFILE = 'Pilatus_1M.tif'
EXP_MASKFILE = 'Pilatus_1M.tif'

# data preparation parameters
PREMADE_SIM = True # simulations are generated in advance (loaded)
PREMADE_EXP = True # experiments are prepared in advance (loaded)
SIM_SOURCE = 'ready_file'
BETA = None

# validation parameters
VALIDATION = 'exp' # 'exp_reduced' # 'sim'
# exp for using a whole dataset of experiment for the validation
# exp_reduced for using only subset of the experiment dataset
# sim using only simulations for validatio

# deep learning parameters
NN = 'CNN' #'CNN' # 'MLP' # 'VGG' # 'DENSE'
FLAG = 'radius' # 'distance'
DISTR = False

ESTIMATION = 'naive' # naive confidence estimation
LABELS_R = 280
LABELS_D = 370
LABELS_S = 200
LABELS_W = 200
#'cross_val' # Monte Carlo cross validation to obtain estimation -> only if validation data is same as for training



def initialize_argument_parser():
    """Common configuration for parsers"""
    parser = argparse.ArgumentParser()
    # modes
    parser.add_argument('--test', dest='test', type=bool, help='train and test the network', default=TEST)
    parser.add_argument('--real', dest='real', type=bool, help='train and execute the network on unlabeled data', default=REAL)
    parser.add_argument('--check', dest='check', type=bool, help='check that pipeline works as expected', default=CHECK)
    parser.add_argument('--sim', dest='sim', type=bool, help='create new simulations and save them', default=SIM)
    parser.add_argument('--plot', dest='plot', type=bool, help='create sample plots in runtime', default=PLOT)

    # experiment parameter
    parser.add_argument('--materials', dest='materials', type=str, help='considered materials', default=MATERIALS)
    parser.add_argument('--wavelength', dest='wavelength', type=float, help='wavelength of the X-Ray beam [nm]', default=WAVELENGTH)
    parser.add_argument('--incidence_angle', dest='incidence_angle', type=float, help='incidence angle of X-Ray beam [degree]', default=INCIDENCE_ANGLE)
    parser.add_argument('--db_x', dest='db_x', type=int, help='direct beam position on x-axis of the detector [pixel]', default=DB_X)
    parser.add_argument('--db_y', dest='db_y', type=int, help='direct beam position on y-axis of the detector [pixel]', default=DB_Y)
    parser.add_argument('--distance', dest='distance', type=float, help='distance between sample and detector [mm].', default=DISTANCE)
    parser.add_argument('--detector', dest='detector', type=str, help='select detector',
                        choices=['Eiger500k', 'Eiger1M', 'Eiger4M', 'Eiger9M', 'Eiger16M', 'Pilatus100k', 'Pilatus200k', 'Pilatus300k',
                                 'Pilatus300kw', 'Pilatus1M', 'Pilatus2M', 'Pilatus6M'], default=DETECTOR)
    parser.add_argument('--maskfile', dest='maskfile', type=str, help='file containing the mask', default=MASKFILE)

    # data preparation parameters
    parser.add_argument('--fast_sim', dest='fast_sim', type=bool, help='use simulations prepared in advance', default=PREMADE_SIM)
    parser.add_argument('--fast_exp', dest='fast_exp', type=bool, help='use experiment images prepared in advance', default=PREMADE_EXP)
    parser.add_argument('--sim_source', dest='sim_source', type=str, help='select source of simulations data', choices=['factors_h5', 'factors_file', 'ready_h5', 'ready_file'], default=SIM_SOURCE)
    parser.add_argument('--beta', dest='beta', type=float, help='choose parameter for intensity thresholding', default=BETA)

    # deep learning parameter
    parser.add_argument('--validation', dest='validation', type=str, help='select data to use for validation, if "other" specified: modify path to validation data and adjust file format',
                                                                     choices=['exp', 'exp_reduced', 'sim', 'other', 'none'], default=VALIDATION)
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='select deep learning algorithm', choices=['CNN', 'MLP', 'VGG', 'Dense'], default=NN)
    parser.add_argument('--morph', dest='morphology', type=str, help='select morphological parameter to predict', choices=['all', 'radius', 'distance'], default=FLAG)
    parser.add_argument('--distr', dest='distr', type=bool, help='predict distributions', default=DISTR)

    parser.add_argument('--estimation', dest='estimation', type=str, help='select confidence estimation methode, keep in mind "cross_validation" suitable only when simulations are used as validation data',
                                                                     choices=['naive', 'cross_validation', 'none'], default=ESTIMATION)
    parser.add_argument('--radius_classes', dest='radius_classes', type=int, help='number of possible classes for radius', default=LABELS_R)
    parser.add_argument('--distance_classes', dest='distance_classes', type=int, help='number of possible classes for distance', default=LABELS_D)
    parser.add_argument('--sigma_classes', dest='sigma_classes', type=int, help='number of possible classes for sigma/radius', default=LABELS_S)
    parser.add_argument('--omega_classes', dest='omega_classes', type=int, help='number of possible classes for omega/distance', default=LABELS_W)
    return parser


def parse():
    parser = initialize_argument_parser()
    return parser.parse_args()
