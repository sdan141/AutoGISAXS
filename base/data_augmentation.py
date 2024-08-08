# courtesy of E. Almamedov with modifications for our purposes

import numpy as np
import pandas as pd
import math
from skimage import morphology, segmentation, filters, util, transform

class DataAugmentation:

    def __init__(self, experiment_setup, detector, beta=None):
        self.experiment_setup = experiment_setup
        self.detector = detector
        self.beta = beta
        self.ROI = None
        self.add_noise = False
        self.median = False
        self.gradient = False
        self.tau = None     # global intensity threshold
        self.gamma = None   # global background level

    ####################################################
    # functions for simuation-experiment domain adaption
    ####################################################

    def get_vetical_cut_position(self):
        """
        Cut position on y-axis of image (height).
        :return: position
        """
        ttheta_f, alpha_f, q_y, q_z = self.convert_from_cartesian_to_reciprocal_space()
        y, x = self.experiment_setup.get_direct_beam_position()
        return y + np.count_nonzero(alpha_f < 0.0)

    def get_horizontal_cut_position(self):
        """
        Cut position on x-axis of image (width).
        :return: position
        """
        y, x = self.experiment_setup.get_direct_beam_position()
        return x

    def convert_from_cartesian_to_reciprocal_space(self):
        # create detector interval with respective pixel size
        height, width = self.detector.get_maximum_shape()
        db_y, db_x = self.experiment_setup.get_direct_beam_position()
        ps_y, ps_x = (self.detector.get_pixel1() * 1000, self.detector.get_pixel2() * 1000)                 # [mm]
        y = np.linspace(start=0, stop=(height - db_y), num=(height - db_y), dtype=np.float32) * ps_y  # [mm]
        x = np.linspace(start=0, stop=(width - db_x), num=(width - db_x), dtype=np.float32) * ps_x    # [mm]
        d_sd = self.experiment_setup.get_sample_detector_distance()                                         # [mm]
        # calculate angles
        k0 = (2 * np.pi) / self.experiment_setup.get_wavelength()  # [nm^-1]
        alpha_i = self.experiment_setup.get_incidence_angle()         # [degree]
        two_theta_f = np.degrees(np.arctan(x/d_sd))             # [degree]
        alpha_f = np.degrees(np.arctan(y/d_sd)) - alpha_i       # [degree]
        # calculate qy, qz using the grazing incidence approximation: cos(alpha_f) = 1
        q_y = k0 * np.sin(np.radians(two_theta_f)) * 1  # cos(alpha_f) = 1
        q_z = k0 * (np.sin(np.radians(two_theta_f)) + np.sin(np.radians(alpha_i)))
        return two_theta_f, alpha_f, q_y, q_z

    def get_axis_degree(self, x, y, horizon_cut):
        detector_y = np.linspace(0, y, y, dtype=float) * self.detector.get_pixel1()
        detector_x = np.linspace(0, x - horizon_cut, x - horizon_cut, dtype=float) * self.detector.get_pixel2()
        theta = np.arcsin(detector_x / self.experiment_setup.get_sample_detector_distance()) * (180 / np.pi)
        ai = np.arcsin(detector_y / self.experiment_setup.get_sample_detector_distance()) * (180 / np.pi)
        return theta, ai

    def crop_detector_mask(self):
        detector_mask = np.flip(np.flip(self.detector.load_mask(), axis=0), axis=1)
        height, width = detector_mask.shape
        y = self.get_vetical_cut_position()
        x = self.get_horizontal_cut_position()
        return detector_mask[y:, (width-x):width]

    def crop_experiment_mask(self):
        experiment_maskfile = self.experiment_setup.get_experiment_maskfile()
        height, width = experiment_maskfile.shape
        experiment_maskfile = np.flip(experiment_maskfile, axis=0)
        experiment_maskfile = np.flip(experiment_maskfile, axis=1)
        y = self.get_vetical_cut_position()
        x = self.get_horizontal_cut_position()
        return experiment_maskfile[y:, (width-x):width]

    def crop_experiment(self, image):
        image = np.flip(image, axis=1)
        heigth, width = image.shape
        y = self.get_vetical_cut_position()
        x = self.get_horizontal_cut_position()
        return image[y:, (width-x):width]

    def crop_simulation(self, image, y, x):
        return image[0:y, 0:x]

    def crop_window(self, image, y=260, x=13):
        return image[:y, x:]

    def bin_simulation(self, to_bin, bin_to_shape, mode='sum'):
        """
        In general, Binning will combine the information of adjacent pixel into resulting information depending on
        binning mode. That will in any case lead to a reduced resolution by the factor of bin_int. But also it will sum
        the performance of each single pixel. There are three types of bin_int available: horizontal bin_int, vertical
        binning, and full binning. We use the full binning.

        :param to_bin: image to bin_simulation (::type ndarray)
        :param bin_to_shape: image bin_simulation to (::type ndarray)
        :param mode: 'avg' or 'sum' (::type str)
        :return: binned_image image (::rtype ndarray)
        """
        binned_image = np.empty(shape=bin_to_shape).astype(np.float32)
        y_bin = to_bin.shape[0] / bin_to_shape[0]
        x_bin = to_bin.shape[1] / bin_to_shape[1]
        y_to_diff = np.ceil(y_bin) - y_bin
        y_from_diff = 1 - y_to_diff
        x_to_diff = np.ceil(x_bin) - x_bin
        x_from_diff = 1 - x_to_diff
        for i in range(1, bin_to_shape[0] - 1, 1):
            for j in range(1, bin_to_shape[1] - 1, 1):
                # floor: round number down to the nearest integer
                # ceil: round a number upward to its nearest integer
                from_y = i * int(y_bin)
                to_y = (i + 1) * int(y_bin)
                from_x = j * int(x_bin)
                to_x = (j + 1) * int(x_bin)
                binned_image[i, j] = (to_bin[from_y: to_y, from_x: to_x].sum() +
                                          to_bin[from_y : to_y, from_x - 1].sum() * x_from_diff +
                                          to_bin[from_y : to_y, to_x + 1].sum() * x_to_diff +
                                          to_bin[from_y - 1, from_x : to_x].sum() * y_from_diff +
                                          to_bin[to_y + 1, from_x : to_x].sum() * y_to_diff
                                          )
                if mode == 'avg':
                    binned_image[i, j] /= (y_bin * x_bin)

        for y in range(0, binned_image.shape[0]):
            for x in range(0, binned_image.shape[1]):
                if math.isnan(binned_image[y][x]) == True:
                    binned_image[y][x] == float(0)

        return binned_image.astype(np.float32)

    def bin_mask(self, mask_to_bin, bin_to_shape):
        mask = np.empty(shape=bin_to_shape)
        y_bin = mask_to_bin.shape[0] / bin_to_shape[0]
        x_bin = mask_to_bin.shape[1] / bin_to_shape[1]
        for i in range(0, bin_to_shape[0], 1):
            for j in range(0, bin_to_shape[1], 1):
                # floor: round number down to the nearest integer
                # ceil: round a number upward to its nearest integer
                from_y = math.floor(i * y_bin)
                to_y = math.ceil((i + 1) * y_bin)
                from_x = math.floor(j * x_bin)
                to_x = math.ceil((j + 1) * x_bin)
                mask[i, j] = mask_to_bin[from_y: to_y, from_x: to_x].sum() / (y_bin * x_bin)

        for i in range(0, mask.shape[0]):
            for j in range(0, mask.shape[1]):
                if mask[i, j] < 1.0:
                    if mask[i, j] >= 0.5:
                        mask[i, j] = 1.0
                    else:
                        mask[i, j] = 0.0
                elif mask[i, j] >= 1.0:
                    mask[i, j] = 1.0
        return mask.astype(np.float32)

    def bin_float(self, to_bin, bin_to_shape):
        binned_image = np.empty(shape=bin_to_shape)
        y_bin = to_bin.shape[0] / bin_to_shape[0]
        x_bin = to_bin.shape[1] / bin_to_shape[1]
        for i in range(0, bin_to_shape[0],1):
            for j in range(0, bin_to_shape[1],1):
                from_y = math.floor(i*y_bin)
                to_y = math.ceil((i+1)*y_bin)
                from_x = math.floor(j*x_bin)
                to_x = math.ceil((j+1)*x_bin)
                binned_image[i,j] = to_bin[from_y:to_y,from_x:to_x].sum()
        return binned_image

    def bin_int(self, to_bin, bin_to_shape, mode='sum'):
        binned_image = np.empty(shape=bin_to_shape)
        y_bin = to_bin.shape[0] / bin_to_shape[0]
        x_bin = to_bin.shape[1] / bin_to_shape[1]
        for i in range(0, bin_to_shape[0], 1):
            for j in range(0, bin_to_shape[1], 1):
                # floor: round number down to the nearest integer
                # ceil: round a number upward to its nearest integer
                from_y = i * int(y_bin)
                to_y = (i + 1) * int(y_bin)
                from_x = j * int(x_bin)
                to_x = (j + 1) * int(x_bin)
                if mode == 'sum':
                    binned_image[i, j] = to_bin[from_y: to_y, from_x: to_x].sum()
                elif mode == 'avg':
                    binned_image[i, j] = to_bin[from_y: to_y, from_x: to_x].sum() / (y_bin * x_bin)
        return binned_image

    def normalize(self, image):
        return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)

    def mask_image(self, image, mask):
        masked_image = np.ma.masked_array(image, mask=mask).astype(np.float32)
        masked_image = masked_image.filled(fill_value=0.0)
        return masked_image


    ###################
    # fitting functions
    ###################

    def fit_simulation_ready(self, simulation, path_to_fast_sim=None):
        """
        Prossece the fitted simulated scattering images
        :param: path_to_fast_sim the path to data
        :return: list of fitted images (:rtype: list of ndarray)
        """
        if not path_to_fast_sim:
            path_to_fast_sim = '/dependencies/fitted_raw_sim_interval_D=[6,15.2]_sigma=30%_del.npy'
        with open(path_to_fast_sim, 'rb') as data:
            simulations = np.load(data)
            simulation_targets = np.load(data,allow_pickle=True)
        simulation_targets = pd.DataFrame(simulation_targets,columns=['radius','sigma','distance','omega','2R/D'])
        detector_mask = self.crop_detector_mask()
        shape_to_bin = (int(math.ceil(0.5 * detector_mask.shape[0])), int(math.ceil(0.5 * detector_mask.shape[1])))
        detector_mask = self.crop_window(self.bin_mask(detector_mask, bin_to_shape=shape_to_bin))
        for i in range(len(simulations)):
            if self.add_noise: simulations[i] = self.add_poisson_shot_noise(simulations[i])
            if self.median: simulations[i] = self.median_filter(simulations[i])
            if self.gradient: simulations[i] = self.gradient_filter(simulations[i])
            if self.beta != None: simulations[i] = self.intensity_background_thresholding(simulations[i],beta=self.beta)
            if self.ROI != None: simulations[i] = self.crop_roi(simulations[i],self.ROI)
            simulations[i] = self.mask_image(simulations[i], mask=detector_mask)

        return simulations, simulation_targets

    def fit_simulation(self, simulation):
        """
        Adapt the simulated scattering images to the experimental scattering images
        :param: simulation the simulation class
        :return: list of fitted images (:rtype: list of ndarray)
        """
        fitted_images = []
        # load data
        images, targets = simulation.load_data()
        # start fitting
        n1, n2, two_theta_f_min, two_theta_f_max, alpha_f_min, alpha_f_max = simulation.get_grid_parameter(key='1')
        two_theta_f, alpha_f, q_y, q_z = self.convert_from_cartesian_to_reciprocal_space()
        two_theta_f_crop_index = int(round((two_theta_f.max()/(two_theta_f_max - two_theta_f_min)) * n2))
        alpha_f_crop_index = int(round((alpha_f.max()/(alpha_f_max - alpha_f_min)) * (n1-1)))
        # crop masks
        detector_mask = self.crop_detector_mask()
        shape_to_bin = (int(math.ceil(0.5 * detector_mask.shape[0])), int(math.ceil(0.5 * detector_mask.shape[1])))
        detector_mask = self.crop_window(self.bin_mask(detector_mask, bin_to_shape=shape_to_bin))
        for i, image in enumerate(images):
            # check if peak is valid if not drop data point
            if simulation.valid_peak:
                if not self.is_valid_peak(image, targets.distance[i]):
                    targets = targets.drop(i)
                    continue
            # crop
            image = self.crop_simulation(image=image, y=alpha_f_crop_index, x=two_theta_f_crop_index)
            # bin simulation image to experiment image
            image = self.bin_int(to_bin=image, bin_to_shape=shape_to_bin)
            # noise
            if self.add_noise: image = self.add_poisson_shot_noise(image=image)
            # crop window
            image = self.crop_window(image=image)
            # median filter
            if self.median: image = self.median_filter(image)
            # gradient filter
            if self.gradient: image = self.gradient_filter(image)
            # intensity and background thresholding
            if self.beta != None: image = self.intensity_background_thresholding(image,beta=self.beta)
            # crop module (ROI cut)
            if self.ROI != None: image = self.crop_roi(image,self.ROI)
            # mask
            image = self.mask_image(image=image, mask=detector_mask)
            # normalize
            image = self.normalize(image=image)
            # append images
            fitted_images.append(image)

        targets = targets.reset_index()

        return fitted_images, targets


    def fit_experiment_ready(self, experiment):
        """
        Prossece the fitted experiment images
        :param:
        :return: list of fitted images (:rtype: list of ndarray)
        """
        # load data
        experiment_images, experiment_target_values = experiment.load_data()

        shape_to_bin = (int(math.ceil(0.5 * experiment_images[0].shape[0])), int(math.ceil(0.5 * experiment_images[0].shape[1])))
        detector_mask = self.crop_detector_mask()
        detector_mask = self.crop_window(self.bin_mask(detector_mask, bin_to_shape=shape_to_bin))

        for i in range(len(experiment_images)):
            if self.median: experiment_images[i] = self.median_filter(experiment_images[i])
            if self.gradient: experiment_images[i] = self.gradient_filter(experiment_images[i])
            if self.beta != None: experiment_images[i] = self.intensity_background_thresholding(experiment_images[i],beta=self.beta)
            if self.ROI != None: experiment_images[i] = self.crop_roi(experiment_images[i],self.ROI)
            experiment_images[i] = self.mask_image(experiment_images[i],mask=detector_mask)

        return experiment_images, experiment_target_values

    def fit_experiment(self, experiment):
        """
        Adapt the experimental scattering images
        :return: list of fitted images (:rtype: list of ndarray)
        """
        # load data
        experiment_images, experiment_target_values = experiment.load_data()
        print(len(experiment_images))
        print(experiment_images[0].shape)
        # start fitting
        fitted_experiment_images = []
        # shape to bin
        shape_cropped_experiment = self.crop_experiment(image=experiment_images[0])
        shape_to_bin = (int(math.ceil(0.5 * shape_cropped_experiment.shape[0])), int(math.ceil(0.5 * shape_cropped_experiment.shape[1])))
        detector_mask = self.crop_detector_mask()
        detector_mask = self.crop_window(self.bin_mask(detector_mask, bin_to_shape=shape_to_bin))
        for image in experiment_images:
            # crop
            image = self.crop_experiment(image=image)
            # bin
            image = self.bin_float(to_bin=image, bin_to_shape=shape_to_bin)
            # crop window
            image = self.crop_window(image=image)
            # median filter
            if self.median: image = self.median_filter(image=image, median_smooth_factor=7)
            # gradient filter
            if self.gradient: image = self.gradient_filter(image=image)
            # intensity and background thresholding
            if self.beta != None: image = self.intensity_background_thresholding(image=image, beta=self.beta)
            # crop module (rigion of interest cut)
            if self.ROI != None: image = self.crop_roi(image=image, cut=self.ROI)
            # mask
            image = self.mask_image(image=image, mask=detector_mask)
            # normalize
            image = self.normalize(image=image)
            fitted_experiment_images.append(image)
        return fitted_experiment_images, experiment_target_values

    def fit_real(self, real):
        pass

    ####################################################
    # image processing functions - most will not be used
    ####################################################

    def add_poisson_shot_noise(self, image):
        """
        Photon shot noise arises from the random statistical fluctuations of photons colliding with a detector. Often,
        the photon source creates a Poisson distribution which means the probability of each photon arriving at a given
        time is independent of the other photons. Poisson shot noise can be translated into pixel intensity variations due to the particle distribution in the beam
        during the experiments. This function creates a noise mask through the Poisson process where the parameter of
        the Poisson distribution (lambda) is the pixel intensity value.

        :param image: image (:type ndarray)
        :return: noisy image
        """
        return util.random_noise(image=image, mode='poisson').astype(np.float32)

    def median_filter(self, image, median_smooth_factor=3):
        image = filters.rank.median(image, morphology.disk(radius=median_smooth_factor))
        return image

    def gradient_filter(self, image):
        image = filters.rank.gradient(image, morphology.disk(radius=1))
        return image

    def crop_roi(self, image, cut):
        if cut == 'bottom':
            return image[:88, :] # 1/3 image
        elif cut == 'middle':
            return image[98:194, :]
        elif cut == 'top':
            return image[98:, :]
        elif cut == 'all':
            return np.concatenate((image[:88, :], image[98:194, :], image[204:, :]), axis=0)
        else:
            return image


    def compute_background_level_and_intensity_threshold(self, intensities, beta):
        '''
        the following algorithm is based on the method described in the paper:
        "Active learning-assisted neutron spectroscopy with log-Gaussian processes"
        by Teixeira Parente, M., Brandl, G., Franz, C. et al.
        published in Nature, 2023. https://doi.org/10.1038/s41467-023-37418-8
        '''
        # sort initial intensity observations in ascending order
        sorted_intensities = np.sort(intensities)
        # determine deciles to divide the observations into 10 buckets
        deciles = np.percentile(sorted_intensities, [10 * i for i in range(11)])
        # create the buckets based on the deciles
        buckets = []
        for l in range(10):
            lower_bound = deciles[l]
            upper_bound = deciles[l + 1]
            bucket = sorted_intensities[(sorted_intensities > lower_bound) & (sorted_intensities <= upper_bound)]
            buckets.append(bucket)
        # compute the median of each bucket
        medians = np.array([np.median(bucket) for bucket in buckets])
        # compute relative and absolute differences of the bucket medians
        delta_rel = (medians[1:] - medians[:-1]) / medians[:-1]
        delta_abs = medians[1:] - medians[:-1]
        # thresholds for the relative and absolute difference between the medians of consecutive buckets
        # more on their impact in the description document
        delta_max_rel = 0.001
        delta_min_abs = 0.001
        # select the first bucket that meets the criteria for relative and absolute differences
        # median of selected bucket is the background level
        l_max = 9  # can choose other one (0 < l_max < 10)
        l_star = l_max
        for l in range(9):
            if delta_rel[l] > delta_max_rel and delta_abs[l] >= delta_min_abs:
                l_star = l
                break # 0 <= l_star <= 9
        # compute the background level (gamma)
        gamma = medians[l_star]
        # compute the intensity threshold (tau)
        # the difference between gamma (median of selected bucket) and the median of highest bucket
        # is multiplied with the beta factor (0 < beta < 1)
        # gamma is added to ensure that tau is anchored relative to the background level
        m10 = medians[-1]
        tau = gamma + beta * (m10 - gamma)
        # gamma + beta*m10 - beta*gamma = beta*m10 + gamma * (1 - beta)
        return gamma, tau

    def intensity_background_thresholding(self, image, beta, sample_perc=0.25, global_tau=False, global_gamma=False):
        if global_tau and global_gamma:
            if not self.tau and not self.gamma:
                step = int(1/sample_perc)
                intensities = image[1:image.shape[0]:step,1:image.shape[1]:step].flatten()
                self.gamma, self.tau = self.compute_background_level_and_intensity_threshold(intensities, beta)
            image[image < thresh_gama] = self.gamma
            image[image > thresh_tau] = self.tau

        else:
            step = int(1/sample_perc)
            intensities = image[1:image.shape[0]:step,1:image.shape[1]:step].flatten()
            gamma, tau = self.compute_background_level_and_intensity_threshold(intensities, beta)
            if global_gamma:
                if not self.gamma:
                    self.gamma = gamma
                gamma = self.gamma
            if global_tau:
                if not self.tau:
                    self.tau = tau
                tau = self.tau
            image[image < thresh_gama] = gamma
            image[image > thresh_tau] = tau

        return image


    def calculate_py_max(self, D, mod='raw_exp'):
        lamda = self.experiment_setup.get_wavelength()
        d_sd = self.experiment_setup.get_sample_detector_distance()
        db_x = self.experiment_setup.get_direct_beam_position()[0]
        px_y = self.detector.get_pixel2() * 1000 # pixel size in mm
        if mod=='raw_exp':
            py_max = d_sd * (np.tan(2*np.arcsin(lamda/(2*D)))/px_y) + db_x # coordinate offset
        elif mod=='raw_sim':
            py_max = d_sd * (np.tan(2*np.arcsin(lamda/(2*D)))/px_y) * 0.5 # cause simulation is half size
        elif mod=='fitted':
            offset = self.detector.get_maximum_shape()[1] - self.get_horizontal_cut_position()
            py_max = d_sd * (np.tan(2*np.arcsin(lamda/(2*D)))/px_y) +  offset # coordinate offset

        return py_max

    def find_intensity_center(self, image, offset=15):
        return np.argmax(np.sum(image[:,offset:],axis=0))+offset

    def is_valid_peak(self, image, distance, threshold=5):
        target_peak = self.calculate_py_max(distance, mod="raw_sim")
        is_peak = self.find_intensity_center(image)
        if abs(target_peak - is_peak) > threshold:
            return False
        else:
            return True
