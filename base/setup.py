# courtesy of E. Almamedov with modifications for our purposes

import fabio
import numpy as np

class Experiment:

    def __init__(self, path_project, experiment_parameter, detector, experiment_maskfile):
        self.path_project = path_project
        self.materials = experiment_parameter['materials']
        self.wavelength = experiment_parameter['wavelength']
        self.incidence_angle = experiment_parameter['incidence_angle']
        self.direct_beam_position = experiment_parameter['direct_beam_position']
        self.sample_detector_distance = experiment_parameter['sample_detector_distance']
        self.detector = detector
        self.experiment_maskfile = experiment_maskfile

    def get_materials(self):
        return self.materials

    def get_wavelength(self):
        return self.wavelength

    def get_incidence_angle(self):
        return self.incidence_angle

    def get_direct_beam_position(self):
        return self.direct_beam_position

    def get_sample_detector_distance(self):
        return self.sample_detector_distance

    def get_detector(self):
        return self.detector

    def get_experiment_maskfile(self):
        experiment_mask = fabio.open(self.path_project + 'base/masks/' + self.experiment_maskfile)
        return experiment_mask.data

