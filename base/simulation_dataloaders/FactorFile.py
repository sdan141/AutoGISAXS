# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy
import h5py
import pandas
#import tqdm
import math


class FactorFile:
    def __init__(self, path, file, id):
        self.id = id
        #self.database = h5py.File(path + file, 'r')
        self.structure_factor_target_values = pandas.read_csv(path + file['target_value_files'][0]).rename(columns={'omega/D':'omega_distance'}).assign(id=0)
        self.form_factor_target_values = pandas.read_csv(path + file['target_value_files'][1]).rename(columns={'sigma':'sigma_radius'}).assign(id=0)
        self.structure_factor_target_values.key = self.structure_factor_target_values.index.to_numpy()
        self.form_factor_target_values.key = self.form_factor_target_values.index.to_numpy()
        with numpy.load(isGISAXS_data,'r',allow_pickle=True) as data:
            self.form_factors = data['form_factors_ordered']
            self.structure_factors = data['structure_factors_ordered']

        key_sf = self.structure_factor_target_values.key.to_numpy()
        key_ff = self.form_factor_target_values.key.to_numpy()
        self.keys = numpy.concatenate((key_ff, key_sf))
        self.length = len(self.keys)

    def get_id(self):
        return self.id

    def get_keys(self):
        return self.keys

    def get_mff1(self, key):
        return self.form_factors[key]

    def get_structure(self, key):
        return self.structure_factors[key]

    def get_structure_factor(self, key):
        distance = self.structure_factor_target_values[structure_factor_target_values.key==key].distance.iloc[0]
        omega_distance = self.structure_factor_target_values[structure_factor_target_values.key==key].omega_distance.iloc[0]
        return round(distance,1), round(omega_distance,2)

    def get_form_factor(self, key):
        radius = self.form_factor_target_values[form_factor_target_values.key==key].radius.iloc[0]
        sigma_radius = self.form_factor_target_values[form_factor_target_values.key==key].sigma_radius.iloc[0]
        return round(radius,1), round(sigma_radius,2)

    def get_beam_setup(self, key):
        lambda_b = 0.096
        alpha_i = 0.395
        layer_thickness = 1
        return lambda_b, alpha_i, layer_thickness

    def get_grid_parameter(self, key):
        n1 = 500
        n2 = 500
        two_theta_min = 0.0
        two_theta_max = 4.0
        alphaf_min = 0.0
        alphaf_max = 4.0
        return n1, n2, two_theta_min, two_theta_max, alphaf_min, alphaf_max

    def get_structure_factors(self):
        return self.structure_factors, self.structure_factor_target_values

    def get_form_factors(self):
        return self.form_factors, self.form_factor_target_values

    def get_target_values(self):
        target_values = pandas.DataFrame(columns=['key', 'distance', 'omega_distance', 'radius', 'sigma_radius'])
        for key in self.keys:
            distance, omega_distance = self.get_structure_factor(key)
            radius, sigma_radius = self.get_form_factor(key)
            row = {'key': key, 'distance': distance, 'omega_distance': omega_distance, 'radius': radius, 'sigma_radius': sigma_radius}
            target_values = target_values.append(row, ignore_index=True)
        return target_values

    def get_images(self):
        images = []
        for key in self.keys:
            images.append(self.get_gisaxs(key))
        return images
