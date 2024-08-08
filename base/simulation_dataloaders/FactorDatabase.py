# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy
import h5py
import pandas
#import tqdm
import math

class FactorDatabase:
    def __init__(self, path, file, id):
        self.id = id
        self.database = h5py.File(path + file, 'r')
        self.keys = list(self.database.keys())
        self.length = len(self.keys)

    def get_id(self):
        return self.id

    def get_keys(self):
        return self.keys

    def get_gisaxs(self, key):
        return numpy.asarray(self.database[key]['gisaxs'], dtype=numpy.float32)[1:]

    def get_mff1(self, key):
        return numpy.asarray(self.database[key]['mff1'], dtype=numpy.float32)[1:]

    def get_mff2(self, key):
        return numpy.asarray(self.database[key]['mff2'], dtype=numpy.float32)[1:]

    def get_structure(self, key):
        return numpy.asarray(self.database[key]['structure'], dtype=numpy.float32)[1:]

    def get_structure_factor(self, key):
        group = self.database[key]['parameters']['particle_position_parameters']
        particle_distribution = group.attrs['Distribution_of_particles']
        # mean distance between two particles (peak position)
        distance = round(group.attrs['Correlation_function_D'], 2)
        # gaussian distribution of the mean distances between two particles
        omega_distance = round(group.attrs['Correlation_function_w'] / distance, 2)
        return distance, omega_distance

    def get_form_factor(self, key):
        group = self.database[key]['parameters']['particle_parameters']['particle_1']
        # mean radius of the particle
        radius = round(group.attrs['R0_(nm)'], 2)
        # standard derivation of the mean radius of the particle
        sigma_radius = round(group.attrs['sigR/R0'], 2)
        return radius, sigma_radius

    def get_beam_setup(self, key):
        lambda_b = self.database[key]['parameters']['beam_and_substrate_parameters'].attrs['Wl0_(nm)']
        alpha_i = self.database[key]['parameters']['beam_and_substrate_parameters'].attrs['Alphai_(deg)']
        layer_thickness = self.database[key]['parameters']['beam_and_substrate_parameters'].attrs['Layer_thickness_(nm)']
        return lambda_b, alpha_i, layer_thickness

    def get_grid_parameter(self, key):
        group = self.database[key]['parameters']['main']
        n1 = group.attrs['n1']
        n2 = group.attrs['n2']
        two_theta_min = group.attrs['2_theta_min_(deg)']
        two_theta_max = group.attrs['2_theta_max_(deg)']
        alphaf_min = group.attrs['alphaf_min_(deg)']
        alphaf_max = group.attrs['alphaf_max_(deg)']
        return n1, n2, two_theta_min, two_theta_max, alphaf_min, alphaf_max

    def get_structure_factors(self):
        images = []
        target_values = pandas.DataFrame(columns=['id', 'key', 'distance', 'omega_distance'])
        for key in self.keys:
            images.append(self.get_structure(key))
            distance, omega_distance = self.get_structure_factor(key)
            # target_values = target_values.append(
            #     {'id': self.id, 'key': key, 'distance': distance, 'omega_distance': omega_distance},
            #     ignore_index=True)
            target_values = pandas.concat(
                [target_values, pandas.DataFrame([{'id': self.id, 'key': key, 'distance': distance, 'omega_distance': omega_distance}])],
                ignore_index=True)
        return images, target_values

    def get_form_factors(self):
        images = []
        target_values = pandas.DataFrame(columns=['id', 'key', 'radius', 'sigma_radius'])
        for key in self.keys:
            images.append(self.get_mff1(key))
            radius, sigma_radius = self.get_form_factor(key)
            #target_values = target_values.append({'id': self.id, 'key': key, 'radius': radius, 'sigma_radius': sigma_radius}, ignore_index=True)
            target_values = pandas.concat([target_values, pandas.DataFrame([{'id': self.id, 'key': key, 'radius': radius, 'sigma_radius': sigma_radius}])], ignore_index=True)

        return images, target_values

    def get_target_values(self):
        target_values = pandas.DataFrame(columns=['key', 'distance', 'omega_distance', 'radius', 'sigma_radius'])
        for key in self.keys:
            distance, omega_distance = self.get_structure_factor(key)
            radius, sigma_radius = self.get_form_factor(key)
            row = {'key': key, 'distance': distance, 'omega_distance': omega_distance, 'radius': radius, 'sigma_radius': sigma_radius}
            #target_values = target_values.append(row, ignore_index=True)
            target_values = pandas.concat([target_values, pandas.DataFrame([row])], ignore_index=True)
        return target_values

    def get_images(self):
        images = []
        for key in self.keys:
            images.append(self.get_gisaxs(key))
        return images
