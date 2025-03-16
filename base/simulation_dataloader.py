# courtesy of E. Almamedov with modifications for our purposes

import numpy as np
import pandas as pd
from base.simulation_dataloaders import FactorDatabase, ReadyDatabase, FactorFile
from base import utilities

# intervals for trainning scope (for simulations)
training_interval = {
    'radius':
            {'all': {'start': 0.5, 'stop': 7.6, 'step': 0.1}, 'low': {'start': 0.5, 'stop': 4.6, 'step': 0.1}, 'medium': {'start': 1.2, 'stop': 7.6, 'step': 0.1},
                'high': {'start': 2.5, 'stop': 7.6, 'step': 0.1}, 'stepsize': {'start': 1.0, 'stop': 8.0, 'step': 1.0}, 'costume':{'start': 1.2, 'stop': 7.6, 'step': 0.1},
                'test':{'start': 2.5, 'stop': 5.1, 'step': 0.1}},
    'distance':
            {'all': {'start': 0.5, 'stop': 7.6, 'step': 0.1}, 'low': {'start': 0.5, 'stop': 4.6, 'step': 0.1}, 'medium': {'start': 6.0, 'stop': 15.1, 'step': 0.1},
                'high': {'start': 2.5, 'stop': 7.6, 'step': 0.1}, 'stepsize': {'start': 3.0, 'stop': 16.0, 'step': 1.0}, 'costume':{'start': 6.0, 'stop': 15.1, 'step': 0.1},
                'test':{'start': 10.0, 'stop': 10.4, 'step': 0.1}},
    'sigma_radius':
            {'all': {'start': 0.1, 'stop': 0.51, 'step': 0.01}, 'low': {'start': 0.1, 'stop': 0.25, 'step': 0.01}, 'medium': {'start': 0.18, 'stop': 0.33, 'step': 0.01},
                'high': {'start': 0.33, 'stop': 0.51, 'step': 0.01}, 'stepsize': {'start': 0.1, 'stop': 0.51, 'step': 0.1}, 'costume': {'start': 0.1, 'stop': 0.35, 'step': 0.05},
                'test':{'start': 0.3, 'stop': 0.31, 'step': 0.01}},
    'omega_distance':
            {'all': {'start': 0.1, 'stop': 0.51, 'step': 0.01}, 'low': {'start': 0.1, 'stop': 0.25, 'step': 0.01}, 'medium': {'start': 0.00, 'stop': 0.51, 'step': 0.05},
                'high': {'start': 0.33, 'stop': 0.51, 'step': 0.01}, 'stepsize': {'start': 0.1, 'stop': 0.51, 'step': 0.1}, 'costume': {'start': 0.16, 'stop': 0.31, 'step': 0.01},
                'test':{'start': 0.2, 'stop': 0.21, 'step': 0.01}}
                }

def Database(sim_source, path, file, id):
    if sim_source=='factors_h5':
        return FactorDatabase.FactorDatabase(path, file, id)
    elif sim_source=='ready_h5':
        return ReadyDatabase.ReadyDatabase(path, file, id)
    elif sim_source=='factors_file':
        return FactorFile.FactorFile(path, file, id)
    else:
        print('\nInvalid simulation source argument\n')


class Simulation:

    def __init__(self, path, constrains, intervals, sim_source):
        self.databases = []
        for i in range(0, len(path['files'])):
            self.databases.append(Database(sim_source=sim_source, path=path['path'], file=path['files'][i], id=i))
        self.samples = constrains['samples']
        self.distance_interval = intervals['distance'] # {constraint['interval']['distance']['costume'], 'costume'}
        self.radius_interval = intervals['radius']# (1.2, 7.5)
        self.omega_interval = intervals['omega_distance'] # (0.19, 0.27)
        self.sigma_interval = intervals['sigma_radius'] # (0.30, 0.32)
        # self.fast_simulation = constraint['fast_sim']
        self.percolation_thresh = constrains['constrain']#True # False #bool if to constraint percolation threshold 1/2 <= 2R/D <= 1
        self.valid_peak = constrains['constrain']#True # False #constraint['peak'] # bool if to constrain the distance spreading D ~ 2pi/q_o
        self.sim_source = sim_source
        #utilities.record_simulation_parameters(self)


    def get_databases(self):
        return self.databases

    def get_keys(self):
        keys = []
        for database in self.databases:
            keys.extend(database.get_keys())
        return keys

    def key_is_in_database(self, key, database):
        if key in database.get_keys():
            return True
        else:
            return False

    def get_gisaxs(self, key):
        for database in self.databases:
            if self.key_is_in_database(key, database):
                return database.get_gisaxs(key)

    def get_mff1(self, key):
        for database in self.databases:
            if self.key_is_in_database(key, database):
                return database.get_mff1(key)

    def get_mff2(self, key):
        for database in self.databases:
            if self.key_is_in_database(key, database):
                return database.get_mff2(key)

    def get_structure(self, key):
        for database in self.databases:
            if self.key_is_in_database(key, database):
                return database.get_structure(key)

    def get_grid_parameter(self, key):
        for database in self.databases:
            if self.key_is_in_database(key, database):
                return database.get_grid_parameter(key)


    # delete: for test_on_experiment cases
    def get_target_values(self):
        dataframes = []
        for database in self.databases:
            dataframes.append(database.get_target_values())
        target_values = pd.concat(dataframes, ignore_index=True)
        return target_values

    # delete: for test_on_experiment cases
    def get_images(self):
        images = []
        for database in self.databases:
            images.extend(database.get_images())
        return images

    def load_data_ready(self):
        images = []
        targets = self.get_target_values()
        targets = targets.sort_values(by=['radius','distance','sigma_radius','omega_distance'])
        # filter structure factors
        if self.distance_interval['step']==0.2:
            targets = targets[(targets.distance >= self.distance_interval['start']) & \
                               (targets.distance < self.distance_interval['stop'])]
        else:
            distance_interval = np.round(np.arange(self.distance_interval),1)
            targets = targets[targets.round({'distance':1}).distance.isin(distance_interval)]

        if self.omega_interval['step']==0.01:
            targets = targets[(targets.omega_distance >= self.omega_interval['start']) & \
                               (targets.omega_distance < self.omega_interval['stop'])]
        else:
            omega_interval = np.round(np.arange(self.omega_interval),2)
            targets = targets[targets.round({'omega_distance':2}).omega_distance.isin(omega_interval)]
        # filter form factors
        if self.radius_interval['step']==0.2:
            targets = targets[(targets.radius >= self.radius_interval['start']) & \
                               (targets.radius < self.radius_interval['stop'])]
        else:
            radius_interval = np.round(np.arange(self.radius_interval),1)
            targets = targets[targets.round({'radius':1}).radius.isin(radius_interval)]

        if self.sigma_interval['step']==0.01:
            targets = targets[(targets.sigma_radius >= self.sigma_interval['start']) & \
                              (targets.sigma_radius < self.sigma_interval['stop'])]
        else:
            sigma_interval = np.round(np.arange(self.sigma_interval),2)
            targets = targets[targets.round({'sigma_radius':2}).sigma_radius.isin(sigma_interval)]

        if self.samples:
            targets = targets.sample(frac=self.samples).sort_values(by=['radius','distance','sigma_radius','omega_distance'])

        # iterate over targets
        # get the image by respective key save used targets and images
        for tar_index in range(0, len(targets)):
            if self.percolation_thresh:
                D0 = targets[tar_index][2]
                if targets[tar_index][1] > D0/2 or targets[tar_index][1] < D0/4:
                    targets = targets[~tar_index]
                    continue
                value = {
                   'id': targets[index][0],
                   'key': targets[index][1],          
                   'distance': targets[sf_index][2],        # distance depends only on sf
                   'omega_distance': structure_factor_target_values[sf_index][3],  # omega of distance depends only on sf
                   'radius': form_factor_target_values[ff_index][2],               # radius depends only on ff
                   'sigma_radius': form_factor_target_values[ff_index][3]          # sigma of radius depends only on ff
                }
                sf_image = self.databases[value['id_sf']].get_structure(value['key_sf'])
                ff_image = self.databases[value['id_ff']].get_mff1(value['key_ff'])
                gisaxs = np.round(sf_image * ff_image, 5)
                targets.append(value)
                images.append(gisaxs)
        target_values = pd.DataFrame(targets)

        return images, targets

    def get_index_of_key(self, key, target_values):
        """
        Images are stored in list and to access them it is necessary to convert the unique key into an index using the
        target value dataframe
        """
        index = target_values.index[target_values['key'] == key].tolist()
        return index[0]

    def get_structure_factors(self):
        all_images = []
        all_targets = []
        for database in self.databases:
            images, target_values = database.get_structure_factors()
            all_images.extend(images)
            all_targets.append(target_values)
        all_target_values = pd.concat(all_targets, ignore_index=True)
        return all_images, all_target_values

    def get_form_factors(self):
        all_images = []
        all_targets = []
        for database in self.databases:
            images, target_values = database.get_form_factors()
            all_images.extend(images)
            all_targets.append(target_values)
        all_target_values = pd.concat(all_targets, ignore_index=True)
        return all_images, all_target_values


    def load_data(self):
        if 'ready' in self.sim_source:
            return self.load_data_ready()

        images = []; targets = []
        # get structure and form factors
        structure_factors, structure_factor_target_values = self.get_structure_factors()
        form_factors, form_factor_target_values = self.get_form_factors()
        # sort data
        structure_factor_target_values = structure_factor_target_values.sort_values(by=['distance','omega_distance'])
        form_factor_target_values = form_factor_target_values.sort_values(by=['radius', 'sigma_radius'])
        # filter structure factors
        if self.distance_interval['step']==0.2:
            structure_factor_target_values = structure_factor_target_values[(structure_factor_target_values.distance >= self.distance_interval['start']) & \
                                                                            (structure_factor_target_values.distance < self.distance_interval['stop'])]
        else:
            distance_interval = np.round(np.arange(*self.distance_interval.values()),1)
            structure_factor_target_values = structure_factor_target_values[structure_factor_target_values.round({'distance':1}).distance.isin(distance_interval)]

        if self.omega_interval['step']==0.01:
            structure_factor_target_values = structure_factor_target_values[(structure_factor_target_values.omega_distance >= self.omega_interval['start']) & \
                                                                            (structure_factor_target_values.omega_distance < self.omega_interval['stop'])]
        else:
            omega_interval = np.round(np.arange(*self.omega_interval.values()),2)
            structure_factor_target_values = structure_factor_target_values[structure_factor_target_values.round({'omega_distance':2}).omega_distance.isin(omega_interval)]
        # filter form factors
        if self.radius_interval['step']==0.2:
            form_factor_target_values = form_factor_target_values[(form_factor_target_values.radius >= self.radius_interval['start']) & \
                                                                            (form_factor_target_values.radius < self.radius_interval['stop'])]
        else:
            radius_interval = np.round(np.arange(*self.radius_interval.values()),1)
            form_factor_target_values = form_factor_target_values[form_factor_target_values.round({'radius':1}).radius.isin(radius_interval)]

        if self.sigma_interval['step']==0.01:
            form_factor_target_values = form_factor_target_values[(form_factor_target_values.sigma_radius >= self.sigma_interval['start']) & \
                                                                            (form_factor_target_values.sigma_radius < self.sigma_interval['stop'])]
        else:
            sigma_interval = np.round(np.arange(*self.sigma_interval.values()),2)
            form_factor_target_values = form_factor_target_values[form_factor_target_values.round({'sigma_radius':2}).sigma_radius.isin(sigma_interval)]

        # sample data database
        if self.samples:
            structure_factor_target_values = structure_factor_target_values.sample(frac=self.samples).sort_values(by=['distance','omega_distance'])
            form_factor_target_values = form_factor_target_values.sample(frac=self.samples).sort_values(by=['radius', 'sigma_radius'])
        # convert to list because its faster
        structure_factor_target_values = structure_factor_target_values.values.tolist()
        form_factor_target_values = form_factor_target_values.values.tolist()
        # iterate over structure factor and form factor
        for sf_index in range(0, len(structure_factor_target_values)):
            for ff_index in range(0, len(form_factor_target_values)):
                if self.percolation_thresh:
                    D0 = structure_factor_target_values[sf_index][2]
                    if form_factor_target_values[ff_index][2] > D0/2 or form_factor_target_values[ff_index][2] < D0/4:
                        continue
                value = {
                   'id_sf': structure_factor_target_values[sf_index][0],
                   'id_ff': form_factor_target_values[ff_index][0],
                   'key_sf': structure_factor_target_values[sf_index][1],          # key of sf
                   'key_ff': form_factor_target_values[ff_index][1],               # key of ff
                   'distance': structure_factor_target_values[sf_index][2],        # distance depends only on sf
                   'omega_distance': structure_factor_target_values[sf_index][3],  # omega of distance depends only on sf
                   'radius': form_factor_target_values[ff_index][2],               # radius depends only on ff
                   'sigma_radius': form_factor_target_values[ff_index][3]          # sigma of radius depends only on ff
                }
                sf_image = self.databases[value['id_sf']].get_structure(value['key_sf'])
                ff_image = self.databases[value['id_ff']].get_mff1(value['key_ff'])
                gisaxs = np.round(sf_image * ff_image, 5)
                targets.append(value)
                images.append(gisaxs)
        target_values = pd.DataFrame(targets)
        target_values = target_values.sort_values(by=['radius','distance','sigma_radius','omega_distance'])
        images = np.array(images)[target_values.index] 
        target_values.reset_index(drop=True, inplace=True)

        return images, target_values
