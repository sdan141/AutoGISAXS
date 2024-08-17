
import numpy as np

class LabelCoder:

    def __init__(self, output_units, morphology='all', distr=False):
        self.n_radius_units = output_units['radius']
        self.n_distance_units = output_units['distance']
        self.n_sigma_units = output_units['sigma']
        self.n_omega_units = output_units['omega']

        self.radius = np.arange(1.2, 1.2+self.n_radius_units*0.1, 0.1)
        self.distance = np.arange(3.8, 3.8+self.n_distance_units*0.1, 0.1)
        self.sigma_radius = np.arange(0.1, 0.1+self.n_sigma_units*0.01, 0.01)
        self.omega_distance= np.arange(0.1, 0.1+self.n_omega_units*0.01, 0.01)

        self.prob_space = {'radius':self.radius, 'distance': self.distance, 'sigma_radius': self.sigma_radius, 'omega_radius': self.omega_distance}
        '''
        if distr else ['radius', 'distance']
        if morphology != 'all':
            self.keys = [key in self.keys if morphology in key]
        '''
    def create_labels(self, target_values, key):
        '''
        parameters
        returns
        '''
        labels = []
        for target in target_values:
             labels.append(self.get_label(target, key))
        return labels

    def get_label(self, target, key):
        if '_' in self.prob_space[key]:
            label = [1 if round(target,2)==round(param,2) else 0 for param in self.prob_space[key]]
        else: 
            label = [1 if round(target,1)==round(param,1) else 0 for param in self.prob_space[key]]
        assert sum(label) >= 1, f"One-hot label sum smaller than one: {sum(label)}, target: {round(target,2)} \n prob_space = {np.round(self.prob_space[key],2)}"
        assert sum(label) <= 1, f"One-hot label sum larger than one: {sum(label)}, target: {round(target,2)} \n prob_space = {np.round(self.prob_space[key],2)}"
        return label
    
    def coldbatch_maximum(self, labels, key):
        values = []
        for label in labels:
            values.append(self.get_value_max(label, key))
        return values

    def coldbatch_average(self, labels, key):
        values = []
        for label in labels:
            values.append(self.get_value_avg(label, key))
        return values

    def get_value_max(self, label, key):
        value = self.prob_space[key][np.argmax(label)]
        return value

    def get_value_avg(self, label, key):
        value = sum(self.prob_space[key]*label)
        return value
