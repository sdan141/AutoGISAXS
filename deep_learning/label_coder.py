import numpy as np
from base import utilities as utils
from scipy.stats import lognorm, norm

class LabelCoder:

    def __init__(self, output_units, morphology='all', distr=False):

        self.mode = "one-hot" #"lognorm" #   "normal"

        self.n_radius_units = output_units['radius']
        self.n_distance_units = output_units['distance']
        self.n_sigma_units = output_units['sigma']
        self.n_omega_units = output_units['omega']

        # self.radius = np.arange(1.2, 1.2+self.n_radius_units*0.1, 0.1)
        # self.distance = np.arange(3.8, 3.8+self.n_distance_units*0.1, 0.1)
        # self.sigma_radius = np.arange(0.1, 0.1+self.n_sigma_units*0.01, 0.01)
        # self.omega_distance= np.arange(0.1, 0.1+self.n_omega_units*0.01, 0.01)
        self.radius = np.round(np.linspace(1.2, 28.1, self.n_radius_units),1)
        self.distance = np.round(np.linspace(3.8, 40.8, self.n_distance_units),1)
        self.sigma_radius = np.round(np.linspace(0.01, 0.75, self.n_sigma_units),2)
        self.omega_distance= np.round(np.linspace(0.01, 0.75, self.n_omega_units),2)        

        self.prob_space = {'radius':self.radius, 'distance': self.distance, 'sigma_radius': self.sigma_radius, 'omega_radius': self.omega_distance}

        print(f"Probability Space for predictions: {self.prob_space}")

    def create_labels(self, target_values, key, sigmas=None):
        '''
        parameters
        returns
        '''
        labels = []
        for i, target in enumerate(target_values):
             label = self.get_label(target, key, sigmas[i] if sigmas is not None else None)
             labels.append(label)    
        return np.array(labels)

    def get_label(self, target, key, sigma):
        if '_' in key:
            label = [1 if round(target,2)==round(param,2) else 0 for param in self.prob_space[key]]
        else: 
            if self.mode == "lognorm" and sigma:
                mu, s = utils.get_param_lognorm(m=target, v=sigma) 
                label = utils.lognorm_func(self.prob_space[key], s, loc=mu)
                label /= np.sum(label)

            if self.mode == "norm" and sigma:
                label = norm.pdf(self.prob_space[key], loc=target, scale=sigma)
                label /= np.sum(label)

            else:
                label = [1 if round(target,1)==round(param,1) else 0 for param in self.prob_space[key]]

        assert np.round(np.sum(label.copy()),9) >= 1, f"One-hot label sum smaller than one: {sum(label)}, target: {round(target,2)} \n prob_space = {np.round(self.prob_space[key],2)}"
        assert np.round(np.sum(label.copy()),9) <= 1, f"One-hot label sum larger than one: {sum(label)}, target: {round(target,2)} \n prob_space = {np.round(self.prob_space[key],2)}"
        return np.array(label, dtype=np.float64)
    
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
        value = np.sum(self.prob_space[key]*label)
        return value
