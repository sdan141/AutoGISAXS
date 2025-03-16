import numpy as np
#import yaml
import json

def record_run_parameters(desc, run_path, morphology, training_scope, augmentation, model, labels, validation, 
                          estimation, n_training, n_batchs_and_epochs, output_units, loss, learning_rate, decay, informed, constrain, shuffled):
    global_run_logger = {"simulation_parameters":{},"preprocessing_parameters":{},
                        "algorithm_parameters":{},"architecture_parameters":{}}
    global_run_logger["simulation_parameters"] = {
					"scope":
						{
						"radius": training_scope["radius"],                   # dictionary
						"sigma_radius": training_scope["sigma_radius"],       # dictionary
						"distance": training_scope["distance"],               # dictionary
                        "omega_distance": training_scope["omega_distance"]    # dictionary
						},
					"percolation_thresh": constrain,                               # bool (radius distance bounds used)
					"valide_peak": constrain                                       # bool (only simulations with valid peak py_max ~ 2pi/D used)
					}
    print(f"learning_rate: {learning_rate.numpy()}, decay: {decay}")
    global_run_logger["preprocessing_parameters"] = {
					"beta": augmentation.beta,
					"ROI": augmentation.ROI,
					"noise": augmentation.add_noise,
					"median": augmentation.median,
					"gradient": augmentation.gradient
					}
    
    global_run_logger["algorithm_parameters"] = {
        			"run_desc": desc,
					"label_model": labels,
					"validation": validation,
					"estimation": estimation,
                    "informed": informed,
                    "shuffle": shuffled,
					"output_units":
							{
                            "radius": output_units["radius"],
                            "sigma_radius": output_units["sigma"],
                            "distance": output_units["distance"],
                            "omega_distance": output_units["omega"]
							},
					"n_training":n_training,
					}
    
    global_run_logger["architecture_parameters"] = {
					"optimizer":
						{
						"optimizer": "ADAM",
						"learning_rate": np.round(float(learning_rate.numpy()),7),
						"decay": decay
                        },
					"loss": loss,
						# {
                        # ["radius"]: loss["radius"],
                        # ["sigma_radius"]: loss["sigma"],
                        # ["distance"]: loss["distance"],
                        # ["omega_distance"]: loss["omega"]
						# },
					"n_epochs": n_batchs_and_epochs.split('_')[2],
					"batch_size": n_batchs_and_epochs.split('_')[1],
					"model_name": model
					}
    # with open(run_path + "/config.yaml", "w") as file:
    #     yaml.dump(global_run_logger, file)

    with open(run_path + "/config.json", mode="w", encoding="utf-8") as write_file:
        json.dump(global_run_logger, write_file, indent=4)

def get_param_lognorm(m, v):
    sigma_squared = np.log(1 + (v / m**2))
    sigma = np.sqrt(sigma_squared)
    mu = np.log(m) - (sigma_squared / 2)
    return mu, sigma

def lognorm_func(x, s, loc, scale=1):
    return scale/(s*x*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-loc)**2/(2*s**2))



def expand2d(vect, size2, vertical=True):
    """
    This expands a vector to a 2d-array.
    # courtesy of E. Almamedov with modifications for our purposes

    The result is the same as:

    .. code-block:: python

        if vertical:
            numpy.outer(numpy.ones(size2), vect)
        else:
            numpy.outer(vect, numpy.ones(size2))

    This is a ninja optimization: replace \\*1 with a memcopy, saves 50% of
    time at the ms level.

    :param vect: 1d vector
    :param size2: size of the expanded dimension
    :param vertical: if False the vector is expanded to the first dimension.
        If True, it is expanded to the second dimension.
    """
    size1 = vect.size
    size2 = int(size2)
    if vertical:
        out = np.empty((size2, size1), vect.dtype)
        q = vect.reshape(1, -1)
        q.strides = 0, vect.strides[0]
    else:
        out = np.empty((size1, size2), vect.dtype)
        q = vect.reshape(-1, 1)
        q.strides = vect.strides[0], 0
    out[:, :] = q
    return out
