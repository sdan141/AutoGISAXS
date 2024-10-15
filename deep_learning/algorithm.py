import numpy
import pandas
import os
#import tensorflow as tf
from tensorflow.keras import models, utils, callbacks, optimizers, losses, metrics, Model
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from deep_learning import label_coder
#from network_models import MLPNet, CNNNet
from glob import glob
import re
from datetime import date
from .costume_loss import monotonic_categorical_crossentropy, monotonic_mse, monotonic_kl_divergence


MAX_VALIDATION_ROUND = 2

def generate_ordered_train_test_indices(X, max_splits=10, test_size=0.1):
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    indices = numpy.arange(num_samples)
    train_test_indices = []
    for _ in range(max_splits):
        # randomly choose test indices, and sort them
        test_indices = numpy.sort(numpy.random.choice(indices, num_test_samples, replace=False))
        # train indices are all other indices not in test indices
        train_indices = numpy.setdiff1d(indices, test_indices)
        train_test_indices.append((train_indices, test_indices))
    
    return train_test_indices

def generate_train_test_indices(X, max_splits=10, test_size=0.1):
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    indices = numpy.arange(num_samples)
    numpy.random.seed(42) 
    numpy.random.shuffle(indices)
    train_test_indices = []
    for i in range(max_splits):
        if i * num_test_samples >= num_samples:
            break
        test_indices = indices[i * num_test_samples: (i + 1) * num_test_samples]
        train_indices = numpy.setdiff1d(indices, test_indices)
        train_test_indices.append((train_indices, test_indices))

    return train_test_indices

def generate_train_test_splits(X, y, max_splits=10, test_size=0.2):
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    if num_test_samples == 0:
        num_test_samples = 1  # ensure at least one sample in the test set
    indices = numpy.arange(num_samples)
    numpy.random.seed(42)  # for reproducibility
    numpy.random.shuffle(indices)
    splits = []
    for i in range(max_splits):
        if i * num_test_samples >= num_samples:
            break
        test_indices = indices[i * num_test_samples: (i + 1) * num_test_samples]
        train_indices = numpy.setdiff1d(indices, test_indices)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        splits.append((X_train, X_test, y_train, y_test))
    return splits

class AlgorithmBase:
    TYPE = None

    def __init__(self, model, parameter, morphology, output_units):
        self.model = model
        self.parameter = parameter
        self.model_compiled = False
        self.keys = morphology
        self.label_coder = label_coder.LabelCoder(output_units)
        # set path to model
        dt = date.today()
        morphology = ''
        for key in self.keys:
            morphology += key + '_'
        morphology = morphology[:-1]
        self.model_path = f"{parameter['path']}/results/{morphology}/{self.TYPE.lower()}" # middle
        self.record_file = self.model_path + '/model_parameters_archive'
        if not os.path.exists(self.model_path):
            self.model_path += '/run_0' 
            os.makedirs(self.model_path)
        else:
            folders = sorted(os.listdir(self.model_path))
            newest = max([int(folder.split('_')[1]) for folder in folders if folder.startswith('run_')])
            self.model_path += '/run_' + str(newest + 1)
            if self.check: self.model_path += '_test'
            os.makedirs(self.model_path)

    def reset_weights(self):
        '''
        re-randomalize the model's weights
        this functionality is not yet part of tensorflow but it might be in the future!
        this solution is taken from the thread: https://github.com/keras-team/keras/issues/341
        *NOTE* This will not work with every layer type! (e.g. LTSM)
        '''
        for layer in self.model.layers:
            if isinstance(layer, Model): #if you're using a model as a layer
                self.reset_weights(layer) #apply function recursively
                continue
            #where are the initializers?
            if hasattr(layer, 'cell'):
                init_container = layer.cell
            else:
                init_container = layer

            for key, initializer in init_container.__dict__.items():
                if "initializer" not in key: #is this item an initializer?
                      continue #if no, skip it
                # find the corresponding variable, like the kernel or the bias
                if key == 'recurrent_initializer': #special case check
                    var = getattr(init_container, 'recurrent_kernel')
                else:
                    var = getattr(init_container, key.replace("_initializer", ""))
                if var is not None:
                    var.assign(initializer(var.shape, var.dtype))
                    #use the initializer

    def create_input_tensor(self, images):
        if type(images) is not list:
            images = list(images)
        # reshape x to (len(images), height, width, 1)
        images = numpy.reshape(images, (len(images),) + images[0].shape + (1,)).astype(numpy.float32)
        return images

    def generate_labels(self, target_values, keys=None):
        labels = {}
        if not keys: keys = self.keys
        for key in keys:
            # NOTE: label_coder should return numpy arrays!
            if self.label_coder.mode != "one-hot":
                dist_key = next((dist for dist in target_values if "_" + key in dist), None)
                sigmas = target_values[dist_key] if dist_key else None
            else: sigmas = None
            labels[key] = self.label_coder.create_labels(target_values[key], key, sigmas)
        return labels
    
    def set_validation_data(self, validation_data):
        if 'exp' in self.parameter['validation']:
            images_validation, targets_validation = validation_data
            images_validation = self.create_input_tensor(images_validation)
            if '_' not in '\t'.join(self.keys):
                self.keys_validation = self.keys
                labels_validation = self.generate_labels(targets_validation)
            else:
                keys = [key for key in self.keys if '_' not in key]
                self.keys_validation = keys
                labels_validation = self.generate_labels(targets_validation, keys=keys)
        else:
            pass #TODO if sim:
            # split training data to train and validation
        return images_validation, targets_validation, labels_validation
    
    def train_on_simulations_validate_with_experiment(self, images, target_values, validation_data):

        # split simulation data to training and test
        train_test_splits = generate_ordered_train_test_indices(X=images, max_splits=1)
        images = numpy.array(images)
        images_train = images[train_test_splits[0][0].astype(int)]
        images_test = images[train_test_splits[0][1].astype(int)]

        # extract training and test simulation targets
        target_values_train = target_values.loc[train_test_splits[0][0]].reset_index(drop=True)
        target_values_test = target_values.loc[train_test_splits[0][1]].reset_index(drop=True)        
        target_values_test.to_csv(self.model_path + '/' + 'simulation_test_targets.csv')

        # set_validation_data
        images_validation, targets_validation, labels_validation = self.set_validation_data(validation_data)

        # convert simulation images to tensor
        images_train = self.create_input_tensor(images_train)
        images_test = self.create_input_tensor(images_test)

        # set optimizer
        opt = optimizers.Adam(learning_rate = 0.0001)#, weight_decay = 1e-6)

        # get training labels (one-hot vectors/ matrices)
        labels_training = self.generate_labels(target_values_train)
        print(next(iter(labels_training.values()))[0][0])
        # convert dictionary labels to list of lists
        labels_training_list = [numpy.array(labels_training[k]) for k in labels_training]
        labels_training_list = [numpy.array(labels_training[k]) for k in labels_training]
        print('example label:', labels_training_list[0][4])

        # compile the model
        if self.parameter['loss']=='mse':
            loss = [losses.MeanSquaredError(),losses.MeanSquaredError()]
        elif self.parameter['loss']=='monotonic_mse':
            loss = [monotonic_mse]
        elif self.parameter['loss']=='cce':
            loss = [losses.CategoricalCrossentropy(label_smoothing=0.1)]
        elif self.parameter['loss']=='monotonic_cce':
            print("loss:", {self.parameter['loss']})
            loss = [monotonic_categorical_crossentropy]
        else:
            loss = [losses.KLDivergence()]#[monotonic_mse]#[monotonic_kl_divergence]#[losses.CategoricalCrossentropy(label_smoothing=0.1)]#[monotonic_categorical_crossentropy]#[losses.KLDivergence()]#[losses.MeanSquaredError(),losses.MeanSquaredError()]#[losses.MeanAbsoluteError(),losses.MeanAbsoluteError()]
        metric = [metrics.MeanSquaredError(), metrics.RootMeanSquaredError()]
        self.model.compile(optimizer=opt, loss=loss[:len(self.keys)], metrics=metric[:len(self.keys)])#,'accuracy'])
        self.model_compiled = True
        print(self.model.summary())

        # start iteration
        # repeat training to derive estimation for the model accuracy
        model_scores = []
        num_epochs = 2 if self.check else 25
        for i in range(MAX_VALIDATION_ROUND):
            # define model callbacks
            checkpoint_filepath = self.model_path + "/round_" + str(i) + "/weights_epoch-{epoch:02d}.weights.h5"
            h5_logger_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='accuracy', save_weights_only=True, mode='max')
            csv_logger_callback = callbacks.CSVLogger(filename=self.model_path + '/round_' + str(i) + '_history.csv')

            # train the model
            history = self.model.fit(x=images_train, y=labels_training_list, callbacks=[h5_logger_callback, csv_logger_callback], epochs=num_epochs, batch_size=256, shuffle=False)
            
            # obtain optimal number of epoch for the model, save scores to csv, load weights back to the model, delete weights
            print('finished training')
            model_score = self.update_model_with_optimal_parameters(images_validation=images_validation, targets_validation=targets_validation, model_path=self.model_path + "/round_" + str(i))
            # and register the model score
            model_scores.append(model_score)
            
            # save predictions of validation data from the best model wrt expected MSE
            labels_validation_prediction = self.model.predict(images_validation)
            targets_validation_prediction = self.get_numerical_prediction(labels_validation_prediction)
            self.save_experiment_prediction(target_values_true=targets_validation, target_values_prediction=targets_validation_prediction, data='validation', model_path=self.model_path + "/round_" + str(i))
            
            # save predictions of validation labels
            self.save_raw_prediction(raw_labels=labels_validation_prediction, model_path=self.model_path + "/round_" + str(i), data='validation')
            
            # prediction of test data
            labels_test_prediction = self.model.predict(images_test)
            targets_test_prediction = self.get_numerical_prediction(labels_test_prediction)
            self.save_prediction(target_values_true=target_values_test, target_values_prediction=targets_test_prediction, data='test_sim', model_path=self.model_path + "/round_" + str(i))
            
            # save predictions of test labels
            self.save_raw_prediction(raw_labels=labels_test_prediction, model_path=self.model_path + "/round_" + str(i), data='test_sim')

            # re-initialize model weights before next round
            self.reset_weights()
        # estimate network TRAINING reproducibility based on trained models
        self.record_model_scores(model_scores, data='validation')
        return 0

    def train_on_simulations_validate_with_simulations(self, images, target_values):
        # case if validation data is simulation data
        # get indices to split data to training and test
        train_test_splits = generate_ordered_train_test_indices(X=images)
        # set optimizer
        opt = optimizers.Adam(learning_rate = 0.0001, weight_decay = 1e-6)
        # compile the model
        loss = [losses.MeanSquaredError(),losses.MeanSquaredError()]#[losses.MeanAbsoluteError(),losses.MeanAbsoluteError()]
        metric = [metrics.MeanSquaredError(), metrics.RootMeanSquaredError()]
        self.model.compile(optimizer=opt, loss=loss[:len(self.keys)], metrics=metric[:len(self.keys)])#,'accuracy'])
        self.model_compiled = True
        print(self.model.summary())
        # strart iteration
        # repeat training to derive estimation for the model accuracy
        images = numpy.array(images)
        model_scores = []
        num_epochs = 2 if self.check else 50
        for i in range(MAX_VALIDATION_ROUND):
            # convert training and test simulation images to tensor
            x_train = images[train_test_splits[i][0].astype(int)]
            x_test = images[train_test_splits[i][1].astype(int)]
            # extract training and test simulation targets
            y_train = target_values.loc[train_test_splits[i][0]].reset_index(drop=True)
            y_test = target_values.loc[train_test_splits[i][1]].reset_index(drop=True)
            # get indices to split training data to training and validation
            train_validation_splits = generate_ordered_train_test_indices(X=x_train, max_splits=1)
            # extract training and test simulation images
            x_train = x_train[train_validation_splits[0][0]]
            x_validation = x_train[train_validation_splits[0][1]]
            # convert training, test& validation simulation images to tensor
            x_train = self.create_input_tensor(x_train)
            x_test = self.create_input_tensor(x_test)
            x_validation = self.create_input_tensor(x_validation)
            # extract training and validation simulation targets
            y_validation = y_train.loc[train_validation_splits[0][1]].reset_index(drop=True)
            y_train = y_train.loc[train_validation_splits[0][0]].reset_index(drop=True)
            # get training, test& validation labels (one-hot vectors/ matrices)
            y_train = self.generate_labels(y_train)
            # convert dictionary labels to list of lists
            labels_training_list = [numpy.array(y_train[k]) for k in y_train]
            # define model callbacks
            checkpoint_filepath = self.model_path + "/round_" + str(i) + "/weights_epoch-{epoch:02d}.weights.h5"
            h5_logger_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='accuracy', save_weights_only=True, mode='max')
            csv_logger_callback = callbacks.CSVLogger(filename=self.model_path + '/round_' + str(i) + '_history.csv')
            # train the model
            history = self.model.fit(x=x_train, y=labels_training_list, callbacks=[h5_logger_callback, csv_logger_callback], epochs=num_epochs, batch_size=256, shuffle=False)
            # obtain optimal number of epoch for the model, save scores to csv, load weights back to the model, delete weights
            print('finished training')
            model_score = self.update_model_with_optimal_parameters(images_validation=x_validation, targets_validation=y_validation, model_path=self.model_path + "/round_" + str(i))
            # and register the model score
            model_scores.append(model_score)
            # save predictions of validation data from the best model wrt expected MSE
            labels_validation_prediction = self.model.predict(x_validation)
            targets_validation_prediction = self.get_numerical_prediction(labels_validation_prediction)
            # if validation data are experiment we save with save_experiment_prediction else with save_simulation_prediction
            self.save_prediction(target_values_true=y_validation, target_values_prediction=targets_validation_prediction, data='validation', model_path=self.model_path + "/round_" + str(i))
            # save predictions of validation labels
            self.save_raw_prediction(raw_labels=labels_validation_prediction, model_path=self.model_path + "/round_" + str(i), data='validation')
            # prediction of training data
            labels_test_prediction = self.model.predict(x_test)
            targets_test_prediction = self.get_numerical_prediction(labels_test_prediction)
            self.save_prediction(target_values_true=y_test, target_values_prediction=targets_test_prediction, data='test_sim', model_path=self.model_path + "/round_" + str(i))
            # re-initialize model weights before next round
            self.reset_weights()
        # estimate network TRAINING reproducibility based on trained models
        self.record_model_scores(model_scores, data='validation')
        return 0

    def train_on_simulations_with_validation(self, images, target_values, validation_data):

        images_validation, targets_validation, labels_validation = self.set_validation_data(validation_data)
        # set_validation_data
        #images_validation, targets_validation = self.set_validation_data(validation_data)
        # convert simulation images and validation images to tensor
        images = self.create_input_tensor(images)
        # set optimizer
        opt = optimizers.Adam(learning_rate = 0.0001, decay = 1e-6)
        # get training and validation labels (one-hot vectors/ matrices)
        labels_training = self.generate_labels(target_values)
        # convert dictionary labels to list of lists
        labels_validation_list = [numpy.array(labels_validation[k]) for k in labels_validation]
        validation_data = (images_validation,labels_validation_list)
        # compile the model
        loss = [losses.MeanSquaredError(),losses.MeanSquaredError()]#[losses.MeanAbsoluteError(),losses.MeanAbsoluteError()]
        metric = [metrics.MeanSquaredError(), metrics.RootMeanSquaredError()]
        self.model.compile(optimizer=opt, loss=loss[:len(self.keys)], metrics=metric[:len(self.keys)])#,'accuracy'])
        self.model_compiled = True
        print(self.model.summary())
        # strart iteration
        # repeat training to derive estimation for the model accuracy
        model_scores = []
        num_epochs = 2 if self.check else 25
        for i in range(MAX_VALIDATION_ROUND):
            # define model callbacks
            checkpoint_filepath = self.model_path + "/round_" + str(i) + "/weights_epoch-{epoch:02d}.weights.h5"
            h5_logger_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_weights_only=True, mode='max')
            csv_logger_callback = callbacks.CSVLogger(filename=self.model_path + '/round_' + str(i) + '_history.csv')
            # train the model
            labels_training_list = [numpy.array(labels_training[k]) for k in labels_training]
            history = self.model.fit(x=images, y=labels_training_list, validation_data=validation_data, callbacks=[h5_logger_callback, csv_logger_callback], epochs=num_epochs, batch_size=256, shuffle=False)
            # obtain optimal number of epoch for the model, save scores to csv, load weights back to the model, delete weights
            print('finished training')
            model_score = self.update_model_with_optimal_parameters(images_validation=images_validation, targets_validation=targets_validation, model_path=self.model_path + "/round_" + str(i))
            # and register the model score
            model_scores.append(model_score)
            # save model
            #self.save_model(only_weights=True, model_path=self.model_path + "/round_" + str(i))
            # save predictions of validation data from the best model wrt expected MSE
            labels_validation_prediction = self.model.predict(images_validation)
            targets_validation_prediction = self.get_numerical_prediction(labels_validation_prediction)
            # if validation data are experiment we save with save_experiment_prediction else with save_simulation_prediction
            self.save_experiment_prediction(target_values_true=targets_validation, target_values_prediction=targets_validation_prediction, data='validation', model_path=self.model_path + "/round_" + str(i))
            # save predictions of validation labels
            self.save_raw_prediction(raw_labels=labels_validation_prediction, model_path=self.model_path + "/round_" + str(i), data='validation')
            # prediction of training data
            labels_train_prediction = self.model.predict(images)
            targets_train_prediction = self.get_numerical_prediction(labels_train_prediction)
            self.save_prediction(target_values=target_values, target_values_prediction=targets_train_prediction, data='train', model_path=self.model_path + "/round_" + str(i))
            # save predictions of training labels
            #self.save_raw_prediction(raw_labels=labels_train_prediction, model_path=self.model_path + "/round_" + str(i), data='train')
            # re-initialize model weights before next round
            self.reset_weights()

        # estimate network TRAINING reproducibility based on trained models
        self.record_model_scores(model_scores, data='validation')
        return 0

    def save_weights(self, model, model_path):
        """
        Save model weights.
        :param model: The model to be saved.
        """
        # serialize weights to HDF5
        model.save_weights(model_path + "/weights.h5")

    def save_model(self, model, model_path):
        """
        Save the complete model.
        :param model: The model to be saved.
        """
        model.save(model_path + "/model.keras") 

    def save_raw_prediction(self, raw_labels, model_path, data):
        path_raw_data = model_path + '/raw_labels_' + data + '_data.npz'
        numpy.savez(path_raw_data, **{key: raw_labels[i] for i, key in enumerate(self.keys)})

    def save_prediction(self, target_values_true, target_values_prediction, model_path, data):
        for key in self.keys:
            prediction_path = model_path + '/' + key + '_predictions_' + data + '.csv'
            max_prediction = target_values_prediction[0][key]
            avg_prediction = target_values_prediction[1][key]
            ground_truth = numpy.round(target_values_true[key],2)
            df = pandas.DataFrame({'max': numpy.round(max_prediction,2), 'avg': numpy.round(avg_prediction,2), 'gt': ground_truth})
            df.to_csv(prediction_path)

    def record_model_scores(self, model_scores, data):
        df_model_scores = pandas.DataFrame(model_scores) # for saving the model scores
        df_model_scores.to_csv(self.model_path + '/model_scores.csv')


        # get model with lowest expected MSE value
        # iterate all rounds and delete weights except for best model
        
        # run_parameters = pandas.DataFrame([data], columns=['col1', 'col2', 'col3'])
        # if os.path.exists(self.record_file):
        #     # append to the existing file
        #     run_parameters.to_csv(self.record_file, mode='a', header=False, index=False)
        # else:
        #     # create a new file and write the row
        #     run_parameters.to_csv(self.record_file, mode='w', header=True, index=False)

    def update_model_with_optimal_parameters(self,images_validation, targets_validation, model_path):
        # list weight files
        weight_files = glob(model_path+"/weights_*-*.h5")
        weight_files = sorted(weight_files)
        epochs = []
        #val_accs = []
        val_MSE = []
        best_id = 0
        acc_estimation = []
        # loop through weight files
        for i,weights in enumerate(weight_files):
            # load weights back to model
            self.model.load_weights(weights)
            # predict labels for validation data
            labels_validation_prediction = self.model.predict(images_validation) # --->>> this is a list of lists!!! need to make a for loop to take it into acount!
            # coldbatch labels
            targets_validation_prediction_max, targets_validation_prediction_avg = self.get_numerical_prediction(labels_validation_prediction)
            # calculate expected MSE"
            expected_MSE = self.get_MSE(targets_validation_prediction_avg, targets_validation)
            estimated_acc = self.get_relative_accuracy(targets_validation_prediction_avg, targets_validation)
            acc_estimation.append(estimated_acc)
            val_MSE.append(expected_MSE)
            # pattern-match epoch number
            epoch = int(re.search('epoch-(\d+)',weights).groups(1)[0])
            epochs.append(epoch)
            # potentially update index of best epoch
            if i!=0 and val_MSE[best_id] > expected_MSE:
                best_id = i
        # load best weights back to model
        self.model.load_weights(weight_files[best_id])
        # save model scores
        df_model_scores = pandas.DataFrame({'epoch':epochs,'expected_MSE':val_MSE, 'estimated_acc': acc_estimation}) # for saving the model scores
        df_model_scores.to_csv(model_path + '/model_scores.csv')
        # set best score
        model_best_score = {'epoch':epochs[best_id],'expected_MSE':val_MSE[best_id], 'estimated_acc': acc_estimation[best_id]}
        # remove unnecessary weight files- keep only the optimal one
        files_rm = glob(os.path.join(model_path,"weights_*-*.h5"))
        for f in files_rm:
            if weight_files[best_id] not in f:
                os.remove(f)
        # retrun model score: best epoch, best expected MSE as dict
        return model_best_score

    def test_on_experiment(self, images, target_values):
        # preprocess image shape
        images = self.create_input_tensor(images)
        if self.model_compiled and self.model_path:
            #model_scores = []
            for i in range(MAX_VALIDATION_ROUND):
               # load weights
               #self.model = self.load_weights(model=self.model)
               # predict on labeled and experiment data
                labels_test_prediction = self.model.predict(images)
                self.save_raw_prediction(raw_labels=labels_test_prediction, model_path=self.model_path + "/round_" + str(i), data='test_experiment')
                targets_test_prediction = self.get_numerical_prediction()
                self.save_experiment_prediction(target_values_true=target_values, target_values_prediction=targets_test_prediction, data='test_experiment', model_path=self.model_path + "/round_" + str(i))

    def test_on_experiment_in_batch(self, images, target_values):
        # preprocess image shape
        # iterate experiment images based on measurement column
        model_scores = {}
        for i, measurement in enumerate(target_values.measurement.unique()):
            experiment_targets = target_values[target_values.measurement == measurement]
            filtered_targets_by_measurement = target_values.measurement == measurement
            experiment_targets = target_values[filtered_targets_by_measurement]
            experiment_images = numpy.array(images)[filtered_targets_by_measurement]
            experiment_images = self.create_input_tensor(experiment_images)
            if self.model_compiled and self.model_path:
                model_scores[measurement] = []
                for i in range(MAX_VALIDATION_ROUND):
                    weight_file = glob(self.model_path + "/round_" + str(i) + "/weights_*-*.h5")[0]
                    # load weights
                    self.model.load_weights(weight_file)
                    # predict on experiment labeled data
                    experiment_prediction_labels = self.model.predict(experiment_images)
                    self.save_raw_prediction(raw_labels=experiment_prediction_labels, model_path=self.model_path + "/round_" + str(i), data='test_' + measurement)
                    experiment_prediction_values = self.get_numerical_prediction(experiment_prediction_labels)
                    self.save_experiment_prediction(target_values_true=experiment_targets, target_values_prediction=experiment_prediction_values, data='test_experiment_' + measurement, model_path=self.model_path + "/round_" + str(i))
                    score = self.get_MSE(values=experiment_prediction_values[1], targets=experiment_targets)
                    model_scores[measurement].append(score)
        df = pandas.read_csv(self.model_path + '/model_scores.csv', index_col=0)
        df['expected_MSE_test'] = numpy.round(numpy.mean(numpy.array(list(model_scores.values())).T, axis=1),2)
        df.to_csv(self.model_path + '/model_scores.csv', index=False)            
        # estimate network reproducibility based on trained models
        # self.record_model_scores(model_scores, data='test')
        best_id = df.idxmin()['expected_MSE']
        weight_files = glob(self.model_path+"/round_*/weights_*-*.h5")
        weight_files = sorted(weight_files)
        for f in weight_files:
            if weight_files[best_id] not in f:
                os.remove(f)

    def test_on_real(self, images, files):
        # preprocess image shape
        images = self.create_input_tensor(images)
        # load weights
        #model = self.load_weights(self.model)
        # compile model
        #self.model.compile(optimizer=self.parameter['optimizer'], loss=self.parameter['loss'], metrics=self.parameter['metrics'])
        if self.model_compiled and self.model_path:
            for i in range(MAX_VALIDATION_ROUND):
                weight_file = glob(self.model_path + "/round_" + str(i) + "/weights_*-*.h5")[0]
                # load weights
                self.model.load_weights(weight_file)
                # predict on unlabeled real data
                predictions = self.model.predict(images)
                self.save_raw_prediction(raw_labels=predictions, model_path=self.model_path + "/round_" + str(i), data='real')
                prediction_values = self.get_numerical_prediction(predictions)
                self.save_real_prediction(files=files, target_values_prediction=prediction_values, data='real', model_path=self.model_path + "/round_" + str(i))

    def get_numerical_prediction(self, labels_prediction):
        # uses label_coder to decode labels and retrun dictionary
        targets_prediction_max = dict.fromkeys(self.keys, [])
        targets_prediction_avg = dict.fromkeys(self.keys, [])
        if len(self.keys) <= 1:
            labels_prediction = [labels_prediction]
        for i, key in enumerate(self.keys):
            targets_prediction_max[key] = self.label_coder.coldbatch_maximum(labels_prediction[i].copy(), key)
            targets_prediction_avg[key] = self.label_coder.coldbatch_average(labels_prediction[i], key)
        return targets_prediction_max, targets_prediction_avg
        # targets_prediction = dict.fromkeys(self.keys, [])
        # for i, key in enumerate(self.keys):
        #     targets_prediction[key] = dict.fromkeys(['max', 'average'], [])
        #     numeric_max = label_coder.coldbatch_maximum(labels_prediction[i], key)
        #     targets_prediction[key]['max'] = numeric_max
        #     numeric_avg = label_coder.coldbatch_average(labels_prediction[i], key)
        #     targets_prediction[key]['average'] = numeric_avg

    def get_MSE(self, values, targets):
        # recursive function
        # values is a dictionary at first
        # empty list is created to save MSE value for each key
        # the average of the MSE of the keys is returned
        # when only one key the MSE of that key is returned 
        keys = self.keys_validation if 'exp' in self.parameter['validation'] else self.keys
        if isinstance(values, dict):
            mse_values = []
            for i, key in enumerate(keys):
                mse_values.append(self.get_MSE(values[key], targets[key]))
            
            return sum(mse_values)/len(keys)
        else:
            return numpy.square(numpy.subtract(numpy.array(values), numpy.array(targets))).mean()
        
    def get_relative_accuracy(self, values, targets, thresh=2, keys=None):
        # 
        keys = self.keys_validation if 'exp' in self.parameter['validation'] else self.keys
        if isinstance(values, dict):
            acc_true = 0
            for i, key in enumerate(keys):
                acc_true += self.get_relative_accuracy(values[key], targets[key])
            return acc_true/len(keys)
        else:
            return sum([1 for i in range(len(values)) if values[i]-thresh <= targets[i] <= values[i]+thresh])/len(targets)     

    def save_experiment_prediction(self, target_values_true, target_values_prediction, data, model_path):
        self.keys_validation = [key for key in self.keys if '_' not in key]
        for key in self.keys:
            prediction_path = model_path + '/' + key + '_predictions_' + data + '.csv'
            max_prediction = target_values_prediction[0][key]
            avg_prediction = target_values_prediction[1][key]
            ground_truth = numpy.round(target_values_true[key],2) if key in self.keys_validation else pandas.NA
            layer_thickness = target_values_true['thickness']
            df = pandas.DataFrame({'max': numpy.round(max_prediction,2), 'avg': numpy.round(avg_prediction,2), 'gt': ground_truth, 'thickness': numpy.round(layer_thickness,2)})
            df.to_csv(prediction_path)

    def save_real_prediction(self,files, target_values_prediction, data, model_path):
        for i,key in enumerate(self.keys):
            prediction_path = model_path + '/' + key + '_predictions_' + data + '.csv'
            max_prediction = target_values_prediction[0][key]
            avg_prediction = target_values_prediction[1][key]
            df = pandas.DataFrame({'filename': files[i], 'max': numpy.round(max_prediction,2), 'avg': numpy.round(avg_prediction,2)})
            df.to_csv(prediction_path)
        
    def plot_model_accurarcy_and_loss(self, history, model_path):
        max_val_acc = round(max(history['val_accuracy']),3)
        min_val_loss = round(min(history['val_loss']),3)
        best_epoch = numpy.argmin(history['val_loss'])


    def plot_model(self, path):
        utils.plot_model(self.model, path + 'results/' + self.TYPE.lower() + '_model_plot.pdf', show_shapes=True, show_layer_names=False)


    def load_weights(self, model):
        pass
        """
        Load model weights. To load the model, instantiate the model abd load the weights into the empty model object.
        :param model: The empty model.
        :return model with weights.
        """
        #model.load_weights(self.parameter['path'] + 'deep_learning/models/' + self.TYPE.lower() + '_weights_' + self.parameter['run'] + '.h5')
        #return model

    def load_weights_from_path(self, model, path):
        """
        Load model weights from path. To load the model, instantiate the model abd load the weights into the empty model object.
        :param model: The empty model.
        :return model with weights.
        """
        model.load_weights(path)
        return model

    def estimate_model(self, model_path=None, thresh=None):
        if model_path is not None:
            self.model_path = model_path
        if not os.path.exists(self.model_path):
            print("No such model exists in given path")
            exit()
        if not thresh:
            scores_file = self.model_path + '/model_scores.csv'
            scores = pandas.read_csv(scores_file)
            estimation = scores.estimated_acc.mean()
            print(estimation)
        else:
            prediction_files = glob(self.model_path + '/round*/*_predictions_*.csv')
            s = 0.0
            for f in prediction_files:
                df = pandas.read_csv(f)
                s += self.get_relative_accuracy(df['avg'], df['gt'], thresh=thresh)
            print(s/len(prediction_files))