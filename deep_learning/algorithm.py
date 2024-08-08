import numpy
import pandas
import os
from tensorflow.keras import models, utils, callbacks, optimizers, losses, metrics
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from deep_learning import label_coder
#from network_models import MLPNet, CNNNet
from glob import glob
import re


MAX_ROUNDS = 1

def custom_train_test_split(X, y, max_splits=10, test_size=0.2):
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    if num_test_samples == 0:
        num_test_samples = 1  # Ensure at least one sample in the test set
    indices = numpy.arange(num_samples)
    numpy.random.seed(42)  # For reproducibility
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

    def __init__(self, model, parameter, morphology):
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
        print('morphology:', morphology)
        self.model_path = f"{parameter['path']}/results/{morphology}/{self.TYPE.lower()}/{dt.month:02d}{dt.day:02d}_{self.parameter['beta']}_intensity" # middle
        print(self.model_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def reset_weights(self):
        '''
        re-randomalize the model's weights
        this functionality is not yet part of tensorflow but it might be in the future!
        this solution is taken from the thread: https://github.com/keras-team/keras/issues/341
        *NOTE* This will not work with every layer type! (e.g. LTSM)
        '''
        for layer in self.model.layers:
            if isinstance(layer, keras.Model): #if you're using a model as a layer
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


    def train_on_simulations(self, images, target_values, validation_data):
        # set_validation_data
        images_validation, targets_validation = self.set_validation_data(validation_data)
        # convert images to tensor
        images = self.create_input_tensor(images)
        # set optimizer
        opt = optimizers.Adam(learning_rate = 0.0001, decay = 1e-6)
        # get training and validation labels (one-hot vectors)
        training_labels = {}
        validation_labels = {}
        for key in self.keys:
            # NOTE: label_coder should return numpy arrays!
            training_labels[key] = self.label_coder(target_values[key].to_numpy(), key)
            validation_labels[key] = self.label_coder(targets_validation[key].to_numpy(), key)
        # convert dictionary labels to list of lists
        validation_data = (images_validation,list(validation_labels.values()))
        # compile the model
        self.model.compile(optimizer=opt, loss=losses.MeanAbsoluteError(), metrics=[metrics.MeanSquaredError(), metrics.RootMeanSquaredError()])
        self.model_compiled = True
        print(model.summary())

        # strart iteration
        # repeat training to derive estimation for the model
        model_scores = []
        for i in MAX_VALIDATION_ROUND:
            # define model callbacks
            checkpoint_filepath = model_path + "/round_" + str(i) + "/weights_epoch-{epoch:02d}_acc-{val_accuracy:.3f}.h5"
            h5_logger_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_weights_only=True, mode='max')
            csv_logger_callback = callbacks.CSVLogger(filename=model_path + '/round_' + str(i) + '/history.csv')
            # train the model
            history = self.model.fit(x=images, y=list(training_labels.values()), validation_data=validation_data, callbacks=[h5_logger_callback, csv_logger_callback], epochs=25, batch_size=256, shuffle=False)
            # draw training history plot
            #self.plot_model_accurarcy_and_loss(history=history.history, model_path=self.model_path + "/round_" + str(i))
            # obtain optimal number of epoch for the model, save scores to csv, load weights back to the model, delete weights
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
            self.save_simulation_prediction(target_values_true=target_values, target_values_prediction=targets_train_prediction, data='train', model_path=self.model_path + "/round_" + str(i))
            # save predictions of training labels
            self.save_raw_prediction(raw_labels=labels_train_prediction, model_path=self.model_path + "/round_" + str(i), data='train')
            # re-initialize model weights before next round
            self.reset_weights()

        # estimate network TRAINING reproducibility based on trained models
        self.record_model_scores(model_scores, data='validation')

    def save_weights(self, model, model_path):
        """
        Save model weights.
        :param model: The model to be saved.
        """
        # serialize weights to HDF5
        model.save_weights(model_path + "/weights.h5")

    def update_model_with_optimal_parameters(self,images_validation, targets_validation, model_path):
        # list weight files
        weight_files = glob(model_path+"/weights_*-*.h5")
        weights_files = sorted(weights_files)

        epochs = []
        val_accs = []
        val_MSE = []
        best_id = 0
        # loop through weight files
        for i,weights in enumerate(weights_files):
            # load weights back to model
            self.model.load_weights(weights)
            # predict labels for validation data
            labels_validation_prediction = self.model.predict(images_validation) # --->>> this is a list of lists!!! need to make a for loop to take it into acount!
            # coldbatch labels
            targets_validation_prediction_max, targets_validation_prediction_avg = self.get_numerical_prediction(labels_validation_prediction)
            # calculate expected MSE
            expected_MSE = self.get_MSE(targets_validation_prediction_avg, targets_validation)
            val_MSE.append(expected_MSE)
            # pattern-match epoch number
            epoch = int(re.search('epoch-(\d+)',weights).groups(1)[0])
            epochs.append(epoch)
            # pattern-match validation accuracy
            val_acc = float((re.findall('\d+\.\d+',weights))[0])
            val_accs.append(val_acc)
            if i!=0 and val_MSE[best_id] > expected_MSE:
                # set new best epoch
                best_id = i
        # load best weights back to model
        self.model.load_weights(weights_files[best_id])
        # save model scores
        df_model_scores = pd.DataFrame({'epoch':epochs,'val_acc_MSE':val_accs,'expected_MSE':val_MSE}) # for saving the model scores
        df_model_scores.to_csv(model_path + '/model_scores.csv')
        # set best score
        model_best_score = {'epoch':epochs[best_id],'val_acc_MSE':val_accs[best_id],'expected_MSE':val_MSE[best_id]}
        # remove unnecessary weight files- keep only the optimal one
        files_rm = glob(os.path.join(model_path,"weights_*-*.h5"))
        for f in files_rm:
            if weights_files[best_id] not in f:
                os.remove(f)
        # retrun model score: best epoch, best expected MSE as dict
        return model_best_score

    def test_on_experiment(self, images, target_values):
        # preprocess image shape
        images = self.create_input_tensor(images)
        if self.model_compiled and self.model_path:
            #model_scores = []
            for i in MAX_VALIDATION_ROUND:
               # load weights
               self.model = self.load_weights(model=self.model)
               # predict on labeled and experiment data
               targets_test_prediction = self.coldbatch(self.model.predict(images))
               self.save_experiment_prediction(target_values_true=target_values, target_values_prediction=targets_test_prediction, data='test_experiment', model_path=self.model_path + "/round_" + str(i))

        # estimate network reproducibility based on trained models
        # self.record_model_scores(model_scores, data='test')

    def get_numerical_prediction(self, labels_prediction):
        # uses label_coder to decode labels and retrun dictionary
        targets_prediction_max = dict.fromkeys(self.keys, [])
        targets_prediction_avg = dict.fromkeys(self.keys, [])
        for i, key in enumerate(self.keys):
            targets_prediction_max[key] = label_coder.coldbatch_maximum(labels_prediction[i], key)
            targets_prediction_avg[key] = label_coder.coldbatch_average(labels_prediction[i], key)

        return targets_prediction_max, targets_prediction_avg
        # targets_prediction = dict.fromkeys(self.keys, [])
        # for i, key in enumerate(self.keys):
        #     targets_prediction[key] = dict.fromkeys(['max', 'average'], [])
        #     numeric_max = label_coder.coldbatch_maximum(labels_prediction[i], key)
        #     targets_prediction[key]['max'] = numeric_max
        #     numeric_avg = label_coder.coldbatch_average(labels_prediction[i], key)
        #     targets_prediction[key]['average'] = numeric_avg

    def get_MSE(self, values, targets):
        if isinstance(values, dict):
            mse_values = []
            for i, key in enumerate(values):
                mse_values.append(self.get_MSE(values[key], targets[i]))
            return sum(mse_values)/len(values)
        return numpy.square(numpy.subtract(values, targets)).mean()


    def save_experiment_prediction(self, target_values_true, target_values_prediction, data, model_path):
        #
        #
        pass

    def plot_model_accurarcy_and_loss(self, history, model_path):
        max_val_acc = round(max(history['val_accuracy']),3)
        min_val_loss = round(min(history['val_loss']),3)
        best_epoch = numpy.argmin(history['val_loss'])


    def plot_model(self, path):
        utils.plot_model(self.model, path + 'results/' + self.TYPE.lower() + '_model_plot.pdf', show_shapes=True, show_layer_names=False)


    def load_weights(self, model):
        """
        Load model weights. To load the model, instantiate the model abd load the weights into the empty model object.
        :param model: The empty model.
        :return model with weights.
        """
        model.load_weights(self.parameter['path'] + 'deep_learning/models/' + self.TYPE.lower() + '_weights_' + self.parameter['run'] + '.h5')
        return model

    def load_weights_from_path(self, model, path):
        """
        Load model weights from path. To load the model, instantiate the model abd load the weights into the empty model object.
        :param model: The empty model.
        :return model with weights.
        """
        model.load_weights(path)
        return model
