# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy as np
import pandas as pd
import fabio
import glob

class Dataset:
    def __init__(self, path, folder, file, fast_data, first_frame=None, test=False):
        self.path = path
        self.folder = folder
        self.file = file
        self.fast_data = fast_data
        self.first_frame = None
        self.sample = 10 if test else False


    def get_dataframe(self):
        targets_dataframe = pd.read_csv(self.path + self.folder + "/" + self.file)
        return targets_dataframe

    def preprocess_dataframe(self, dataframe):
        # convert 'Frame' from float64 to int64
        dataframe['Frame'] = dataframe['Frame'].astype(int) + (260 if '300' in self.folder else 0) # need to modify the dataset eventually
        # keep relevant columns
        dataframe = dataframe[['Frame', 'Distance', 'Radius', 'thickness']]
        dataframe.insert(loc=0, column='Measurement', value=self.folder)
        # drop rows with NaN entries
        dataframe = dataframe.dropna()
        dataframe = dataframe.reset_index(drop=True)

        # set first frame
        if not self.first_frame:
            # set first frame to highest frame where thickness is less than 1.0 nm if first_frame not defined
            largest_thickness_smaller_one = max(dataframe[dataframe.thickness<1.0].index)
            self.first_frame = dataframe.Frame[largest_thickness_smaller_one]
        while dataframe.Frame[0] != self.first_frame:
            dataframe = dataframe.drop(index=0)
            dataframe = dataframe.reset_index(drop=True)
        # set last frame
        if True:
            two_r_to_d_ratio_at_most_one = min(dataframe[(dataframe.Radius*2)/dataframe.Distance > 1.0].index)
            dataframe = dataframe[:two_r_to_d_ratio_at_most_one]
        return dataframe[['Measurement', 'Frame', 'Distance', 'Radius', 'thickness']]

    def get_target_values(self):
        if self.fast_data:
            return self.get_ready_target_values()

        target_values = self.get_dataframe()
        target_values = self.preprocess_dataframe(dataframe=target_values)
        if self.sample:
            target_values = target_values[:10]
        return target_values

    def get_ready_target_values(self):
        # if experimental data and targets are in npy file
        with open(self.path + self.folder + "/fitted_experiments.npy", 'rb') as file:
            _ = np.load(file)
            target_values = np.load(file)
        target_values = pd.DataFrame(target_values, columns=['Frame', 'Distance', 'Radius', 'thickness'])
        target_values.insert(loc=0, column='Measurement', value=self.folder)
        return target_values


    def get_summed_images(self):
        if self.fast_data:
            return self.get_ready_images()

        target_values = self.get_target_values()
        number_images_summed = 10
        summed_images = []
        for index, row in target_values.iterrows():
            images = []
            for frame in range(row.Frame - number_images_summed + 1, row.Frame + 1):
                number = str(frame).zfill(5) # format: "00000"
                try:
                    # flip top and bottom (vertical flip in y-direction)
                    image_file = self.path + self.folder + '/*' + number + '.cbf'
                    image_file = glob.glob(image_file)[0]
                    image = np.flip(fabio.open(image_file).data, axis=0)
                    images.append(image.astype(np.float32))
                except FileNotFoundError as e:
                    print(f'\n Frame {frame} not available or does not exist. \nError: {e}\n')
                    continue
                except IndexError as e:
                    print(f'\n Frame {frame} not available or does not exist. \nError: {e}\n')
                    continue
            summed_images.append(sum(images))
            if self.sample:
                if len(summed_images)>=self.sample:
                    break
        return summed_images

    def get_images(self):
        if self.fast_data:
            return self.get_ready_images()
        target_values = self.get_target_values()
        images = []
        for index, row in target_values.iterrows():
            number = str(row.Frame).zfill(5) # format: "00000"
            # flip top and bottom (vertical flip in y-direction)
            image = np.flip(fabio.open(self.path + self.folder + '/sputter/' + '_sputter_' + number + '.cbf').data, axis=0)
            images.append(image.astype(np.float32))
            if self.sample:
                if len(images)>=self.sample:
                    break
        return images

    def get_ready_images(self):
        with open(self.path + self.folder + "/fitted_experiments.npy", 'rb') as file:
            images = np.load(file)
        return images


class Experiment:
    def __init__(self, data_path, fast_exp=False, test=False):
        self.datasets = []
        for i in range(0, len(data_path['folders']), 1):
            self.datasets.append(Dataset(path=data_path['path'], folder=data_path['folders'][i], file=data_path['target_value_files'][i], fast_data=fast_exp, test=test))

    def get_target_values(self):
        dataframes = []
        for dataset in self.datasets:
            dataframes.append(dataset.get_target_values())
        target_values = pd.concat(dataframes)
        target_values.columns = target_values.columns.str.lower()

        return target_values

    def get_images(self, sum=True):
        images = []
        for dataset in self.datasets:
            if sum:
                images.extend(dataset.get_summed_images())
            else:
                images.extend(dataset.get_images())
        return images

    def load_data(self):
        return self.get_images(), self.get_target_values()

    def get_type(self, K, images, targets, sample=None):
        filtered_targets_by_K = targets.measurement=='sputter_'+K
        targets_K = targets[filtered_targets_by_K]
        images_K = np.array(images)[filtered_targets_by_K]
        assert len(images_K)==len(targets_K)
        if sample:
            targets_K = targets_K.sample(n=sample)
            images_K = images_K[targets_K.index]
        return images_K, targets_K
