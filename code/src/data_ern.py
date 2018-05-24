
import numpy as np
from mne.filter import resample,filter_data
import pickle
import os

from data import Data, data_shuffle

def merge_dicts(dict1,dict2):
    if set(dict1.keys()) != set(dict1.keys()):
        raise ValueError('Dicts to merge inconsistent')
    res_dict = {key:(dict1[key],dict2[key]) for key in dict1.keys()}
    return res_dict

class DataERN(Data):
    def __init__(self, path_to_data):

        start_epoch = 0  # seconds
        end_epoch = 1.25  # seconds
        sample_rate = 200
        Data.__init__(self, path_to_data, start_epoch, end_epoch, sample_rate)

    def _process_csv_data_file(self, path_to_csv_file, subjects_data):
        print path_to_csv_file
        key = path_to_csv_file[-13:-11]
        array = np.genfromtxt(path_to_csv_file, delimiter=',')[1:,1:]  # We doesn't need 1 row (header) and 1 column (timestamps)
        indecies_of_events = np.where(array[:, -1] == 1)[0]
        dind_before = int(abs(self.start_epoch) * self.sample_rate)
        dind_after = int(abs(self.end_epoch) * self.sample_rate)
        array = array[:, :-2]  # Cut off labels and EOG column

        array = filter_data(array.transpose(1,0), sfreq=self.sample_rate, l_freq=1., h_freq=40.).transpose(1,0)

        ind = []
        map(lambda x: ind.extend(range(x - dind_before, x + dind_after)), indecies_of_events)
        if key in subjects_data.keys():
            subjects_data[key] = np.append(subjects_data[key],
                                           np.reshape(array[ind, :], (
                                           len(indecies_of_events), dind_after + dind_before, array.shape[1]),
                                                      order='C'),
                                           axis=0)
        else:
            subjects_data[key] = np.reshape(array[ind, :],
                                            (len(indecies_of_events), dind_after + dind_before, array.shape[1]),
                                            order='C')

    def _process_csv_label_file(self, target_subjects):
        path_to_file = os.path.join(self.path_to_data, 'TrainLabels.csv')
        labels = np.genfromtxt(path_to_file, delimiter=',', dtype=None)[1:,:]  # We doesn't need 1 row (header)
        res = {subj: [] for subj in target_subjects}
        for ind in range(labels.shape[0]):
            subj = labels[ind, 0][1:3]
            if subj in target_subjects:
                res[subj].append(int(labels[ind, 1]))
        return res

    def _load_from_spec_folder(self, path_to_folder):
        # Process all .csv files in specified folder
        list_of_files = sorted([f for f in os.listdir(path_to_folder) if f.endswith(".csv")])
        subjects_data = {}
        f = lambda x: self._process_csv_data_file(os.path.join(path_to_folder, x), subjects_data)
        map(f, list_of_files)
        return subjects_data

    def get_data(self, shuffle=False, resample_to=None, subject_indices=range(12), balance_classes=False):
        """
        Args:
            shuffle: bool
            resample_to: resample signal to N Hz
        Returns:
            A tuple of 3 numpy arrays: data (Trials x Time x Channels), class labels and subject labels

        """

        if os.path.exists('./data_ern/train.pkl'):
            subjects_data = pickle.load(open('./data_ern/train.pkl', "rb"))
        else:
            folder_with_datacsv = os.path.join(self.path_to_data, 'train')
            subjects_data = self._load_from_spec_folder(folder_with_datacsv)
            pickle.dump(subjects_data, open('./data_ern/train.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        keys = subjects_data.keys()
        keys.sort()
        subjects = [keys[ind] for ind in subject_indices]
        class_labels = self._process_csv_label_file(subjects)
        subjects_data = merge_dicts(subjects_data,class_labels)

        X, y,subj_labels = self._get_distinct_subjects(subjects_data, subjects, balance_classes)
        if shuffle:
            X, y,subj_labels = data_shuffle(X,y,subj_labels)

        if (resample_to is None) or (resample_to == self.sample_rate):
            return X, y,subj_labels
        else:
            duration = self.end_epoch - self.start_epoch
            downsample_factor = X.shape[1]/(resample_to * duration)
            return resample(X,up=1., down=downsample_factor, npad='auto',axis=1), y, subj_labels


    def _balance_classes(self,x,y):
        y=np.array(y)
        target_indices = np.where(y == 1)[0]
        nontarget_indices = np.where(y == 0)[0]
        target_indices = np.random.choice(target_indices, len(nontarget_indices), False)
        indices = np.hstack((target_indices, nontarget_indices))

        indices = list(sorted(indices))
        x = x[indices]
        y = y[indices]
        return x,y

    def _get_distinct_subjects(self, subjects_data, subjects, balance_classes):
        """
        Function returns array ob observations for first(sic!) N subjects
        Args:
            num_subjects:
        Returns:
            Numpy array of shape

        """
        if balance_classes:
            subjects_data = {k:self._balance_classes(*subjects_data[k]) for k in subjects}

        X = np.concatenate([subjects_data[k][0] for k in subjects], axis=0)
        y = np.concatenate([np.array(subjects_data[k][1]) for k in subjects], axis=0)
        subj_labels = np.concatenate([np.full_like(subjects_data[k][1], k) for k in subjects])


        return X, y, subj_labels

    # def _get_distinct_subjects(self, subjects_data, subjects, balance_classes):
    #     """
    #     Function returns array ob observations for first(sic!) N subjects
    #     Args:
    #         num_subjects:
    #     Returns:
    #         Numpy array of shape
    #
    #     """
    #     X=np.concatenate([subjects_data[k][0] for k in subjects],axis=0)
    #     y = np.concatenate([np.array(subjects_data[k][1]) for k in subjects], axis=0)
    #     subj_labels = np.concatenate([np.full_like(subjects_data[k][1],k) for k in subjects])
    #
    #
    #
    #     if balance_classes:
    #         target_indices = np.where(y == 1)[0]
    #         nontarget_indices = np.where(y == 0)[0]
    #         target_indices = np.random.choice(target_indices, len(nontarget_indices), False)
    #         indices = np.hstack((target_indices,nontarget_indices))
    #
    #         indices = list(sorted(indices))
    #         X = X[indices, ...]
    #         y = y[indices]
    #         subj_labels = subj_labels[indices]
    #     return X, y, subj_labels

if __name__ == '__main__':

    data = DataERN('/home/likan_blk/BCI/eegnet/data_ern')
    x, y,subj_labels = data.get_data(shuffle=True,resample_to=128,subject_indices=range(12),balance_classes=True)
    print x.shapem