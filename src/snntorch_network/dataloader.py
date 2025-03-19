from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class WisdmDatasetParser():
    """Split the given file into training, validation and test datasets.
    For each one decide if do shuffle or not.

    The percentages for the split are already defined in the dataset.
    """
    def __init__(self,
                 file_name: str,
                 norm: str = "std",
                 class_sublset: str | None = None,
                 subset_list: list[int] | None = None):

        """Initialize the class.

        Parameters
        ----------

        file_name : str
            The path to the file containing the dataset.

        norm : str
            The normalization to apply to the dataset. It can be "std" for
            standard normalization, custom to just divide by the standard
            deviation or None to not normalize the dataset.

        class_sublset : str
            The subset of classes to use. It can be "7BC" for the 7 best
            classes or "7WC" for the 7 worst classes in terms of separability
            score. It can also be "subset_2" for a custom subset of classes
            identified by Vittorio, or custom to give use the custom
            subset list.

        subset_list : list[int]
            The list of classes to use if class_subset is set to custom.
        """


        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        self.class_sublset = class_sublset
        self.norm = norm
        self.mean = np.mean(x_train, axis=(0,1))
        self.std = np.std(x_train, axis=(0,1))
        print(self.mean.shape)
        print(self.std.shape)
        if self.norm == "std":
            x_train = x_train - self.mean
            x_train = x_train/self.std

            x_val = x_val - self.mean
            x_val = x_val/self.std

            x_test = x_test - self.mean
            x_test = x_test/self.std

        elif self.norm == "custom":
            x_train = x_train/self.std
            x_val = x_val/self.std
            x_test = x_test/self.std

        elif self.norm == None:
            pass

        print(f'ytrain shape {y_train.shape}')
        print(f'yval shape {y_val.shape}')
        print(f'ytest shape {y_test.shape}')

        x_train = np.transpose(x_train,axes=(0,2,1))
        x_val = np.transpose(x_val,axes=(0,2,1))
        x_test = np.transpose(x_test,axes=(0,2,1))

        if self.class_sublset is not None:
            if self.class_sublset == '7BC':
                selected_classes =  [1,6,7,8,13,14,17]
            elif self.class_sublset == '7WC':
                selected_classes = [11,12,13,14,15,16,17]
            elif self.class_sublset == 'subset_2':
                selected_classes = [6, 7, 8, 9, 10, 11, 12]
            elif self.class_sublset == 'custom':
                selected_classes = subset_list

            x_train, y_train = filter_dataset(x_train, y_train, selected_classes)
            x_val, y_val = filter_dataset(x_val, y_val, selected_classes)
            x_test, y_test = filter_dataset(x_test, y_test, selected_classes)

        # In case the dataset is in one-hot encoded, convert it to integer
        if len(y_test.shape) > 1:
            self.train_dataset = (x_train, np.argmax(y_train, axis=-1))
            self.val_dataset = (x_val, np.argmax(y_val, axis=-1))
            self.test_dataset = (x_test, np.argmax(y_test, axis=-1))
        else:
            self.train_dataset = (x_train, y_train)
            self.val_dataset = (x_val,y_val)
            self.test_dataset = (x_test,y_test)

        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])

        return self.val_dataset

    def get_test_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])

        return self.test_dataset

    def de_std(self, data):
        if self.norm == "norm":
            data= data * self.std
            data= data + self.mean
        if self.norm == "custom":
            data= data * self.std

    def do_std(self, data):
        data= data - self.mean
        data= data / self.std

    @staticmethod
    def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])

def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])


def filter_dataset(x_train, y_train, selected_classes):
    # Create a mapping dictionary for the selected classes
    class_mapping = {original: new for new, original in enumerate(selected_classes)}

    # Convert selected_classes to a set for faster look-up
    selected_set = set(selected_classes)

    # Get the indices of the selected classes in y_train
    original_class_indices = np.argmax(y_train, axis=1)
    mask = np.isin(original_class_indices, selected_classes)

    # Filter the data and labels using the mask
    filtered_x = x_train[mask]
    filtered_y = y_train[mask]

    # Map the original class indices to new indices
    new_class_indices = np.vectorize(class_mapping.get)(original_class_indices[mask])

    # Create the new one-hot encoded labels
    new_one_hot_y = np.zeros((filtered_y.shape[0], len(selected_classes)))
    new_one_hot_y[np.arange(filtered_y.shape[0]), new_class_indices] = 1

    return filtered_x, new_one_hot_y

class WisdmDataset(Dataset):
    """This is the class that will be used to actually call the pytorch
    dataloader.

    something like:
    dataset = WisdmDatasetParser(f'{Path.home ...
    train_set = dataset.get_training_set()
    val_set = dataset.get_validation_set()

    data, label = train_set
    print(data.shape)

    train_dataset = WisdmDataset(train_set)
    val_dataset = WisdmDataset(val_set)

    train_loader = DataLoader(dataset=train_datase, ....
    val_loader  = DataLoader(dataset= val_dataset, ....

    """

    def __init__(self, data, transform=None, augument=None):
        xs, y = data
        self.x = []
        self.y = y
        self.augument = augument
        self.transform = transform
        if self.transform is not None:
            print("transforming array ....")
            for x in xs:
                tmp = self.transform(x)
                self.x.append(tmp)
            print(f' lenght of transformed array {len(self.x)}')
            self.x = np.array(self.x)

        else:
            self.x = xs

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augument:
            x = self.augument(x)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.y.shape[0]


class WisdmEncodedDatasetParser():
    """Split the given file into training, validation and test datasets.
    For each one decide if do shuffle or not.

    This if for the dataset already in spikes. NOT USED.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)

        self.train_dataset = (x_train, y_train)
        self.val_dataset = (x_val,y_val)
        self.test_dataset = (x_test,y_test)


        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])

        return self.val_dataset

    def get_test_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])

        return self.test_dataset

    def load_wisdm2_data(self,file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['x_train'], data['x_val'], data['x_test'], data['y_train'], data['y_val'], data['y_test'])
