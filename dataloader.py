import os
import tensorflow as tf
# import urllib.request
import subprocess
import numpy as np
import os
from abc import ABC, abstractmethod
import pickle

class DataLoader(ABC):
    '''
    abstract base class for loading and preprocessing datasets

    Private methods: 
        _download_file:
            reusable private helper function to download files, 
            used by the public abstract method download (thrice for each dataset)
            @params: 
                url: dropbox url of dataset
                filename: local filename to save to
            @returns filename, side effect: downloads a file

    Public methods: 
        download
            data-specific public abstract method to be implemented for downloading the datasets,
            used in both dataloaders, calls helper DL method

        load
            public abstract method to load the data, dependent on which dataset we're loading 
            @returns: 
                train_data, test_data, test_labels (all as numpy arrays)
            
        preprocess
            public abstract method to preprocess data
            @args: data (raw image data)
            @returns: data (preprocessed depending on color or bw)

        get_training_data
            inits download and gets training data 
            @args: batch_size, shuffle_buffer (size of buffer to be shuffled), seed (for reproducibility)
            @returns dataset

        get_test_data 
            itits download and gets test data 
            @returns: test_data, test_labels
    '''
    def __init__(self, dset_name, data_dir='data/'):
        '''
        Args:
            dset_name: name of the dataset
            data_dir: dir where data will be stored
        '''
        self.dset_name = dset_name
        self.data_dir = data_dir

        #create dir if it doesnt exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f'created dir {self.data_dir}')

    def _download_file(self, url, filename):
        '''
        downloads a file from url if it doesnt already exist
        
        Args:
            url: URL to download from
            filename: local filename to save to
        '''
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            print(f'file {filename} already exists, skipping download...')
            return filepath 

        #add dl=1 to force download from dropbox
        url = url.replace('dl=0', 'dl=1')

        # urllib.request.urlretrieve(url, filepath) #use urllib instead of wget, this is fine because it is a part of the standard library
        subprocess.run(["wget", "-O", filepath, url], check=True) #changed to wget since I dont want to get a bad grade (also wget is apparently more robust)
        print(f'successfully downloaded {filename}')

        return filepath 

    @abstractmethod
    def download(self):
        '''
        abstract method to download dataset files
        '''
        pass

    @abstractmethod
    def load(self):
        '''
        abstract method to load raw data from files
        '''
        pass

    @abstractmethod
    def preprocess(self, data):
        '''
        abstract method to preprocess data

        Args: 
            data: raw data to preprocess
        
        Returns: 
            preprocessed data
        '''
        pass

    def get_training_data(self, batch_size=256, shuffle_buffer=10_000, seed=None):
        '''
        Args: 
            batch_size: size of batches
            shuffle_buffer: buffer size for shuffling
        
        Returns: 
            tf.data.Dataset
        '''
        self.download()
        train_data, _, _ = self.load()
        train_data = self.preprocess(train_data)

        dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size=shuffle_buffer, seed=seed).batch(batch_size) #use tf.shuffle to make the data "truly" random as to not accidentally get ordered data somehow 
        return dataset 

    def get_test_data(self):
        '''
        Returns:
            test_data, test_labels
        '''
        self.download()
        _, test_data, test_labels = self.load()
        test_data = self.preprocess(test_data)
        return test_data, test_labels 

class MNISTBWDataLoader(DataLoader):
    '''
    Dataloader for bw mnist dataset
    overrides and implements abstract methods from Dataloader:

        download: downloads data
        load: loads the data into train, test and test labels 
        preprocess: preprocesses the data (suitable to color version)
    '''

    TRAIN_URL = 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'
    TEST_URL = 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0'
    LABELS_URL = 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'

    def __init__(self, dset_name='mnist_bw', data_dir='data/'):
        super().__init__(dset_name, data_dir)

    def download(self):
        '''
        overrides DataLoader.download() to download bw mnist files
        ''' 
        self._download_file(self.TRAIN_URL, 'mnist_bw.npy')
        self._download_file(self.TEST_URL, 'mnist_bw_te.npy')
        self._download_file(self.LABELS_URL, 'mnist_bw_y_te.npy')

    def load(self):
        '''
        overrides DataLoader.load() to load .npy files

        Returns: 
            train_data, test_data, test_labels (all as numpy arrays)
        ''' 
        train_path = os.path.join(self.data_dir, 'mnist_bw.npy')
        test_path = os.path.join(self.data_dir, 'mnist_bw_te.npy')
        labels_path = os.path.join(self.data_dir, 'mnist_bw_y_te.npy')

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(labels_path)

        print(f'loaded {self.dset_name}\ntrain: {train_data.shape}\ntest: {test_data.shape}')
        return train_data, test_data, test_labels 

    def preprocess(self, data):
        '''
        overrides DataLoader.preprocess() for bw mnist
        applies normalization and vectorization

        Args: 
            data: raw image data with shape (N, 28, 28)

        Returns: 
            preprocessed data with shape (N, 28x28)
        '''

        data = data.astype(np.float32) / 255.0 #normalization
        data = data.reshape(data.shape[0], -1) #vectorization/flattening

        return data 

class MNISTColorDataLoader(DataLoader):
    '''
    dataloader for color mnist dataset 
    overrides abstract methods from Dataloader

        download: downloads data
        load: loads the data into train, test and test labels 
        preprocess: preprocesses the data (suitable to color version)
    '''
    
    TRAIN_URL = 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'
    TEST_URL = 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'
    LABELS_URL = 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'

    def __init__(self, dset_name='mnist_color', data_dir='data/', color_version='m0'):
        '''
        Args: 
            dset_name: name of dataset 
            data_dir: dir for data storage 
            color_version: which color version to use (m0-4)
        '''
        super().__init__(dset_name, data_dir)
        self.color_version = color_version

    def download(self):
        '''
        overrides DataLoader.download() to download color mnist files
        ''' 
        self._download_file(self.TRAIN_URL, 'mnist_color.pkl')
        self._download_file(self.TEST_URL, 'mnist_color_te.pkl')
        self._download_file(self.LABELS_URL, 'mnist_color_y_te.npy')

    def load(self):
        '''
        overrides DataLoader.load() to load .npy files

        Returns: 
            train_data, test_data, test_labels
        ''' 
        train_path = os.path.join(self.data_dir, 'mnist_color.pkl')
        test_path = os.path.join(self.data_dir, 'mnist_color_te.pkl')
        labels_path = os.path.join(self.data_dir, 'mnist_color_y_te.npy')

        #load pickle files and extract specific color version
        with open(train_path, 'rb') as f:
            train_dict = pickle.load(f)
        train_data = train_dict[self.color_version]
        
        with open(test_path, 'rb') as f:
            test_dict = pickle.load(f)
        test_data = test_dict[self.color_version]

        test_labels = np.load(labels_path)

        print(f'loaded {self.dset_name} with color version: {self.color_version}\ntrain shape: {train_data.shape}\ntest shape: {test_data.shape}')
        return train_data, test_data, test_labels 

    def preprocess(self, data):
        '''
        overrides DataLoader.preprocess() for color mnist
        data is already normalized so just cast to float32 for type safety

        Args: 
            data: raw image data with shape (N, 28, 28, 3)

        Returns: 
            preprocessed data with shape (N, 28, 28, 3)
        '''
        data = data.astype(np.float32)
        return data 