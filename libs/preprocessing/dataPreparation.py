import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
import os
import itertools as ITT
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

class PlantDiseasesDataset:


    def __init__(self, train_dir, test_dir, 
                 split_size=60, from_folder=False,
                 use_multiprocess=True, **kwargs):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.split_size = split_size / 100
        self.use_multiprocess = use_multiprocess
        self.train_data, self.test_data = (list() for _ in range(2))

        if not(os.path.exists(self.train_dir) and os.path.exists(self.test_dir)):
            raise FileNotFoundError('Directories for train or test do not exist!')
        
        if from_folder:
            self.classes = os.listdir(train_dir)
            self.no_classes = len(self.classes)
        else:
            self.classes = kwargs.get("classes")
            self.no_classes = len(self.classes)
        
        if not all([item_1 == item_2 for item_1, item_2 in zip(os.listdir(self.train_dir), os.listdir(self.test_dir))]):
            raise ValueError("Classes in test directory doesn't match with classes in train directory")
        
        self.dataset()
    
    def dataset(self):

        class_path = lambda dir: [os.path.join(dir, class_path) for class_path in self.classes]

        for images_folder_test, images_folder_train in ITT.zip_longest(class_path(self.test_dir), class_path(self.train_dir)):
            if images_folder_train is not None:
                images_data = [os.path.join(images_folder_train, img) for img in os.listdir(images_folder_train)]
                self.train_data.extend(images_data)
            
            if images_folder_test is not None:
                images_data = [os.path.join(images_folder_test, img) for img in os.listdir(images_folder_test)]
                self.test_data.extend(images_data)
            
    def _run_multithread(func):

        def decorator(self, data, shp):

            if self.use_multiprocess:
                pool = Pool(processes=4)
                res = pool.apipe(partial(func, self=self, data=data, shp=shp))
                return res.get()
            else:
                images, labels = func(self, data, shp)
                return images, labels
        
        return decorator
                

    @_run_multithread
    def populate_dataset(self, data: str(), shp: tuple=(150, 150, 3)):

        assert data in ['train_data', 'test_data'], "Invalid input"
        
        data_in = getattr(self, data)
        
        images = np.empty((len(data_in),) + shp, dtype='uint8')
        labels = np.empty((len(data_in), self.no_classes), dtype='uint8')

        for image_path in tqdm(data_in):
            image = Image.open(image_path).convert("RGB")
            image = image.resize(shp[:-1])
            image_array = np.array(image).astype('uint8').reshape((1,) + shp)
            label = np.array(to_categorical(self.classes.index(image_path.split('/')[-2]), self.no_classes)) \
                        .reshape(1, -1)

            np.append(images, image_array)
            np.append(labels, label)

        return images, labels