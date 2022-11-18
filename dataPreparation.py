import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
import os
import itertools as ITT
from tqdm import tqdm

class PlantDiseasesDataset:


    def __init__(self, train_dir, test_dir, split_size=60, from_folder=False):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.split_size = split_size / 100
        self.train_data, self.valid_data, self.test_data = (list() for _ in range(3))

        if not(os.path.exists(self.train_dir) and os.path.exists(self.test_dir)):
            raise FileNotFoundError('Directories for train or test do not exist!')
        
        if from_folder:
            self.classes = os.listdir(train_dir)
            self.no_classes = len(self.classes)
        
        if not all([item_1 == item_2 for item_1, item_2 in zip(os.listdir(self.train_dir), os.listdir(self.test_dir))]):
            raise ValueError("Classes in test directory doesn't match with classes in train directory")
        
        self.dataset()
    
    def dataset(self):

        class_path = lambda dir: [os.path.join(dir, class_path) for class_path in self.classes]

        for images_folder_test, images_folder_train in ITT.zip_longest(class_path(self.test_dir), class_path(self.train_dir)):
            if images_folder_train is not None:
                images_data = [os.path.join(images_folder_train, img) for img in os.listdir(images_folder_train)]
                split = int(np.floor(self.split_size*len(images_data)))
                self.train_data.extend(images_data[:split])
                self.valid_data.extend(images_data[split:])
            
            if images_folder_test is not None:
                images_data = [os.path.join(images_folder_test, img) for img in os.listdir(images_folder_test)]
                self.test_data.extend(images_data)

    def populate_dataset(self, data:str()):
        assert data in ['train_data', 'test_data', 'valid_data'], "Invalid input"
        
        data_in = getattr(self, data)
        images = np.empty((len(data_in), 150, 150, 3), dtype='uint8')
        labels = np.empty((len(data_in), self.no_classes), dtype='uint8')
        for image_path in tqdm(data_in):
            image = Image.open(image_path).convert("RGB")
            image.resize(150, 150)
            image_array = np.array(image).astype('uint8').reshape(1,150,150,3)
            label = np.array(to_categorical(self.classes.index(image_path.split('/')[-2]), self.no_classes)) \
                      .reshape(1, -1)
            np.append(images, image_array, axis=0)
            np.append(labels, label, axis=0)

        return images, labels