from keras import layers
from keras.optimizers import RMSprop
from keras import models
import libs.preprocessing.dataPreparation as DP
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from libs.architectures import Resnet
from libs import misc
import sys, os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

# Load Tensorboard callback
tensorboard = TensorBoard(
    log_dir=os.path.join(os.getcwd(), "logs"),
    histogram_freq=1,
    write_images=True
)

# Save a model checkpoint after every epoch
checkpoint = ModelCheckpoint(
    os.path.join(os.getcwd(), "model_checkpoint"),
    save_freq="epoch"
)

callbacks = [tensorboard, checkpoint]

if __name__ == "__main__":
    config = misc.load_config(sys.argv[1])
    datagen = ImageDataGenerator(rescale=1./255)
    # dataset = DP.PlantDiseasesDataset(train_dir=config.get("train_dir"), 
    #                                 test_dir=config.get("test_dir"),
    #                                 from_folder=True, 
    #                                 classes=config.get("classes"),
    #                                 use_multiprocess=False)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = np.asarray([to_categorical(y, num_classes=len(config.get("classes")))[0] for y in y_train])
    y_test = np.asarray([to_categorical(y, num_classes=len(config.get("classes")))[0] for y in y_test])

    train_generator = datagen.flow(x_train, y_train, batch_size=20)
    test_generator = datagen.flow(x_test, y_test, batch_size=20)
    resnet = Resnet(input_shape=config.get("shape"), n=config.get("stack_n"), **config)
    history = resnet().fit(x=x_train, 
                           y=y_train,
                           callbacks=[callbacks],
                           validation_split=config.get("validation_split"),
                           epochs=config.get("number_of_epochs"))

    resnet().save(os.path.join('./models/', 'trained_model.h5'))
    
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label="Training acc")
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.savefig("acc_per_epoch.png")

    plt.figure()

    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.savefig("loss_per_epoch.png")