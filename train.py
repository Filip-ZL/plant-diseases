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
    dataset = DP.PlantDiseasesDataset(train_dir=config.get("train_dir"), 
                                    test_dir=config.get("test_dir"),
                                    from_folder=True, 
                                    classes=config.get("classes"),
                                    use_multiprocess=False)

    train_data, train_labels = dataset.populate_dataset("train_data", tuple(config.get("shape")))
    test_data, test_labels = dataset.populate_dataset("test_data", tuple(config.get("shape")))

    train_generator = datagen.flow(train_data, train_labels, batch_size=20)
    test_generator = datagen.flow(test_data, test_labels, batch_size=20)

    resnet = Resnet(input_shape=config.get("shape"), n=config.get("stack_n"), classes=config.get("classes"), kwargs=config)
    history = resnet.fit_data(train_data, 
                              train_labels,
                              callbacks=[callbacks],
                              validation_split=config.get("validation_split"),
                              epochs=config.get("number_of_epochs"))
    
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
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