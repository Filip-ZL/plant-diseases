from keras import layers
from keras.optimizers import RMSprop
from keras import models
import dataPreparation as DP
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import param

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Softmax())

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-4),
              metrics=['acc'])


datagen = ImageDataGenerator(rescale=1./255)
dataset = DP.PlantDiseasesDataset(train_dir=param.train_dir, test_dir=param.test_dir, from_folder=True)

train_data, train_labels = dataset.populate_dataset("train_data")
validation_data, validation_labels = dataset.populate_dataset("valid_data")
test_data, test_labels = dataset.populate_dataset("test_data")

train_generator = datagen.flow(train_data, train_labels, batch_size=20)
test_generator = datagen.flow(test_data, test_labels, batch_size=20)
validation_generator = datagen.flow(validation_data, validation_labels, batch_size=20)

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator)

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