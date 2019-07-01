import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    original_dataset_dir = '/home/lei/Pycharmprojects/dogs-vs-cats/train'

    base_dir = '/home/lei/Pycharmprojects/dog-cat'
    os.mkdir(base_dir) if not os.path.exists(base_dir) else print()

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir) if not os.path.exists(train_dir) else print()

    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir) if not os.path.exists(validation_dir) else print()
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir) if not os.path.exists(test_dir) else print()

    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir) if not os.path.exists(train_cats_dir) else print()

    train_dogs_dir = os.path.join(train_dir, 'dos')
    os.mkdir(train_dogs_dir) if not os.path.exists(train_dogs_dir) else print()

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir) if not os.path.exists(validation_cats_dir) else print()

    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir) if not os.path.exists(validation_dogs_dir) else print()

    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir) if not os.path.exists(test_cats_dir) else print()

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir) if not os.path.exists(test_dogs_dir) else print()

    n = 1000
    for i, dir_ in enumerate([train_cats_dir, validation_cats_dir, test_cats_dir]):
        fnames = ['cat.{}.jpg'.format(i) for i in range(n - 1000, n)]
        n += 500
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(dir_, fname)
            shutil.copy(src, dst)

    n = 1000
    for i, dir_ in enumerate([train_dogs_dir, validation_dogs_dir, test_dogs_dir]):
        fnames = ['dog.{}.jpg'.format(i) for i in range(n, n+500)]
        n += 500
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(dir_, fname)
            shutil.copy(src, dst)


    model = get_model()

    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data_gen.flow_from_directory(train_dir,
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='binary')
    validation_generator = train_data_gen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=20,
                                                              class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )
    model.save('cats_dog_small_1.h5')
