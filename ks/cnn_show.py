from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from ks.cnn_dos import get_model

img_path = '/home/lei/Pycharmprojects/dog-cat/test/cats/cat.1235.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

model = get_model()
layer_outputs = [layer.output for layer in model.layers[:8]]
layer_names = [layer.name for layer in model.layers[:8]]

active_model = models.Model(inputs=model.input, outputs=layer_outputs)
actives = active_model.predict(img_tensor)

img_per_row = 16
for layer_name, layer_active in zip(layer_names, actives):
    n_features = layer_active.shape[-1]

    size = layer_active.shape[1]
    n_cols = n_features // img_per_row
    display_grid = np.zeros((size * n_cols, img_per_row * size))

    for col in range(n_cols):
        for row in range(img_per_row):
            channel_image = layer_active[0, :, :, col * img_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))

    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()