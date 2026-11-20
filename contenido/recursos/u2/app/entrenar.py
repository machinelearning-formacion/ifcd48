import json
import numpy as np
import keras
from keras import layers

(x, y), (x_pru, y_pru) = keras.datasets.mnist.load_data()
x, x_pru = x[..., None] / 255.0, x_pru[..., None] / 255.0

keras.utils.set_random_seed(0)
red = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Flatten(), layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")])
red.compile("adam", "sparse_categorical_crossentropy",
            metrics=["accuracy"])
red.fit(x, y, epochs=5, batch_size=128, validation_split=0.1,
        verbose=0)
print(f"parámetros {red.count_params():,}; prueba "
      f"{red.evaluate(x_pru, y_pru, verbose=0)[1]:.4f}")
red.save("mnist_cnn.keras")

# exportar: cada capa con pesos, en el orden en que se aplica
capas = []
for c in red.layers:
    w = c.get_weights()
    capas.append({"tipo": type(c).__name__,
                  "pesos": [np.round(a, 6).tolist() for a in w]})
with open("modelo.json", "w") as f:
    json.dump(capas, f, separators=(",", ":"))
np.save("prueba_x.npy", x_pru[:1000, ..., 0].astype("float32"))
np.save("prueba_y.npy", y_pru[:1000])
print(f"modelo.json: {len(open('modelo.json').read()) / 1e6:.2f} MB")
