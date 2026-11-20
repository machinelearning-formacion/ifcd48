import json, subprocess
import numpy as np
import keras

x = np.load("prueba_x.npy")               # 1000 imágenes de prueba
y = np.load("prueba_y.npy")
red = keras.models.load_model("mnist_cnn.keras")

def en_js(imagenes):
    """Ejecuta la red de JavaScript (red.js) con Node.js."""
    with open("imgs.json", "w") as f:
        json.dump(imagenes.reshape(len(imagenes), 784).tolist(), f)
    r = subprocess.run(["node", "lote.js", "modelo.json",
                        "imgs.json"], capture_output=True, text=True)
    s = json.loads(r.stdout)
    return np.array(s["p"]), s["ms"]

p_js, ms = en_js(x)
p_py = red.predict(x[..., None], verbose=0)

def test_misma_prediccion_que_keras():
    assert (p_js.argmax(1) == p_py.argmax(1)).all()
    assert np.abs(p_js - p_py).max() < 1e-3

def test_acierto_minimo():
    assert np.mean(p_js.argmax(1) == y) >= 0.98

def test_probabilidades_validas():
    assert np.allclose(p_js.sum(1), 1)
    p_vacia, _ = en_js(np.zeros((1, 28, 28)))  # lienzo en blanco
    assert np.isfinite(p_vacia).all()

def test_tiempo_de_respuesta():
    assert ms < 50                     # milisegundos por imagen

if __name__ == "__main__":
    print(f"coinciden con Keras: "
          f"{np.mean(p_js.argmax(1) == p_py.argmax(1)):.3f}; "
          f"diferencia máxima {np.abs(p_js - p_py).max():.1e}")
    print(f"acierto en JavaScript: {np.mean(p_js.argmax(1) == y):.3f}"
          f"; {ms:.1f} ms por imagen")
