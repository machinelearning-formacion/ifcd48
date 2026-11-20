// Predicción de la red convolucional de entrenar.py, sin bibliotecas
function conv(img, w, b) {          // img [alto][ancho][canales]
  const k = w.length, n = img.length - k + 1, sal = [];
  for (let i = 0; i < n; i++) { sal.push([]);
    for (let j = 0; j < n; j++) { const v = b.slice();
      for (let a = 0; a < k; a++) for (let c = 0; c < k; c++)
        img[i + a][j + c].forEach((x, e) =>
          w[a][c][e].forEach((p, f) => { v[f] += x * p; }));
      sal[i].push(v.map(z => Math.max(z, 0)));     // ReLU
    } }
  return sal;
}
function pool(m) {                  // máximo de cada bloque 2x2
  const n = Math.floor(m.length / 2), sal = [];
  for (let i = 0; i < n; i++) { sal.push([]);
    for (let j = 0; j < n; j++)
      sal[i].push(m[2 * i][2 * j].map((_, f) => Math.max(
        m[2 * i][2 * j][f], m[2 * i][2 * j + 1][f],
        m[2 * i + 1][2 * j][f], m[2 * i + 1][2 * j + 1][f])));
  }
  return sal;
}
function densa(x, w, b, act) {
  const z = b.map((bj, j) => x.reduce((s, xi, i) =>
    s + xi * w[i][j], bj));
  if (act === "relu") return z.map(v => Math.max(v, 0));
  const m = Math.max(...z), e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, c) => a + c, 0);
  return e.map(v => v / s);                        // softmax
}
function predecir(modelo, pixeles) {   // 784 valores entre 0 y 1
  let x = [];
  for (let i = 0; i < 28; i++)
    x.push(pixeles.slice(28 * i, 28 * i + 28).map(v => [v]));
  for (const capa of modelo) {
    const [w, b] = capa.pesos;
    if (capa.tipo === "Conv2D") x = conv(x, w, b);
    else if (capa.tipo === "MaxPooling2D") x = pool(x);
    else if (capa.tipo === "Flatten") x = x.flat(2);
    else x = densa(x, w, b, w[0].length === 10 ? "softmax" : "relu");
  }
  return x;                         // 10 probabilidades
}
if (typeof module !== "undefined") module.exports = { predecir };
