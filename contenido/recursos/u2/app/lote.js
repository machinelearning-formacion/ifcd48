// Uso: node lote.js modelo.json imagenes.json -> probabilidades
const fs = require("fs");
const { predecir } = require("./red.js");
const modelo = JSON.parse(fs.readFileSync(process.argv[2]));
const imgs = JSON.parse(fs.readFileSync(process.argv[3]));
const t0 = performance.now();
const p = imgs.map(img => predecir(modelo, img));
const ms = (performance.now() - t0) / imgs.length;
console.log(JSON.stringify({ p, ms }));
