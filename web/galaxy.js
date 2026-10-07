// Builds the starting galaxy. Every lane starts from exactly the same
// particles, because they all use this function with the same seed.

export const SEED = 20261002;
const EPS2 = 0.0025;

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeGalaxy(n, seed = SEED) {
  const rand = mulberry32(seed);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const vx = new Float32Array(n);
  const vy = new Float32Array(n);
  const TAU = Math.PI * 2;

  for (let i = 0; i < n; i++) {
    let r, ang;
    if (rand() < 0.15) {
      // Dense central bulge
      r = Math.abs(rand() + rand() + rand() - 1.5) * 0.25 + 0.01;
      ang = rand() * TAU;
    } else {
      // Three spiral arms
      r = 0.05 + Math.pow(rand(), 0.8) * 0.95;
      const spread = (rand() + rand() + rand() - 1.5) * 0.45;
      ang = (i % 3) * TAU / 3 + r * 4.4 + spread;
    }
    const c = Math.cos(ang), s = Math.sin(ang);
    x[i] = r * c;
    y[i] = r * s;
    // Speed for a circular orbit around the central mass, plus a little jitter
    const v = Math.sqrt((r * r) / Math.pow(r * r + EPS2, 1.5)) * (1 + (rand() - 0.5) * 0.1);
    vx[i] = -s * v;
    vy[i] = c * v;
  }
  return { x, y, vx, vy };
}

// Colour lookup table: index = particles in a pixel (0 to 255),
// value = RGBA packed for a little-endian Uint32Array (0xAABBGGRR).
export function buildLut(stops, background = '#0A0E15') {
  const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const bg = rgb(background);
  const cols = stops.map(rgb);
  const lut = new Uint32Array(256);
  const pack = ([r, g, b]) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  lut[0] = pack(bg);
  for (let c = 1; c < 256; c++) {
    const t = Math.min(1, Math.log1p(c) / Math.log1p(24));
    const p = t * (cols.length - 1);
    const i = Math.min(cols.length - 2, Math.floor(p));
    const f = p - i;
    let col = cols[i].map((v, k) => v + (cols[i + 1][k] - v) * f);
    // Single particles fade in from the background so the edges stay soft
    if (c < 3) col = col.map((v, k) => bg[k] + (v - bg[k]) * (0.55 + 0.2 * c));
    lut[c] = pack(col.map(Math.round));
  }
  return lut;
}
