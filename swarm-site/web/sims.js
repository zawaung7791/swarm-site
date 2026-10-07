// Builds the starting particles for each simulation. Every lane starts
// from exactly the same particles, because they all use these functions
// with the same seed.

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

// ---------- all simulations ----------

export const MODES = { galaxy: 0, fireworks: 1, flock: 2 };

// Simulation time per frame. The galaxy uses its own time units.
export const DT = { galaxy: 0.003, fireworks: 1 / 60, flock: 1 / 60 };

const EMPTY = new Float32Array(0);

export function makeState(sim, n, seed = SEED) {
  if (sim === 'fireworks') return makeFireworks(n, seed);
  if (sim === 'flock') return makeFlock(n, seed);
  return { mode: 0, ...makeGalaxy(n, seed), ox: EMPTY, oy: EMPTY, svx: EMPTY, svy: EMPTY, age: EMPTY, life: EMPTY };
}

// Fireworks: particles belong to bursts. Each burst has a fixed centre,
// launch speed and lifetime, and restarts when its lifetime runs out.
// Bursts start at staggered times, waiting off screen until then.
function makeFireworks(n, seed) {
  const rand = mulberry32(seed);
  const TAU = Math.PI * 2;
  const B = Math.max(8, Math.min(24, Math.round(n / 12000)));
  const bursts = [];
  for (let b = 0; b < B; b++) {
    const life = 1.6 + rand() * 1.0;
    bursts.push({
      cx: (rand() * 2 - 1) * 1.1,
      cy: -0.75 + rand() * 0.7,
      speed: 0.6 + rand() * 0.5,
      life,
      delay: b < 3 ? b * 0.3 : rand() * life * 1.2,
      shell: rand() < 0.6,
    });
  }
  const f = () => new Float32Array(n);
  const st = { mode: 1, x: f(), y: f(), vx: f(), vy: f(), ox: f(), oy: f(), svx: f(), svy: f(), age: f(), life: f() };
  for (let i = 0; i < n; i++) {
    const b = bursts[i % B];
    const ang = rand() * TAU;
    const sp = b.shell ? b.speed * (0.72 + 0.28 * Math.sqrt(rand())) : b.speed * Math.sqrt(rand());
    st.ox[i] = b.cx; st.oy[i] = b.cy;
    st.svx[i] = Math.cos(ang) * sp; st.svy[i] = Math.sin(ang) * sp;
    st.life[i] = b.life;
    st.age[i] = b.life - b.delay; // reaches its lifetime, and launches, after the delay
    st.x[i] = 99; st.y[i] = 99;   // parked off screen until then
  }
  return st;
}

// Flock: birds start in a handful of loose groups, each heading its own way.
function makeFlock(n, seed) {
  const rand = mulberry32(seed);
  const TAU = Math.PI * 2;
  const gauss = () => rand() + rand() + rand() - 1.5;
  const groups = [];
  for (let g = 0; g < 7; g++) {
    groups.push({ cx: rand() * 2.6 - 1.3, cy: rand() * 1.8 - 0.9, heading: rand() * TAU });
  }
  const f = () => new Float32Array(n);
  const st = { mode: 2, x: f(), y: f(), vx: f(), vy: f(), ox: EMPTY, oy: EMPTY, svx: EMPTY, svy: EMPTY, age: EMPTY, life: EMPTY };
  for (let i = 0; i < n; i++) {
    const g = groups[i % groups.length];
    st.x[i] = g.cx + gauss() * 0.35;
    st.y[i] = g.cy + gauss() * 0.25;
    const h = g.heading + gauss() * 0.3;
    st.vx[i] = Math.cos(h) * 0.45;
    st.vy[i] = Math.sin(h) * 0.45;
  }
  return st;
}
