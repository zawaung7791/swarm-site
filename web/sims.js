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

// Galaxy: a dense central bulge plus three spiral arms, each particle
// moving at the speed of a circular orbit.
function galaxySetup() {
  return { mode: 0, extra: false };
}
function galaxyParticle(rand, i, c, j) {
  const TAU = Math.PI * 2;
  let r, ang;
  if (rand() < 0.15) {
    r = Math.abs(rand() + rand() + rand() - 1.5) * 0.25 + 0.01;
    ang = rand() * TAU;
  } else {
    r = 0.05 + Math.pow(rand(), 0.8) * 0.95;
    const spread = (rand() + rand() + rand() - 1.5) * 0.45;
    ang = (i % 3) * TAU / 3 + r * 4.4 + spread;
  }
  const cs = Math.cos(ang), sn = Math.sin(ang);
  c.x[j] = r * cs;
  c.y[j] = r * sn;
  const v = Math.sqrt((r * r) / Math.pow(r * r + EPS2, 1.5)) * (1 + (rand() - 0.5) * 0.1);
  c.vx[j] = -sn * v;
  c.vy[j] = cs * v;
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
const FIELDS = ['x', 'y', 'vx', 'vy', 'ox', 'oy', 'svx', 'svy', 'age', 'life'];
const BASIC = ['x', 'y', 'vx', 'vy']; // every simulation needs these

// Generates the starting particles in chunks, calling onChunk(start, chunk)
// for each. Chunks keep memory use down for the GPU lane's huge counts.
// Every lane gets exactly the same particles.
export function generate(sim, n, chunkSize, onChunk, seed = SEED) {
  const rand = mulberry32(seed);
  const kind = sim === 'fireworks' ? fireworksSetup(rand, n)
    : sim === 'flock' ? flockSetup(rand)
    : galaxySetup();
  for (let start = 0; start < n; start += chunkSize) {
    const len = Math.min(chunkSize, n - start);
    const c = { mode: kind.mode };
    for (const k of FIELDS) {
      const needed = BASIC.includes(k) || kind.extra;
      c[k] = needed ? new Float32Array(len) : EMPTY;
    }
    for (let j = 0; j < len; j++) {
      const i = start + j;
      if (kind.mode === 1) fireworksParticle(rand, i, c, j, kind);
      else if (kind.mode === 2) flockParticle(rand, i, c, j, kind);
      else galaxyParticle(rand, i, c, j);
    }
    onChunk(start, c);
  }
}

// All the particles in one go, for the CPU lanes.
export function makeState(sim, n, seed = SEED) {
  let state = null;
  generate(sim, n, Math.max(1, n), (start, c) => { state = c; }, seed);
  return state;
}

// Fireworks: particles belong to bursts. Each burst has a fixed centre,
// launch speed and lifetime, and restarts when its lifetime runs out.
// Bursts start at staggered times, waiting off screen until then.
function fireworksSetup(rand, n) {
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
  return { mode: 1, extra: true, bursts };
}
function fireworksParticle(rand, i, c, j, kind) {
  const b = kind.bursts[i % kind.bursts.length];
  const ang = rand() * Math.PI * 2;
  const sp = b.shell ? b.speed * (0.72 + 0.28 * Math.sqrt(rand())) : b.speed * Math.sqrt(rand());
  c.ox[j] = b.cx; c.oy[j] = b.cy;
  c.svx[j] = Math.cos(ang) * sp; c.svy[j] = Math.sin(ang) * sp;
  c.life[j] = b.life;
  c.age[j] = b.life - b.delay; // reaches its lifetime, and launches, after the delay
  c.x[j] = 99; c.y[j] = 99;    // parked off screen until then
}

// Flock: birds start in a handful of loose groups, each heading its own way.
function flockSetup(rand) {
  const groups = [];
  for (let g = 0; g < 7; g++) {
    groups.push({ cx: rand() * 2.6 - 1.3, cy: rand() * 1.8 - 0.9, heading: rand() * Math.PI * 2 });
  }
  return { mode: 2, extra: false, groups };
}
function flockParticle(rand, i, c, j, kind) {
  const gauss = () => rand() + rand() + rand() - 1.5;
  const g = kind.groups[i % kind.groups.length];
  c.x[j] = g.cx + gauss() * 0.35;
  c.y[j] = g.cy + gauss() * 0.25;
  const h = g.heading + gauss() * 0.3;
  c.vx[j] = Math.cos(h) * 0.45;
  c.vy[j] = Math.sin(h) * 0.45;
}
