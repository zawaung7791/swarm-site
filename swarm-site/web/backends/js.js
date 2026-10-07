// The JavaScript lane: the same three simulations as the Rust lane,
// written the fast way (typed arrays, no objects per particle).

const EPS2 = 0.0025;
const HOLE_MASS = 0.6;     // galaxy black holes
const FW_GRAVITY = 0.55;   // fireworks: downward pull
const FW_DRAG = 1.5;       // fireworks: air resistance
const FW_HOLE = 0.15;      // fireworks: black hole strength
const FL_SPEED = 0.45;     // flock: cruising speed
const FL_ALIGN = 2.0;      // flock: match neighbours' heading
const FL_COHERE = 0.3;     // flock: drift towards busier areas
const FL_SEPARATE = 0.06;  // flock: avoid crowding
const FL_WIND = 0.25;      // flock: gentle swirling breeze
const FL_FLEE = 0.05;      // flock: fear of hawks
const FL_CELL = 0.1;       // flock: neighbourhood size

// Fast sine, accurate to about 0.1%. All three languages use this same
// approximation for the flock's breeze.
const TAU = Math.PI * 2, INV_TAU = 1 / TAU, HALF_PI = Math.PI / 2;
const SIN_B = 4 / Math.PI, SIN_C = -4 / (Math.PI * Math.PI), SIN_P = 0.225;
function fastSin(v) {
  v -= TAU * Math.round(v * INV_TAU); // bring into -pi..pi
  const y = SIN_B * v + SIN_C * v * Math.abs(v);
  return SIN_P * (y * Math.abs(y) - y) + y;
}

export async function create() {
  let n = 0, mode = 0, T = 0;
  let x, y, vx, vy, ox, oy, svx, svy, age, life;
  let w = 0, h = 0, scale = 1, WX = 1, WY = 1, GW = 1, GH = 1;
  let dens, pixels, pixels8, lut;
  let cnt, gvx, gvy, tmp, bc, bvx, bvy, crowd, gfx, gfy, cellOf;
  let g = 1, holes = [];

  function allocGrid() {
    const c = GW * GH, f = () => new Float32Array(c);
    cnt = f(); gvx = f(); gvy = f(); tmp = f(); bc = f(); bvx = f(); bvy = f();
    crowd = f(); gfx = f(); gfy = f();
  }

  // Box blur over each cell and its 8 neighbours, wrapping at the edges
  function blur(src, dst) {
    for (let j = 0; j < GH; j++) {
      const row = j * GW;
      for (let i = 0; i < GW; i++) {
        tmp[row + i] = src[row + i] + src[row + (i + GW - 1) % GW] + src[row + (i + 1) % GW];
      }
    }
    for (let j = 0; j < GH; j++) {
      const up = ((j + GH - 1) % GH) * GW, down = ((j + 1) % GH) * GW, row = j * GW;
      for (let i = 0; i < GW; i++) dst[row + i] = tmp[row + i] + tmp[up + i] + tmp[down + i];
    }
  }

  function stepGalaxy(dt) {
    const nh = holes.length, hg = HOLE_MASS * g;
    for (let i = 0; i < n; i++) {
      const px = x[i], py = y[i];
      const r2 = px * px + py * py + EPS2;
      const f = g / (r2 * Math.sqrt(r2));
      let ax = -px * f, ay = -py * f;
      for (let k = 0; k < nh; k++) {
        const dx = holes[k][0] - px, dy = holes[k][1] - py;
        const q2 = dx * dx + dy * dy + EPS2;
        const q = hg / (q2 * Math.sqrt(q2));
        ax += dx * q; ay += dy * q;
      }
      const nvx = vx[i] + ax * dt, nvy = vy[i] + ay * dt;
      vx[i] = nvx; vy[i] = nvy;
      x[i] = px + nvx * dt; y[i] = py + nvy * dt;
    }
  }

  function stepFireworks(dt) {
    const nh = holes.length, drag = 1 - FW_DRAG * dt, ay0 = FW_GRAVITY * g;
    for (let i = 0; i < n; i++) {
      let a = age[i] + dt, px = x[i], py = y[i], pvx = vx[i], pvy = vy[i];
      if (a >= life[i]) {
        // Relaunch somewhere a little different each time
        a -= life[i];
        px = ox[i] + 0.45 * Math.sin(1.3 * T + 7 * ox[i]);
        py = oy[i] + 0.2 * Math.cos(0.9 * T + 5 * oy[i]);
        pvx = svx[i]; pvy = svy[i];
      }
      age[i] = a;
      let ax = 0, ay = ay0;
      for (let k = 0; k < nh; k++) {
        const dx = holes[k][0] - px, dy = holes[k][1] - py;
        const q2 = dx * dx + dy * dy + EPS2;
        const q = FW_HOLE / (q2 * Math.sqrt(q2));
        ax += dx * q; ay += dy * q;
      }
      pvx = (pvx + ax * dt) * drag;
      pvy = (pvy + ay * dt) * drag;
      vx[i] = pvx; vy[i] = pvy;
      x[i] = px + pvx * dt; y[i] = py + pvy * dt;
    }
    T += dt;
  }

  function stepFlock(dt) {
    const cells = GW * GH, LX = 2 * WX, LY = 2 * WY;
    const sx = GW / LX, sy = GH / LY;
    cnt.fill(0); gvx.fill(0); gvy.fill(0);

    // Wrap at the screen edges, then sort birds into grid cells
    for (let i = 0; i < n; i++) {
      let px = x[i], py = y[i];
      if (px < -WX || px >= WX) px = (((px + WX) % LX) + LX) % LX - WX;
      if (py < -WY || py >= WY) py = (((py + WY) % LY) + LY) % LY - WY;
      x[i] = px; y[i] = py;
      const gx = Math.min(((px + WX) * sx) | 0, GW - 1);
      const gy = Math.min(((py + WY) * sy) | 0, GH - 1);
      const k = gy * GW + gx;
      cellOf[i] = k;
      cnt[k] += 1; gvx[k] += vx[i]; gvy[k] += vy[i];
    }

    // Smoothed neighbourhood: average heading, and a crowding field
    blur(cnt, bc); blur(gvx, bvx); blur(gvy, bvy);
    const mean = n / cells + 1;
    for (let k = 0; k < cells; k++) {
      const inv = 1 / Math.max(bc[k], 1);
      bvx[k] *= inv; bvy[k] *= inv;
      const d = bc[k] / 9 / mean;
      crowd[k] = d * (FL_COHERE - FL_SEPARATE * d);
    }
    for (let j = 0; j < GH; j++) {
      const row = j * GW, up = ((j + GH - 1) % GH) * GW, down = ((j + 1) % GH) * GW;
      for (let i = 0; i < GW; i++) {
        gfx[row + i] = crowd[row + (i + 1) % GW] - crowd[row + (i + GW - 1) % GW];
        gfy[row + i] = crowd[down + i] - crowd[up + i];
      }
    }

    const nh = holes.length, target = FL_SPEED * g;
    for (let i = 0; i < n; i++) {
      const k = cellOf[i], px = x[i], py = y[i];
      let pvx = vx[i], pvy = vy[i];
      let ax = (bvx[k] - pvx) * FL_ALIGN + gfx[k] + FL_WIND * fastSin(2.3 * py + 0.6 * T);
      let ay = (bvy[k] - pvy) * FL_ALIGN + gfy[k] + FL_WIND * fastSin(1.9 * px - 0.4 * T + HALF_PI);
      // Hawks: birds flee from them
      for (let m = 0; m < nh; m++) {
        const dx = px - holes[m][0], dy = py - holes[m][1];
        const q2 = dx * dx + dy * dy + 0.002;
        if (q2 < 0.16) { const q = FL_FLEE / q2; ax += dx * q; ay += dy * q; }
      }
      pvx += ax * dt; pvy += ay * dt;
      // Ease back towards cruising speed
      const sp = Math.sqrt(pvx * pvx + pvy * pvy) + 1e-6;
      const s = 1 + 0.08 * (target / sp - 1);
      pvx *= s; pvy *= s;
      vx[i] = pvx; vy[i] = pvy;
      x[i] = px + pvx * dt; y[i] = py + pvy * dt;
    }
    T += dt;
  }

  return {
    init(st) {
      mode = st.mode; T = 0; n = st.x.length;
      x = st.x.slice(); y = st.y.slice(); vx = st.vx.slice(); vy = st.vy.slice();
      ox = st.ox.slice(); oy = st.oy.slice(); svx = st.svx.slice(); svy = st.svy.slice();
      age = st.age.slice(); life = st.life.slice();
      cellOf = new Int32Array(mode === 2 ? n : 0);
    },

    setSize(W, H, L) {
      w = W; h = H; lut = L;
      scale = Math.min(w, h) * 0.42;
      WX = (w * 0.5) / scale; WY = (h * 0.5) / scale;
      GW = Math.max(1, Math.floor((2 * WX) / FL_CELL));
      GH = Math.max(1, Math.floor((2 * WY) / FL_CELL));
      allocGrid();
      dens = new Uint32Array(w * h);
      pixels = new Uint32Array(w * h);
      pixels8 = new Uint8ClampedArray(pixels.buffer);
    },

    setParams(p) { g = p.g; holes = p.holes; },

    step(dt) {
      if (mode === 1) stepFireworks(dt);
      else if (mode === 2) stepFlock(dt);
      else stepGalaxy(dt);
    },

    draw() {
      const cx = w * 0.5, cy = h * 0.5;
      dens.fill(0);
      if (mode === 1) {
        // Fresh sparks shine brighter, then fade out completely
        for (let i = 0; i < n; i++) {
          const fx = cx + x[i] * scale, fy = cy + y[i] * scale;
          if (fx >= 0 && fx < w && fy >= 0 && fy < h) {
            dens[(fy | 0) * w + (fx | 0)] += Math.min(3, Math.floor(4 * (1 - age[i] / life[i])));
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          const fx = cx + x[i] * scale, fy = cy + y[i] * scale;
          if (fx >= 0 && fx < w && fy >= 0 && fy < h) dens[(fy | 0) * w + (fx | 0)]++;
        }
      }
      for (let i = 0, len = w * h; i < len; i++) {
        const d = dens[i];
        pixels[i] = lut[d > 255 ? 255 : d];
      }
      return pixels8;
    },
  };
}
