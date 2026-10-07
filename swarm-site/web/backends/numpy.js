// The NumPy lane: real Python (Pyodide, CPython compiled to WebAssembly)
// running the same three simulations with NumPy whole-array operations.

import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.mjs';

export const PY = `
import numpy as np

f32 = np.float32
EPS2 = f32(0.0025)
HOLE_MASS = f32(0.6)    # galaxy black holes
FW_GRAVITY = f32(0.55)  # fireworks: downward pull
FW_DRAG = f32(1.5)      # fireworks: air resistance
FW_HOLE = f32(0.15)     # fireworks: black hole strength
FL_SPEED = f32(0.45)    # flock: cruising speed
FL_ALIGN = f32(2.0)     # flock: match neighbours' heading
FL_COHERE = f32(0.3)    # flock: move towards neighbours
FL_SEPARATE = f32(0.06) # flock: avoid crowded cells
FL_WIND = f32(0.25)     # flock: gentle swirling breeze
FL_FLEE = f32(0.05)     # flock: fear of hawks
FL_CELL = 0.1           # flock: neighbourhood size

# Fast sine, accurate to about 0.1%. All three languages use this same
# approximation for the flock's breeze.
TAU = f32(2 * np.pi)
INV_TAU = f32(1 / (2 * np.pi))
HALF_PI = f32(np.pi / 2)
SIN_B = f32(4 / np.pi)
SIN_C = f32(-4 / np.pi ** 2)
SIN_P = f32(0.225)

def fast_sin(v):
    v = v - TAU * np.round(v * INV_TAU)  # bring into -pi..pi
    y = SIN_B * v + SIN_C * v * np.abs(v)
    return SIN_P * (y * np.abs(y) - y) + y

MODE = 0
T = f32(0)
x = y = vx = vy = ox = oy = svx = svy = age = life = None
W = H = 0
SCALE = f32(1)
WX = WY = f32(1)
GW = GH = 1
LUT = None
G = f32(1.0)
HOLES = np.zeros((0, 2), dtype=f32)

def _f32(buf):
    return np.frombuffer(buf.to_bytes(), dtype=f32).copy()

def init(mode, jx, jy, jvx, jvy, jox, joy, jsvx, jsvy, jage, jlife):
    global MODE, T, x, y, vx, vy, ox, oy, svx, svy, age, life
    MODE = int(mode)
    T = f32(0)
    x, y, vx, vy = _f32(jx), _f32(jy), _f32(jvx), _f32(jvy)
    ox, oy, svx, svy = _f32(jox), _f32(joy), _f32(jsvx), _f32(jsvy)
    age, life = _f32(jage), _f32(jlife)

def set_size(w, h, jlut):
    global W, H, LUT, SCALE, WX, WY, GW, GH
    W, H = int(w), int(h)
    LUT = np.frombuffer(jlut.to_bytes(), dtype=np.uint32).copy()
    SCALE = f32(min(W, H) * 0.42)
    WX = f32(W * 0.5 / SCALE)
    WY = f32(H * 0.5 / SCALE)
    GW = max(1, int(2 * WX / FL_CELL))
    GH = max(1, int(2 * WY / FL_CELL))

def set_params(g, jholes):
    global G, HOLES
    G = f32(g)
    HOLES = _f32(jholes).reshape(-1, 2)

def step(dt):
    dt = f32(dt)
    if MODE == 1:
        step_fireworks(dt)
    elif MODE == 2:
        step_flock(dt)
    else:
        step_galaxy(dt)

def step_galaxy(dt):
    global x, y, vx, vy
    r2 = x * x + y * y + EPS2
    f = G / (r2 * np.sqrt(r2))
    ax = -x * f
    ay = -y * f
    hg = HOLE_MASS * G
    for hx, hy in HOLES:
        dx = hx - x
        dy = hy - y
        q2 = dx * dx + dy * dy + EPS2
        q = hg / (q2 * np.sqrt(q2))
        ax += dx * q
        ay += dy * q
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt

def step_fireworks(dt):
    global x, y, vx, vy, age, T
    age += dt
    done = age >= life
    if done.any():
        age[done] -= life[done]
        # Relaunch somewhere a little different each time
        x[done] = ox[done] + f32(0.45) * np.sin(f32(1.3) * T + f32(7) * ox[done])
        y[done] = oy[done] + f32(0.2) * np.cos(f32(0.9) * T + f32(5) * oy[done])
        vx[done] = svx[done]
        vy[done] = svy[done]
    ax = np.zeros_like(x)
    ay = np.full_like(y, FW_GRAVITY * G)
    for hx, hy in HOLES:
        dx = hx - x
        dy = hy - y
        q2 = dx * dx + dy * dy + EPS2
        q = FW_HOLE / (q2 * np.sqrt(q2))
        ax += dx * q
        ay += dy * q
    drag = f32(1) - FW_DRAG * dt
    vx += ax * dt
    vy += ay * dt
    vx *= drag
    vy *= drag
    x += vx * dt
    y += vy * dt
    T = f32(T + dt)

def step_flock(dt):
    global x, y, vx, vy, T
    # Wrap around the edges of the screen
    x = np.mod(x + WX, 2 * WX) - WX
    y = np.mod(y + WY, 2 * WY) - WY

    # Sort birds into grid cells and count them, with their total velocity
    gx = np.minimum(((x + WX) * f32(GW / (2 * WX))).astype(np.int32), GW - 1)
    gy = np.minimum(((y + WY) * f32(GH / (2 * WY))).astype(np.int32), GH - 1)
    k = gy * GW + gx
    cells = GW * GH
    cnt = np.bincount(k, minlength=cells).astype(f32).reshape(GH, GW)
    svx = np.bincount(k, weights=vx, minlength=cells).astype(f32).reshape(GH, GW)
    svy = np.bincount(k, weights=vy, minlength=cells).astype(f32).reshape(GH, GW)

    # Smooth over each cell and its 8 neighbours, so the grid doesn't show
    def blur(a):
        a = a + np.roll(a, 1, 1) + np.roll(a, -1, 1)
        return a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
    bc = blur(cnt)
    avx = (blur(svx) / np.maximum(bc, f32(1))).ravel()
    avy = (blur(svy) / np.maximum(bc, f32(1))).ravel()

    # Drift towards busier areas nearby, but away from crowding right here
    mean = f32(len(x) / cells + 1)
    d = bc / f32(9) / mean
    crowd = d * (FL_COHERE - FL_SEPARATE * d)
    fx = (np.roll(crowd, -1, 1) - np.roll(crowd, 1, 1)).ravel()
    fy = (np.roll(crowd, -1, 0) - np.roll(crowd, 1, 0)).ravel()

    ax = (avx[k] - vx) * FL_ALIGN + fx[k] + FL_WIND * fast_sin(f32(2.3) * y + f32(0.6) * T)
    ay = (avy[k] - vy) * FL_ALIGN + fy[k] + FL_WIND * fast_sin(f32(1.9) * x - f32(0.4) * T + HALF_PI)

    # Hawks: birds flee from them
    for hx, hy in HOLES:
        dx = x - hx
        dy = y - hy
        q2 = dx * dx + dy * dy + f32(0.002)
        q = np.where(q2 < f32(0.16), FL_FLEE / q2, f32(0))
        ax += dx * q
        ay += dy * q

    vx += ax * dt
    vy += ay * dt
    # Ease every bird back towards cruising speed
    sp = np.sqrt(vx * vx + vy * vy) + f32(1e-6)
    s = f32(1) + f32(0.08) * (FL_SPEED * G / sp - f32(1))
    vx *= s
    vy *= s
    x += vx * dt
    y += vy * dt
    T = f32(T + dt)

def draw():
    fx = f32(W * 0.5) + x * SCALE
    fy = f32(H * 0.5) + y * SCALE
    m = (fx >= 0) & (fx < W) & (fy >= 0) & (fy < H)
    idx = fy[m].astype(np.int32) * W + fx[m].astype(np.int32)
    if MODE == 1:
        # Fresh sparks shine brighter, then fade out completely
        wgt = np.minimum(f32(3), np.floor(f32(4) * (f32(1) - age[m] / life[m])))
        dens = np.bincount(idx, weights=wgt, minlength=W * H).astype(np.int64)
    else:
        dens = np.bincount(idx, minlength=W * H)
    np.minimum(dens, 255, out=dens)
    return LUT[dens]
`;

export async function create() {
  const py = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/' });
  await py.loadPackage('numpy');
  py.runPython(PY);
  const fn = name => py.globals.get(name);
  const pyInit = fn('init'), pySize = fn('set_size'), pyParams = fn('set_params');
  const pyStep = fn('step'), pyDraw = fn('draw');
  let w = 0, h = 0, proxy = null, buf = null;

  const release = () => {
    if (buf) { buf.release(); buf = null; }
    if (proxy) { proxy.destroy(); proxy = null; }
  };

  return {
    init(st) {
      pyInit(st.mode, st.x, st.y, st.vx, st.vy, st.ox, st.oy, st.svx, st.svy, st.age, st.life);
    },
    setSize(W, H, lut) { w = W; h = H; release(); pySize(W, H, lut); },
    setParams(p) { pyParams(p.g, new Float32Array(p.holes.flat())); },
    step(dt) { pyStep(dt); },
    draw() {
      release();
      proxy = pyDraw();
      buf = proxy.getBuffer('u8');
      const d = buf.data;
      return new Uint8ClampedArray(d.buffer, d.byteOffset, w * h * 4);
    },
  };
}
