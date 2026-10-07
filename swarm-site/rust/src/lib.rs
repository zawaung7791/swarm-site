//! Swarm: the Rust lane.
//!
//! Compiled to WebAssembly with no extra tooling: the browser calls these
//! plain exported functions and reads the particle and pixel arrays
//! directly out of this module's memory.
//!
//! Three simulations (mode 0 galaxy, 1 fireworks, 2 flock), each with a
//! SIMD path that handles four particles per instruction. Each frame
//! does the same two jobs as the JavaScript and NumPy lanes:
//!   step(): move every particle
//!   draw(): count particles per pixel, then colour each pixel from a palette

use std::f32::consts::{FRAC_PI_2, TAU};
use std::ptr::addr_of_mut;

const EPS2: f32 = 0.0025; // softening, so nothing divides by zero
const HOLE_MASS: f32 = 0.6; // galaxy black holes
const MAX_HOLES: usize = 8;
const FW_GRAVITY: f32 = 0.55; // fireworks: downward pull
const FW_DRAG: f32 = 1.5; // fireworks: air resistance
const FW_HOLE: f32 = 0.15; // fireworks: black hole strength
const FL_SPEED: f32 = 0.45; // flock: cruising speed
const FL_ALIGN: f32 = 2.0; // flock: match neighbours' heading
const FL_COHERE: f32 = 0.3; // flock: drift towards busier areas
const FL_SEPARATE: f32 = 0.06; // flock: avoid crowding
const FL_WIND: f32 = 0.25; // flock: gentle swirling breeze
const FL_FLEE: f32 = 0.05; // flock: fear of hawks
const FL_CELL: f32 = 0.1; // flock: neighbourhood size

// Fast sine: a polynomial that's accurate to about 0.1%. All three
// languages use this same approximation for the flock's breeze.
const INV_TAU: f32 = 0.159_154_94;
const SIN_B: f32 = 1.273_239_5; // 4 / pi
const SIN_C: f32 = -0.405_284_73; // -4 / pi^2
const SIN_P: f32 = 0.225;

fn fast_sin(x: f32) -> f32 {
    let x = x - TAU * (x * INV_TAU).round(); // bring into -pi..pi
    let y = SIN_B * x + SIN_C * x * x.abs();
    SIN_P * (y * y.abs() - y) + y
}

fn fast_cos(x: f32) -> f32 {
    fast_sin(x + FRAC_PI_2)
}

/// The same fast sine, four values at once.
#[cfg(target_feature = "simd128")]
#[inline(always)]
unsafe fn sin4(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    let turns = f32x4_nearest(f32x4_mul(x, f32x4_splat(INV_TAU)));
    let x = f32x4_sub(x, f32x4_mul(f32x4_splat(TAU), turns));
    let y = f32x4_add(
        f32x4_mul(f32x4_splat(SIN_B), x),
        f32x4_mul(f32x4_mul(f32x4_splat(SIN_C), x), f32x4_abs(x)),
    );
    f32x4_add(f32x4_mul(f32x4_splat(SIN_P), f32x4_sub(f32x4_mul(y, f32x4_abs(y)), y)), y)
}

struct Sim {
    mode: u32,
    t: f32,
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    // fireworks only
    ox: Vec<f32>,
    oy: Vec<f32>,
    svx: Vec<f32>,
    svy: Vec<f32>,
    age: Vec<f32>,
    life: Vec<f32>,
    // flock only: grid of neighbourhood cells
    cell: Vec<u32>,
    cnt: Vec<f32>,
    gvx: Vec<f32>,
    gvy: Vec<f32>,
    tmp: Vec<f32>,
    bc: Vec<f32>,
    bvx: Vec<f32>,
    bvy: Vec<f32>,
    crowd: Vec<f32>,
    gfx: Vec<f32>,
    gfy: Vec<f32>,
    gw: usize,
    gh: usize,
    wx: f32,
    wy: f32,
    dens: Vec<u32>,
    pixels: Vec<u32>,
    lut: [u32; 256],
    w: usize,
    h: usize,
    g: f32,
    holes: Vec<(f32, f32)>,
}

static mut SIM: Sim = Sim {
    mode: 0,
    t: 0.0,
    x: Vec::new(),
    y: Vec::new(),
    vx: Vec::new(),
    vy: Vec::new(),
    ox: Vec::new(),
    oy: Vec::new(),
    svx: Vec::new(),
    svy: Vec::new(),
    age: Vec::new(),
    life: Vec::new(),
    cell: Vec::new(),
    cnt: Vec::new(),
    gvx: Vec::new(),
    gvy: Vec::new(),
    tmp: Vec::new(),
    bc: Vec::new(),
    bvx: Vec::new(),
    bvy: Vec::new(),
    crowd: Vec::new(),
    gfx: Vec::new(),
    gfy: Vec::new(),
    gw: 1,
    gh: 1,
    wx: 1.0,
    wy: 1.0,
    dens: Vec::new(),
    pixels: Vec::new(),
    lut: [0; 256],
    w: 0,
    h: 0,
    g: 1.0,
    holes: Vec::new(),
};

// WebAssembly here is single-threaded, so one global simulation is safe.
fn sim() -> &'static mut Sim {
    unsafe { &mut *addr_of_mut!(SIM) }
}

/// Allocate room for n particles in the given mode. JavaScript then
/// fills the arrays through the ptr_* functions.
#[no_mangle]
pub extern "C" fn init(n: usize, mode: u32) {
    let s = sim();
    s.mode = mode;
    s.t = 0.0;
    s.x = vec![0.0; n];
    s.y = vec![0.0; n];
    s.vx = vec![0.0; n];
    s.vy = vec![0.0; n];
    let extra = if mode == 1 { n } else { 0 };
    s.ox = vec![0.0; extra];
    s.oy = vec![0.0; extra];
    s.svx = vec![0.0; extra];
    s.svy = vec![0.0; extra];
    s.age = vec![0.0; extra];
    s.life = vec![0.0; extra];
    s.cell = vec![0; if mode == 2 { n } else { 0 }];
}

#[no_mangle]
pub extern "C" fn ptr_x() -> *mut f32 {
    sim().x.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_y() -> *mut f32 {
    sim().y.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_vx() -> *mut f32 {
    sim().vx.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_vy() -> *mut f32 {
    sim().vy.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_ox() -> *mut f32 {
    sim().ox.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_oy() -> *mut f32 {
    sim().oy.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_svx() -> *mut f32 {
    sim().svx.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_svy() -> *mut f32 {
    sim().svy.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_age() -> *mut f32 {
    sim().age.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_life() -> *mut f32 {
    sim().life.as_mut_ptr()
}
#[no_mangle]
pub extern "C" fn ptr_lut() -> *mut u32 {
    sim().lut.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn resize(w: usize, h: usize) {
    let s = sim();
    s.w = w;
    s.h = h;
    s.dens = vec![0; w * h];
    s.pixels = vec![0; w * h];

    // Visible area in world units, and the flock's grid over it
    let scale = (w.min(h) as f32) * 0.42;
    s.wx = w as f32 * 0.5 / scale;
    s.wy = h as f32 * 0.5 / scale;
    s.gw = ((2.0 * s.wx / FL_CELL) as usize).max(1);
    s.gh = ((2.0 * s.wy / FL_CELL) as usize).max(1);
    let cells = s.gw * s.gh;
    for v in [
        &mut s.cnt, &mut s.gvx, &mut s.gvy, &mut s.tmp, &mut s.bc, &mut s.bvx,
        &mut s.bvy, &mut s.crowd, &mut s.gfx, &mut s.gfy,
    ] {
        *v = vec![0.0; cells];
    }
}

#[no_mangle]
pub extern "C" fn set_gravity(g: f32) {
    sim().g = g;
}

#[no_mangle]
pub extern "C" fn clear_holes() {
    sim().holes.clear();
}

#[no_mangle]
pub extern "C" fn add_hole(x: f32, y: f32) {
    let s = sim();
    if s.holes.len() < MAX_HOLES {
        s.holes.push((x, y));
    }
}

/// Move every particle one time step (semi-implicit Euler).
#[no_mangle]
pub extern "C" fn step(dt: f32) {
    let s = sim();
    match s.mode {
        1 => step_fireworks(s, dt),
        2 => step_flock(s, dt),
        _ => step_galaxy(s, dt),
    }
}

fn step_galaxy(s: &mut Sim, dt: f32) {
    #[cfg(target_feature = "simd128")]
    {
        let done = unsafe { step_simd(s, dt) };
        step_scalar(s, dt, done);
    }
    #[cfg(not(target_feature = "simd128"))]
    step_scalar(s, dt, 0);
}

/// Fireworks: sparks fall, slow down in the air, and relaunch when
/// their burst's lifetime runs out.
fn step_fireworks(s: &mut Sim, dt: f32) {
    let n = s.x.len();
    #[cfg(target_feature = "simd128")]
    let done = unsafe { fireworks_simd(s, dt) };
    #[cfg(not(target_feature = "simd128"))]
    let done = 0;
    fireworks_scalar(s, dt, done, n);
    s.t += dt;
}

/// One spark at a time, for sparks from..to.
fn fireworks_scalar(s: &mut Sim, dt: f32, from: usize, to: usize) {
    let t = s.t;
    let ay0 = FW_GRAVITY * s.g;
    let drag = 1.0 - FW_DRAG * dt;
    for i in from..to {
        let mut a = s.age[i] + dt;
        let (mut px, mut py, mut pvx, mut pvy) = (s.x[i], s.y[i], s.vx[i], s.vy[i]);
        if a >= s.life[i] {
            // Relaunch somewhere a little different each time
            a -= s.life[i];
            let ox = s.ox[i];
            let oy = s.oy[i];
            px = ox + 0.45 * (1.3 * t + 7.0 * ox).sin();
            py = oy + 0.2 * (0.9 * t + 5.0 * oy).cos();
            pvx = s.svx[i];
            pvy = s.svy[i];
        }
        s.age[i] = a;
        let mut ax = 0.0;
        let mut ay = ay0;
        for &(hx, hy) in s.holes.iter() {
            let dx = hx - px;
            let dy = hy - py;
            let q2 = dx * dx + dy * dy + EPS2;
            let q = FW_HOLE / (q2 * q2.sqrt());
            ax += dx * q;
            ay += dy * q;
        }
        pvx = (pvx + ax * dt) * drag;
        pvy = (pvy + ay * dt) * drag;
        s.vx[i] = pvx;
        s.vy[i] = pvy;
        s.x[i] = px + pvx * dt;
        s.y[i] = py + pvy * dt;
    }
}

/// Four sparks at a time. Relaunches are rare, so any group of four
/// containing one is handed to the scalar version. Returns how many
/// sparks it covered.
#[cfg(target_feature = "simd128")]
unsafe fn fireworks_simd(s: &mut Sim, dt: f32) -> usize {
    use core::arch::wasm32::*;

    let n4 = s.x.len() / 4 * 4;
    let dtv = f32x4_splat(dt);
    let drag = f32x4_splat(1.0 - FW_DRAG * dt);
    let ay0 = f32x4_splat(FW_GRAVITY * s.g);
    let eps = f32x4_splat(EPS2);
    let hole = f32x4_splat(FW_HOLE);

    let mut i = 0;
    while i < n4 {
        let age = f32x4_add(v128_load(s.age.as_ptr().add(i) as *const v128), dtv);
        let life = v128_load(s.life.as_ptr().add(i) as *const v128);
        if v128_any_true(f32x4_ge(age, life)) {
            fireworks_scalar(s, dt, i, i + 4);
            i += 4;
            continue;
        }
        v128_store(s.age.as_mut_ptr().add(i) as *mut v128, age);

        let px = v128_load(s.x.as_ptr().add(i) as *const v128);
        let py = v128_load(s.y.as_ptr().add(i) as *const v128);
        let mut ax = f32x4_splat(0.0);
        let mut ay = ay0;
        for &(hx, hy) in s.holes.iter() {
            let dx = f32x4_sub(f32x4_splat(hx), px);
            let dy = f32x4_sub(f32x4_splat(hy), py);
            let q2 = f32x4_add(f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)), eps);
            let q = f32x4_div(hole, f32x4_mul(q2, f32x4_sqrt(q2)));
            ax = f32x4_add(ax, f32x4_mul(dx, q));
            ay = f32x4_add(ay, f32x4_mul(dy, q));
        }
        let vx = v128_load(s.vx.as_ptr().add(i) as *const v128);
        let vy = v128_load(s.vy.as_ptr().add(i) as *const v128);
        let vx = f32x4_mul(f32x4_add(vx, f32x4_mul(ax, dtv)), drag);
        let vy = f32x4_mul(f32x4_add(vy, f32x4_mul(ay, dtv)), drag);
        v128_store(s.vx.as_mut_ptr().add(i) as *mut v128, vx);
        v128_store(s.vy.as_mut_ptr().add(i) as *mut v128, vy);
        v128_store(s.x.as_mut_ptr().add(i) as *mut v128, f32x4_add(px, f32x4_mul(vx, dtv)));
        v128_store(s.y.as_mut_ptr().add(i) as *mut v128, f32x4_add(py, f32x4_mul(vy, dtv)));
        i += 4;
    }
    n4
}

/// Box blur over each cell and its 8 neighbours, wrapping at the edges.
fn blur(src: &[f32], dst: &mut [f32], tmp: &mut [f32], gw: usize, gh: usize) {
    for j in 0..gh {
        let row = j * gw;
        for i in 0..gw {
            tmp[row + i] = src[row + i] + src[row + (i + gw - 1) % gw] + src[row + (i + 1) % gw];
        }
    }
    for j in 0..gh {
        let row = j * gw;
        let up = ((j + gh - 1) % gh) * gw;
        let down = ((j + 1) % gh) * gw;
        for i in 0..gw {
            dst[row + i] = tmp[row + i] + tmp[up + i] + tmp[down + i];
        }
    }
}

/// Flock: each bird steers to match the smoothed heading of the birds
/// around it, drifts towards busier areas while avoiding crowding,
/// rides a gentle breeze, and flees hawks.
fn step_flock(s: &mut Sim, dt: f32) {
    let (gw, gh) = (s.gw, s.gh);
    let cells = gw * gh;
    let (wx, wy) = (s.wx, s.wy);
    let (lx, ly) = (2.0 * wx, 2.0 * wy);
    let sx = gw as f32 / lx;
    let sy = gh as f32 / ly;
    let n = s.x.len();

    s.cnt.fill(0.0);
    s.gvx.fill(0.0);
    s.gvy.fill(0.0);

    // Wrap at the screen edges, then sort birds into grid cells
    for i in 0..n {
        let mut px = s.x[i];
        let mut py = s.y[i];
        if px < -wx || px >= wx {
            px = (px + wx).rem_euclid(lx) - wx;
        }
        if py < -wy || py >= wy {
            py = (py + wy).rem_euclid(ly) - wy;
        }
        s.x[i] = px;
        s.y[i] = py;
        let gx = (((px + wx) * sx) as usize).min(gw - 1);
        let gy = (((py + wy) * sy) as usize).min(gh - 1);
        let k = gy * gw + gx;
        s.cell[i] = k as u32;
        s.cnt[k] += 1.0;
        s.gvx[k] += s.vx[i];
        s.gvy[k] += s.vy[i];
    }

    // Smoothed neighbourhood: average heading, and a crowding field
    blur(&s.cnt, &mut s.bc, &mut s.tmp, gw, gh);
    blur(&s.gvx, &mut s.bvx, &mut s.tmp, gw, gh);
    blur(&s.gvy, &mut s.bvy, &mut s.tmp, gw, gh);
    let mean = n as f32 / cells as f32 + 1.0;
    for k in 0..cells {
        let inv = 1.0 / s.bc[k].max(1.0);
        s.bvx[k] *= inv;
        s.bvy[k] *= inv;
        let d = s.bc[k] / 9.0 / mean;
        s.crowd[k] = d * (FL_COHERE - FL_SEPARATE * d);
    }
    for j in 0..gh {
        let row = j * gw;
        let up = ((j + gh - 1) % gh) * gw;
        let down = ((j + 1) % gh) * gw;
        for i in 0..gw {
            s.gfx[row + i] = s.crowd[row + (i + 1) % gw] - s.crowd[row + (i + gw - 1) % gw];
            s.gfy[row + i] = s.crowd[down + i] - s.crowd[up + i];
        }
    }

    #[cfg(target_feature = "simd128")]
    let done = unsafe { flock_birds_simd(s, dt) };
    #[cfg(not(target_feature = "simd128"))]
    let done = 0;
    flock_birds_scalar(s, dt, done);
    s.t += dt;
}

/// Steer each bird, one at a time, starting from bird `from`.
fn flock_birds_scalar(s: &mut Sim, dt: f32, from: usize) {
    let t = s.t;
    let target = FL_SPEED * s.g;
    for i in from..s.x.len() {
        let k = s.cell[i] as usize;
        let px = s.x[i];
        let py = s.y[i];
        let mut pvx = s.vx[i];
        let mut pvy = s.vy[i];
        let mut ax = (s.bvx[k] - pvx) * FL_ALIGN + s.gfx[k] + FL_WIND * fast_sin(2.3 * py + 0.6 * t);
        let mut ay = (s.bvy[k] - pvy) * FL_ALIGN + s.gfy[k] + FL_WIND * fast_cos(1.9 * px - 0.4 * t);
        // Hawks: birds flee from them
        for &(hx, hy) in s.holes.iter() {
            let dx = px - hx;
            let dy = py - hy;
            let q2 = dx * dx + dy * dy + 0.002;
            if q2 < 0.16 {
                let q = FL_FLEE / q2;
                ax += dx * q;
                ay += dy * q;
            }
        }
        pvx += ax * dt;
        pvy += ay * dt;
        // Ease back towards cruising speed
        let sp = (pvx * pvx + pvy * pvy).sqrt() + 1e-6;
        let f = 1.0 + 0.08 * (target / sp - 1.0);
        pvx *= f;
        pvy *= f;
        s.vx[i] = pvx;
        s.vy[i] = pvy;
        s.x[i] = px + pvx * dt;
        s.y[i] = py + pvy * dt;
    }
}

/// Steer four birds at a time. Each bird reads its own grid cell, so
/// those values are gathered one by one, then all the maths is SIMD.
/// Returns how many birds it covered.
#[cfg(target_feature = "simd128")]
unsafe fn flock_birds_simd(s: &mut Sim, dt: f32) -> usize {
    use core::arch::wasm32::*;

    let n4 = s.x.len() / 4 * 4;
    let t = s.t;
    let dtv = f32x4_splat(dt);
    let align = f32x4_splat(FL_ALIGN);
    let wind = f32x4_splat(FL_WIND);
    let k23 = f32x4_splat(2.3);
    let k19 = f32x4_splat(1.9);
    let t06 = f32x4_splat(0.6 * t);
    let t04 = f32x4_splat(0.4 * t);
    let half_pi = f32x4_splat(FRAC_PI_2);
    let flee = f32x4_splat(FL_FLEE);
    let near = f32x4_splat(0.16);
    let soft = f32x4_splat(0.002);
    let target = f32x4_splat(FL_SPEED * s.g);
    let one = f32x4_splat(1.0);
    let ease = f32x4_splat(0.08);
    let tiny = f32x4_splat(1e-6);

    let xp = s.x.as_mut_ptr();
    let yp = s.y.as_mut_ptr();
    let vxp = s.vx.as_mut_ptr();
    let vyp = s.vy.as_mut_ptr();
    let cell = s.cell.as_ptr();
    let bvx = s.bvx.as_ptr();
    let bvy = s.bvy.as_ptr();
    let gfx = s.gfx.as_ptr();
    let gfy = s.gfy.as_ptr();
    let holes = &s.holes;

    let mut i = 0;
    while i < n4 {
        let k0 = *cell.add(i) as usize;
        let k1 = *cell.add(i + 1) as usize;
        let k2 = *cell.add(i + 2) as usize;
        let k3 = *cell.add(i + 3) as usize;
        let cvx = f32x4(*bvx.add(k0), *bvx.add(k1), *bvx.add(k2), *bvx.add(k3));
        let cvy = f32x4(*bvy.add(k0), *bvy.add(k1), *bvy.add(k2), *bvy.add(k3));
        let fx = f32x4(*gfx.add(k0), *gfx.add(k1), *gfx.add(k2), *gfx.add(k3));
        let fy = f32x4(*gfy.add(k0), *gfy.add(k1), *gfy.add(k2), *gfy.add(k3));

        let px = v128_load(xp.add(i) as *const v128);
        let py = v128_load(yp.add(i) as *const v128);
        let vx = v128_load(vxp.add(i) as *const v128);
        let vy = v128_load(vyp.add(i) as *const v128);

        let breeze_x = sin4(f32x4_add(f32x4_mul(k23, py), t06));
        let breeze_y = sin4(f32x4_add(f32x4_sub(f32x4_mul(k19, px), t04), half_pi));
        let mut ax = f32x4_add(f32x4_add(f32x4_mul(f32x4_sub(cvx, vx), align), fx), f32x4_mul(wind, breeze_x));
        let mut ay = f32x4_add(f32x4_add(f32x4_mul(f32x4_sub(cvy, vy), align), fy), f32x4_mul(wind, breeze_y));

        // Hawks: birds flee from them (the mask zeroes far-away birds)
        for &(hx, hy) in holes.iter() {
            let dx = f32x4_sub(px, f32x4_splat(hx));
            let dy = f32x4_sub(py, f32x4_splat(hy));
            let q2 = f32x4_add(f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)), soft);
            let q = v128_and(f32x4_div(flee, q2), f32x4_lt(q2, near));
            ax = f32x4_add(ax, f32x4_mul(dx, q));
            ay = f32x4_add(ay, f32x4_mul(dy, q));
        }

        let vx = f32x4_add(vx, f32x4_mul(ax, dtv));
        let vy = f32x4_add(vy, f32x4_mul(ay, dtv));
        // Ease back towards cruising speed
        let sp = f32x4_add(f32x4_sqrt(f32x4_add(f32x4_mul(vx, vx), f32x4_mul(vy, vy))), tiny);
        let f = f32x4_add(one, f32x4_mul(ease, f32x4_sub(f32x4_div(target, sp), one)));
        let vx = f32x4_mul(vx, f);
        let vy = f32x4_mul(vy, f);
        v128_store(vxp.add(i) as *mut v128, vx);
        v128_store(vyp.add(i) as *mut v128, vy);
        v128_store(xp.add(i) as *mut v128, f32x4_add(px, f32x4_mul(vx, dtv)));
        v128_store(yp.add(i) as *mut v128, f32x4_add(py, f32x4_mul(vy, dtv)));
        i += 4;
    }
    n4
}

/// Plain version: one particle at a time, starting from index `from`.
fn step_scalar(s: &mut Sim, dt: f32, from: usize) {
    let g = s.g;
    let hole_g = HOLE_MASS * g;
    let holes = &s.holes;

    // Zipping the four arrays lets the compiler drop bounds checks
    for (((x, y), vx), vy) in s.x[from..]
        .iter_mut()
        .zip(s.y[from..].iter_mut())
        .zip(s.vx[from..].iter_mut())
        .zip(s.vy[from..].iter_mut())
    {
        let px = *x;
        let py = *y;
        let r2 = px * px + py * py + EPS2;
        let f = g / (r2 * r2.sqrt());
        let mut ax = -px * f;
        let mut ay = -py * f;

        for &(hx, hy) in holes.iter() {
            let dx = hx - px;
            let dy = hy - py;
            let r2 = dx * dx + dy * dy + EPS2;
            let f = hole_g / (r2 * r2.sqrt());
            ax += dx * f;
            ay += dy * f;
        }

        *vx += ax * dt;
        *vy += ay * dt;
        *x = px + *vx * dt;
        *y = py + *vy * dt;
    }
}

/// SIMD version: four particles per instruction using WebAssembly's
/// 128-bit vectors. Returns how many particles it handled; the scalar
/// version finishes off the last 0 to 3.
#[cfg(target_feature = "simd128")]
unsafe fn step_simd(s: &mut Sim, dt: f32) -> usize {
    use core::arch::wasm32::*;

    let n = s.x.len() / 4 * 4;
    let xp = s.x.as_mut_ptr();
    let yp = s.y.as_mut_ptr();
    let vxp = s.vx.as_mut_ptr();
    let vyp = s.vy.as_mut_ptr();
    let holes = &s.holes;

    let g = f32x4_splat(s.g);
    let hole_g = f32x4_splat(HOLE_MASS * s.g);
    let eps = f32x4_splat(EPS2);
    let dtv = f32x4_splat(dt);

    let mut i = 0;
    while i < n {
        let px = v128_load(xp.add(i) as *const v128);
        let py = v128_load(yp.add(i) as *const v128);

        let r2 = f32x4_add(f32x4_add(f32x4_mul(px, px), f32x4_mul(py, py)), eps);
        let f = f32x4_div(g, f32x4_mul(r2, f32x4_sqrt(r2)));
        let mut ax = f32x4_neg(f32x4_mul(px, f));
        let mut ay = f32x4_neg(f32x4_mul(py, f));

        for &(hx, hy) in holes.iter() {
            let dx = f32x4_sub(f32x4_splat(hx), px);
            let dy = f32x4_sub(f32x4_splat(hy), py);
            let q2 = f32x4_add(f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)), eps);
            let q = f32x4_div(hole_g, f32x4_mul(q2, f32x4_sqrt(q2)));
            ax = f32x4_add(ax, f32x4_mul(dx, q));
            ay = f32x4_add(ay, f32x4_mul(dy, q));
        }

        let vx = f32x4_add(v128_load(vxp.add(i) as *const v128), f32x4_mul(ax, dtv));
        let vy = f32x4_add(v128_load(vyp.add(i) as *const v128), f32x4_mul(ay, dtv));
        v128_store(vxp.add(i) as *mut v128, vx);
        v128_store(vyp.add(i) as *mut v128, vy);
        v128_store(xp.add(i) as *mut v128, f32x4_add(px, f32x4_mul(vx, dtv)));
        v128_store(yp.add(i) as *mut v128, f32x4_add(py, f32x4_mul(vy, dtv)));
        i += 4;
    }
    n
}

/// Rasterise: count particles per pixel, then colour by count.
/// Returns a pointer to w*h RGBA pixels, ready for ImageData.
#[no_mangle]
pub extern "C" fn draw() -> *const u32 {
    let s = sim();
    let (w, h) = (s.w, s.h);
    let wf = w as f32;
    let hf = h as f32;
    let scale = (w.min(h) as f32) * 0.42;
    let cx = wf * 0.5;
    let cy = hf * 0.5;

    s.dens.fill(0);
    if s.mode == 1 {
        // Fresh sparks shine brighter, then fade out completely
        for i in 0..s.x.len() {
            let fx = cx + s.x[i] * scale;
            let fy = cy + s.y[i] * scale;
            if fx >= 0.0 && fx < wf && fy >= 0.0 && fy < hf {
                let wgt = (4.0 * (1.0 - s.age[i] / s.life[i])).floor().min(3.0);
                s.dens[fy as usize * w + fx as usize] += wgt as u32;
            }
        }
    } else {
        for (&x, &y) in s.x.iter().zip(s.y.iter()) {
            let fx = cx + x * scale;
            let fy = cy + y * scale;
            if fx >= 0.0 && fx < wf && fy >= 0.0 && fy < hf {
                s.dens[fy as usize * w + fx as usize] += 1;
            }
        }
    }

    let lut = &s.lut;
    for (p, &d) in s.pixels.iter_mut().zip(s.dens.iter()) {
        *p = lut[d.min(255) as usize];
    }
    s.pixels.as_ptr()
}
