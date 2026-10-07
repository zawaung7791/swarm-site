// The WebGPU lane: the same three simulations as compute shaders on the
// graphics chip. Drawing happens on the GPU too: each particle adds
// itself to its pixel's count, and a final pass colours the pixels
// straight onto the canvas, so the image never goes back to the CPU.
//
// Each frame goes to the GPU as one batch. The lane waits for the
// previous frame rather than this one, so the round trip to the GPU
// overlaps with work instead of adding to it. Where the browser allows
// it, the GPU timestamps its own passes, so the Physics and Drawing
// figures show the GPU's real working time.

import { generate, MODES } from '../sims.js';

const WG = 256;              // threads per workgroup
const CHUNK = 1 << 20;       // particles uploaded per chunk
const PARAMS_SIZE = 176;     // bytes in the Params struct below

// Shared by every shader. Must match writeParams().
const COMMON = /* wgsl */ `
struct Params {
  dt: f32, g: f32, t: f32, n: u32,
  w: u32, h: u32, scale: f32, nholes: u32,
  wx: f32, wy: f32, gw: u32, gh: u32,
  holes: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> P: Params;
`;

// Counting pixels is merged into each simulation's final pass: a particle
// adds itself to its pixel straight after moving, while its position is
// still at hand, instead of a separate pass reading every position back
// from memory.
const plot = binding => /* wgsl */ `
@group(0) @binding(${binding}) var<storage, read_write> dens: array<atomic<u32>>;

fn plot(p: vec2f, wgt: u32) {
  let fx = f32(P.w) * 0.5 + p.x * P.scale;
  let fy = f32(P.h) * 0.5 + p.y * P.scale;
  if (fx >= 0.0 && fx < f32(P.w) && fy >= 0.0 && fy < f32(P.h)) {
    // A pixel's colour stops changing at 255, so skip the queue for
    // pixels that are already full (the galaxy's core gets very busy)
    let idx = u32(fy) * P.w + u32(fx);
    if (atomicLoad(&dens[idx]) < 255u) {
      atomicAdd(&dens[idx], wgt);
    }
  }
}
`;

const GALAXY = COMMON + plot(3) + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
const EPS2 = 0.0025;
const HOLE_MASS = 0.6;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.n) { return; }
  var p = pos[i];
  var v = vel[i];
  let r2 = dot(p, p) + EPS2;
  var a = -p * (P.g / (r2 * sqrt(r2)));
  for (var k = 0u; k < P.nholes; k++) {
    let d = P.holes[k].xy - p;
    let q2 = dot(d, d) + EPS2;
    a += d * (HOLE_MASS * P.g / (q2 * sqrt(q2)));
  }
  v += a * P.dt;
  p += v * P.dt;
  pos[i] = p;
  vel[i] = v;
  plot(p, 1u);
}
`;

const FIREWORKS = COMMON + plot(6) + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read> origin: array<vec2f>;
@group(0) @binding(4) var<storage, read> svel: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> agelife: array<vec2f>;
const EPS2 = 0.0025;
const FW_GRAVITY = 0.55;
const FW_DRAG = 1.5;
const FW_HOLE = 0.15;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.n) { return; }
  var al = agelife[i];
  var p = pos[i];
  var v = vel[i];
  al.x += P.dt;
  if (al.x >= al.y) {
    // Relaunch somewhere a little different each time
    al.x -= al.y;
    let o = origin[i];
    p = vec2f(o.x + 0.45 * sin(1.3 * P.t + 7.0 * o.x), o.y + 0.2 * cos(0.9 * P.t + 5.0 * o.y));
    v = svel[i];
  }
  agelife[i] = al;
  var a = vec2f(0.0, FW_GRAVITY * P.g);
  for (var k = 0u; k < P.nholes; k++) {
    let d = P.holes[k].xy - p;
    let q2 = dot(d, d) + EPS2;
    a += d * (FW_HOLE / (q2 * sqrt(q2)));
  }
  let drag = 1.0 - FW_DRAG * P.dt;
  v = (v + a * P.dt) * drag;
  p += v * P.dt;
  pos[i] = p;
  vel[i] = v;
  // Fresh sparks shine brighter, then fade out completely
  plot(p, u32(min(3.0, floor(4.0 * (1.0 - al.x / al.y)))));
}
`;

// ---- Flock: four passes ----
// grid holds three blocks of cells: bird counts, then summed x velocity
// and summed y velocity in fixed point (GPU atomics only add integers).

const FLOCK_BIN = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> cellOf: array<u32>;
@group(0) @binding(4) var<storage, read_write> grid: array<atomic<i32>>;
const FIX = 1024.0;

fn wrap(v: f32, edge: f32) -> f32 {
  if (v < -edge || v >= edge) {
    let span = 2.0 * edge;
    let a = v + edge;
    return a - span * floor(a / span) - edge;
  }
  return v;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.n) { return; }
  let p = vec2f(wrap(pos[i].x, P.wx), wrap(pos[i].y, P.wy));
  pos[i] = p;
  let gx = min(u32((p.x + P.wx) * f32(P.gw) / (2.0 * P.wx)), P.gw - 1u);
  let gy = min(u32((p.y + P.wy) * f32(P.gh) / (2.0 * P.wy)), P.gh - 1u);
  let k = gy * P.gw + gx;
  cellOf[i] = k;
  let cells = P.gw * P.gh;
  let v = vel[i];
  atomicAdd(&grid[k], 1);
  atomicAdd(&grid[cells + k], i32(round(v.x * FIX)));
  atomicAdd(&grid[2u * cells + k], i32(round(v.y * FIX)));
}
`;

const FLOCK_CELLS = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> field: array<vec4f>;
const FIX = 1024.0;
const COHERE = 0.3;
const SEPARATE = 0.06;

// Smoothed over each cell and its 8 neighbours: average heading in xy,
// crowding in z
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let cells = P.gw * P.gh;
  let k = id.x;
  if (k >= cells) { return; }
  let gx = k % P.gw;
  let gy = k / P.gw;
  var c = 0;
  var sx = 0;
  var sy = 0;
  for (var dy = 0u; dy < 3u; dy++) {
    let yy = (gy + P.gh + dy - 1u) % P.gh;
    for (var dx = 0u; dx < 3u; dx++) {
      let xx = (gx + P.gw + dx - 1u) % P.gw;
      let j = yy * P.gw + xx;
      c += atomicLoad(&grid[j]);
      sx += atomicLoad(&grid[cells + j]);
      sy += atomicLoad(&grid[2u * cells + j]);
    }
  }
  let bc = f32(c);
  let inv = 1.0 / max(bc, 1.0);
  let mean = f32(P.n) / f32(cells) + 1.0;
  let d = bc / 9.0 / mean;
  field[k] = vec4f(f32(sx) / FIX * inv, f32(sy) / FIX * inv, d * (COHERE - SEPARATE * d), 0.0);
}
`;

const FLOCK_FORCE = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> field: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> force: array<vec2f>;

// Which way crowding rises, from the neighbouring cells
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let cells = P.gw * P.gh;
  let k = id.x;
  if (k >= cells) { return; }
  let gx = k % P.gw;
  let gy = k / P.gw;
  let right = gy * P.gw + (gx + 1u) % P.gw;
  let left = gy * P.gw + (gx + P.gw - 1u) % P.gw;
  let down = ((gy + 1u) % P.gh) * P.gw + gx;
  let up = ((gy + P.gh - 1u) % P.gh) * P.gw + gx;
  force[k] = vec2f(field[right].z - field[left].z, field[down].z - field[up].z);
}
`;

const FLOCK_BIRDS = COMMON + plot(6) + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read> cellOf: array<u32>;
@group(0) @binding(4) var<storage, read> field: array<vec4f>;
@group(0) @binding(5) var<storage, read> force: array<vec2f>;
const SPEED = 0.45;
const ALIGN = 2.0;
const WIND = 0.25;
const FLEE = 0.05;
const TAU = 6.2831853;
const INV_TAU = 0.15915494;
const HALF_PI = 1.5707963;

// The same fast sine the other lanes use
fn fast_sin(x0: f32) -> f32 {
  let x = x0 - TAU * round(x0 * INV_TAU);
  let y = 1.2732395 * x - 0.40528473 * x * abs(x);
  return 0.225 * (y * abs(y) - y) + y;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.n) { return; }
  let k = cellOf[i];
  let f = field[k];
  var p = pos[i];
  var v = vel[i];
  let breeze = vec2f(fast_sin(2.3 * p.y + 0.6 * P.t), fast_sin(1.9 * p.x - 0.4 * P.t + HALF_PI));
  var a = (f.xy - v) * ALIGN + force[k] + WIND * breeze;
  // Hawks: birds flee from them
  for (var m = 0u; m < P.nholes; m++) {
    let d = p - P.holes[m].xy;
    let q2 = dot(d, d) + 0.002;
    if (q2 < 0.16) {
      a += d * (FLEE / q2);
    }
  }
  v += a * P.dt;
  // Ease back towards cruising speed
  let sp = length(v) + 1e-6;
  v *= 1.0 + 0.08 * (SPEED * P.g / sp - 1.0);
  p += v * P.dt;
  pos[i] = p;
  vel[i] = v;
  plot(p, 1u);
}
`;

// ---- Drawing: colour each pixel from its count ----

const PRESENT = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> dens: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let x = min(u32(frag.x), P.w - 1u);
  let y = min(u32(frag.y), P.h - 1u);
  return unpack4x8unorm(lut[min(dens[y * P.w + x], 255u)]);
}
`;

export async function create({ canvas }) {
  if (!navigator.gpu) {
    throw new Error("WebGPU isn't available in this browser. Try a recent version of Chrome or Edge.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU graphics adapter was found.');
  const canTime = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: canTime ? ['timestamp-query'] : [] });
  device.lost.then(info => {
    if (info.reason !== 'destroyed') {
      self.postMessage({ type: 'error', message: `the graphics device was lost (${info.message})` });
    }
  });
  // Show the first GPU error in the lane, rather than only in the console
  let reported = false;
  device.addEventListener('uncapturederror', e => {
    if (reported) return;
    reported = true;
    self.postMessage({ type: 'error', message: `WebGPU error: ${e.error.message}` });
  });

  // Compile a shader, turning any errors into a readable message
  async function shader(code, label) {
    const module = device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(msg => msg.type === 'error');
    if (errors.length) {
      throw new Error(`the ${label} shader failed to compile: ` +
        errors.map(e => `line ${e.lineNum}: ${e.message}`).join('; '));
    }
    return module;
  }

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const compute = async (code, label) => device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module: await shader(code, label), entryPoint: 'main' },
  });
  const presentModule = await shader(PRESENT, 'present');
  const pipes = {
    galaxy: await compute(GALAXY, 'galaxy'),
    fireworks: await compute(FIREWORKS, 'fireworks'),
    bin: await compute(FLOCK_BIN, 'flock grid'),
    cells: await compute(FLOCK_CELLS, 'flock cells'),
    force: await compute(FLOCK_FORCE, 'flock force'),
    birds: await compute(FLOCK_BIRDS, 'flock birds'),
    present: await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vs' },
      fragment: { module: presentModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    }),
  };

  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const makeBuffer = bytes => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: STORAGE });
  const params = device.createBuffer({ size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lut = makeBuffer(256 * 4);
  const pdata = new ArrayBuffer(PARAMS_SIZE);
  const pf = new Float32Array(pdata);
  const pu = new Uint32Array(pdata);

  // Timestamps: 0-1 around the physics (which now includes counting each
  // particle into its pixel), 2-3 around colouring the pixels.
  // Results are read back a frame or two later, through a small ring of
  // buffers, so reading them never makes the lane wait.
  let querySet = null, resolveBuf = null;
  const readbacks = [];
  let timings = null; // the latest { physics, draw } in milliseconds
  if (canTime) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
    resolveBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < 3; i++) {
      readbacks.push({ free: true, buf: device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) });
    }
  }
  let pending = null; // the previous frame's completion

  let n = 0, mode = 0, T = 0, g = 1, holes = [];
  let w = 0, h = 0, scale = 1, wx = 1, wy = 1, gw = 1, gh = 1;
  let parts = {};  // particle buffers
  let sized = {};  // buffers that depend on the canvas size
  let groups = null;

  const destroyAll = set => { for (const b of Object.values(set)) b.destroy(); };

  function writeParams(dt) {
    pf[0] = dt; pf[1] = g; pf[2] = T; pu[3] = n;
    pu[4] = w; pu[5] = h; pf[6] = scale; pu[7] = Math.min(8, holes.length);
    pf[8] = wx; pf[9] = wy; pu[10] = gw; pu[11] = gh;
    for (let k = 0; k < 8; k++) {
      pf[12 + k * 4] = holes[k] ? holes[k][0] : 0;
      pf[13 + k * 4] = holes[k] ? holes[k][1] : 0;
    }
    device.queue.writeBuffer(params, 0, pdata);
  }

  function bindGroup(pipe, buffers) {
    return device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [params, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  function rebuildGroups() {
    groups = null;
    if (!parts.pos || !sized.dens) return;
    const { pos, vel } = parts;
    const dens = sized.dens;
    groups = { present: bindGroup(pipes.present, [dens, lut]) };
    if (mode === 0) {
      groups.sim = [[pipes.galaxy, bindGroup(pipes.galaxy, [pos, vel, dens]), n]];
    } else if (mode === 1) {
      groups.sim = [[pipes.fireworks, bindGroup(pipes.fireworks, [pos, vel, parts.origin, parts.svel, parts.agelife, dens]), n]];
    } else {
      const cells = gw * gh;
      groups.sim = [
        [pipes.bin, bindGroup(pipes.bin, [pos, vel, parts.cellOf, sized.grid]), n],
        [pipes.cells, bindGroup(pipes.cells, [sized.grid, sized.field]), cells],
        [pipes.force, bindGroup(pipes.force, [sized.field, sized.force]), cells],
        [pipes.birds, bindGroup(pipes.birds, [pos, vel, parts.cellOf, sized.field, sized.force, dens]), n],
      ];
    }
  }

  // Interleave two arrays into x0, y0, x1, y1, ... for a vec2 buffer
  const pairs = (a, b) => {
    const out = new Float32Array(a.length * 2);
    for (let i = 0; i < a.length; i++) { out[2 * i] = a[i]; out[2 * i + 1] = b[i]; }
    return out;
  };

  return {
    drawsItself: true,

    // Builds the particles straight into GPU memory, a chunk at a time
    initFrom(sim, count) {
      destroyAll(parts);
      n = count; mode = MODES[sim]; T = 0;
      parts = { pos: makeBuffer(n * 8), vel: makeBuffer(n * 8) };
      if (mode === 1) {
        parts.origin = makeBuffer(n * 8);
        parts.svel = makeBuffer(n * 8);
        parts.agelife = makeBuffer(n * 8);
      }
      if (mode === 2) parts.cellOf = makeBuffer(n * 4);
      const q = device.queue;
      generate(sim, n, CHUNK, (start, c) => {
        const at = start * 8;
        q.writeBuffer(parts.pos, at, pairs(c.x, c.y));
        q.writeBuffer(parts.vel, at, pairs(c.vx, c.vy));
        if (mode === 1) {
          q.writeBuffer(parts.origin, at, pairs(c.ox, c.oy));
          q.writeBuffer(parts.svel, at, pairs(c.svx, c.svy));
          q.writeBuffer(parts.agelife, at, pairs(c.age, c.life));
        }
      });
      rebuildGroups();
    },

    setSize(W, H, lutData) {
      destroyAll(sized);
      w = W; h = H;
      scale = Math.min(w, h) * 0.42;
      wx = (w * 0.5) / scale; wy = (h * 0.5) / scale;
      gw = Math.max(1, Math.floor((2 * wx) / 0.1));
      gh = Math.max(1, Math.floor((2 * wy) / 0.1));
      const cells = gw * gh;
      sized = {
        dens: makeBuffer(w * h * 4),
        grid: makeBuffer(cells * 3 * 4),
        field: makeBuffer(cells * 16),
        force: makeBuffer(cells * 8),
      };
      device.queue.writeBuffer(lut, 0, lutData);
      rebuildGroups();
    },

    setParams(p) { g = p.g; holes = p.holes; },

    // Wait until the GPU has finished everything already sent
    async idle() {
      if (pending) await pending;
      pending = null;
    },

    // One whole frame: physics, counting and colouring, sent as a batch.
    // Returns the GPU's own timings when available, otherwise null.
    async frame(dt) {
      if (!groups) return null;
      writeParams(dt);
      const enc = device.createCommandEncoder();
      const slot = canTime ? readbacks.find(r => r.free) : null;
      const stamp = (begin, end) => slot
        ? { timestampWrites: { querySet, ...(begin !== null && { beginningOfPassWriteIndex: begin }), ...(end !== null && { endOfPassWriteIndex: end }) } }
        : {};

      if (mode === 2) enc.clearBuffer(sized.grid);
      enc.clearBuffer(sized.dens);
      const sim = enc.beginComputePass(stamp(0, 1));
      for (const [pipe, group, count] of groups.sim) {
        sim.setPipeline(pipe);
        sim.setBindGroup(0, group);
        sim.dispatchWorkgroups(Math.ceil(count / WG));
      }
      sim.end();

      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
        ...stamp(2, 3),
      });
      rp.setPipeline(pipes.present);
      rp.setBindGroup(0, groups.present);
      rp.draw(3);
      rp.end();

      if (slot) {
        enc.resolveQuerySet(querySet, 0, 4, resolveBuf, 0);
        enc.copyBufferToBuffer(resolveBuf, 0, slot.buf, 0, 32);
      }
      device.queue.submit([enc.finish()]);
      if (slot) {
        slot.free = false;
        slot.buf.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigUint64Array(slot.buf.getMappedRange());
          if (t[1] >= t[0] && t[3] >= t[2] && t[0] > 0n) {
            timings = { physics: Number(t[1] - t[0]) / 1e6, draw: Number(t[3] - t[2]) / 1e6 };
          }
          slot.buf.unmap();
          slot.free = true;
        }).catch(() => { slot.free = true; });
      }

      // Wait for the previous frame, not this one
      const done = device.queue.onSubmittedWorkDone();
      if (pending) await pending;
      pending = done;
      T += dt;
      return timings;
    },
  };
}
