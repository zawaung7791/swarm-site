// One lane = one worker, so each language gets its own CPU core and
// its own frame loop. The page only sends settings and receives stats.
//
// CPU lanes step and return pixels for the worker to draw. The WebGPU
// lane draws itself, a whole frame at a time, and reports the graphics
// chip's own timings when the browser allows.

import { makeState, buildLut, DT } from './sims.js';

const FRAME_MS = 1000 / 60;

let backend = null, ctx = null, canvas = null;
let palette, sim = 'galaxy', w = 0, h = 0, count = 0, g = 1, holes = [];
let running = true, timer = null, inFrame = false;
let busy = Promise.resolve(); // the frame in progress, if any
let frames = 0, sumFrame = 0, sumPhysics = 0, sumDraw = 0, lastReport = 0;

function applySize() {
  if (!backend || !w || !h) return;
  canvas.width = w;
  canvas.height = h;
  backend.setSize(w, h, buildLut(palette));
}

function restart() {
  if (!backend) return;
  if (backend.initFrom) backend.initFrom(sim, count);
  else backend.init(makeState(sim, count));
  backend.setParams({ g, holes });
  resetStats();
}

function resetStats() {
  frames = sumFrame = sumPhysics = sumDraw = 0;
  lastReport = performance.now();
}

// Wait for the current frame, and for the GPU to finish its work,
// before anything replaces the buffers it's using
async function settle() {
  await busy;
  if (backend && backend.idle) await backend.idle();
}

function reportError(err) {
  postMessage({ type: 'error', message: String((err && err.message) || err) });
}

function frame() {
  timer = null;
  if (!running || !backend || inFrame) return;
  inFrame = true;
  busy = runFrame().catch(err => { running = false; reportError(err); });
}

async function runFrame() {
  const t0 = performance.now();
  let physics, draw, t2;
  if (backend.drawsItself) {
    const gpu = await backend.frame(DT[sim]);
    t2 = performance.now();
    // Without GPU timings there's no honest per-frame figure (the lane
    // doesn't wait for the frame it just sent), so the page shows '–'
    // and the benchmark relies on the measured frame rate instead
    physics = gpu ? gpu.physics : NaN;
    draw = gpu ? gpu.draw : NaN;
  } else {
    backend.step(DT[sim]);
    const t1 = performance.now();
    const px = backend.draw();
    ctx.putImageData(new ImageData(px, w, h), 0, 0);
    t2 = performance.now();
    physics = t1 - t0;
    draw = t2 - t1;
  }

  frames++;
  sumFrame += physics + draw;
  sumPhysics += physics;
  sumDraw += draw;
  if (t2 - lastReport >= 250) {
    postMessage({
      type: 'stats',
      fps: (frames * 1000) / (t2 - lastReport),
      frame: sumFrame / frames,
      physics: sumPhysics / frames,
      draw: sumDraw / frames,
      count,
    });
    resetStats();
  }
  inFrame = false;
  if (running && timer === null) timer = setTimeout(frame, Math.max(0, FRAME_MS - (t2 - t0)));
}

function start() {
  if (timer === null && running && backend && !inFrame) {
    resetStats();
    timer = setTimeout(frame, 0);
  }
}

self.onmessage = async ({ data: m }) => {
  switch (m.type) {
    case 'start': {
      canvas = m.canvas;
      palette = m.palette; sim = m.sim; count = m.count; g = m.gravity; holes = m.holes;
      w = m.width; h = m.height;
      try {
        const mod = await import(`./backends/${m.backend}.js`);
        backend = await mod.create({ canvas });
        if (!backend.drawsItself) ctx = canvas.getContext('2d', { alpha: false });
        applySize();
        restart();
      } catch (err) {
        backend = null;
        reportError(err);
        return;
      }
      postMessage({ type: 'ready' });
      start();
      break;
    }
    // Anything that rebuilds the simulation waits for the current frame
    // to finish first, so the GPU never works on buffers being replaced.
    case 'resize':
      w = m.width; h = m.height;
      await settle();
      applySize();
      break;
    case 'count':
      count = m.count;
      await settle();
      restart();
      break;
    case 'sim':
      sim = m.sim;
      holes = m.holes;
      await settle();
      restart();
      break;
    case 'reset':
      holes = [];
      await settle();
      restart();
      break;
    case 'gravity':
      g = m.value;
      backend && backend.setParams({ g, holes });
      break;
    case 'holes':
      holes = m.holes;
      backend && backend.setParams({ g, holes });
      break;
    case 'pause':
      running = false;
      clearTimeout(timer);
      timer = null;
      break;
    case 'resume':
      running = true;
      start();
      break;
  }
};
