// One lane = one worker, so each language gets its own CPU core and
// its own frame loop. The page only sends settings and receives stats.

import { makeState, buildLut, DT } from './sims.js';

const FRAME_MS = 1000 / 60;

let backend = null, ctx = null, canvas = null;
let palette, sim = 'galaxy', w = 0, h = 0, count = 0, g = 1, holes = [];
let running = true, timer = null;
let frames = 0, sumFrame = 0, sumPhysics = 0, sumDraw = 0, lastReport = 0;

function applySize() {
  if (!backend || !w || !h) return;
  canvas.width = w;
  canvas.height = h;
  backend.setSize(w, h, buildLut(palette));
}

function restart() {
  if (!backend) return;
  backend.init(makeState(sim, count));
  backend.setParams({ g, holes });
  resetStats();
}

function resetStats() {
  frames = sumFrame = sumPhysics = sumDraw = 0;
  lastReport = performance.now();
}

function frame() {
  timer = null;
  if (!running || !backend) return;
  const t0 = performance.now();
  backend.step(DT[sim]);
  const t1 = performance.now();
  const px = backend.draw();
  ctx.putImageData(new ImageData(px, w, h), 0, 0);
  const t2 = performance.now();

  frames++;
  sumFrame += t2 - t0;
  sumPhysics += t1 - t0;
  sumDraw += t2 - t1;
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
  timer = setTimeout(frame, Math.max(0, FRAME_MS - (t2 - t0)));
}

function start() {
  if (timer === null && running && backend) {
    resetStats();
    timer = setTimeout(frame, 0);
  }
}

self.onmessage = async ({ data: m }) => {
  switch (m.type) {
    case 'start': {
      canvas = m.canvas;
      ctx = canvas.getContext('2d', { alpha: false });
      palette = m.palette; sim = m.sim; count = m.count; g = m.gravity; holes = m.holes;
      w = m.width; h = m.height;
      try {
        const mod = await import(`./backends/${m.backend}.js`);
        backend = await mod.create();
        applySize();
        restart();
      } catch (err) {
        postMessage({ type: 'error', message: String(err && err.message || err) });
        return;
      }
      postMessage({ type: 'ready' });
      start();
      break;
    }
    case 'resize':
      w = m.width; h = m.height;
      applySize();
      break;
    case 'count':
      count = m.count;
      restart();
      break;
    case 'sim':
      sim = m.sim;
      holes = m.holes;
      restart();
      break;
    case 'reset':
      holes = [];
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
