// Swarm: page logic. Each lane runs in its own Web Worker and draws to
// its own canvas, so the page itself only handles controls and stats.

const LANES = {
  rust: {
    name: 'Rust', detail: 'Compiled to WebAssembly, with SIMD', accent: '#FF9A5C',
    palette: ['#B4482C', '#FF8A4C', '#FFC98F', '#FFF3E0'],
  },
  numpy: {
    name: 'Python with NumPy', detail: 'Pyodide, whole-array maths in compiled C', accent: '#7FB0FF',
    palette: ['#33558F', '#6B98E8', '#AECBFF', '#EEF4FF'],
    loading: 'Loading Python and NumPy. About 10 MB, only slow the first time.',
  },
  js: {
    name: 'JavaScript', detail: 'Typed arrays, same algorithm', accent: '#E8D44D',
    palette: ['#7D7025', '#D2BC3A', '#F3E48A', '#FFFBE6'],
  },
};

const SIMS = {
  galaxy: {
    gravity: 'Gravity', what: 'particles, each pulled by gravity every frame',
    click: 'Click anywhere to drop a black hole. Reset clears them.',
  },
  fireworks: {
    gravity: 'Gravity', what: 'sparks, each with its own gravity, drag and lifetime',
    click: 'Click anywhere to drop a black hole that swallows sparks. Reset clears them.',
  },
  flock: {
    gravity: 'Flock speed', what: 'birds, each steering by the birds around it',
    click: 'Click anywhere to send in a hawk. Reset clears them.',
  },
};

const VIEWS = {
  side: {
    lanes: ['rust', 'js'], count: 300000,
    tagline: 'The same simulation, written twice. Left: Rust compiled to WebAssembly. Right: JavaScript. Both run in your browser, on your machine.',
  },
  race: {
    lanes: ['rust', 'numpy', 'js'], count: 300000,
    tagline: 'The same simulation, written three ways and raced in your browser: Rust, Python with NumPy, and JavaScript.',
  },
  solo: { lanes: ['rust'], count: 1000000, tagline: '' },
};

const MIN_COUNT = 1000;
const MAX_COUNT = 3000000;
const BUDGET_MS = 1000 / 60;
const GRAPH_SAMPLES = 40; // stats arrive 4 times a second, so 10 seconds
const MAX_HOLES = 8;

const $ = id => document.getElementById(id);
const fmt = n => Math.round(n).toLocaleString('en-GB');

const state = {
  view: 'side',
  sim: 'galaxy',
  count: 300000,
  gravity: 1,
  paused: false,
  holes: [], // world coordinates, shared by every lane
  lanes: [],
  benching: false,
};

// ---------- slider scales ----------
function niceCount(n) {
  const p = Math.pow(10, Math.floor(Math.log10(n)) - 1);
  return Math.round(n / p) * p;
}
const sliderToCount = v => niceCount(MIN_COUNT * Math.pow(MAX_COUNT / MIN_COUNT, v / 1000));
const countToSlider = n => Math.round((Math.log(n / MIN_COUNT) / Math.log(MAX_COUNT / MIN_COUNT)) * 1000);
const sliderToGravity = v => 0.2 + (v / 1000) * 2.8;
const gravityToSlider = g => Math.round(((g - 0.2) / 2.8) * 1000);

function drawTicks() {
  const ticks = [1000, 10000, 100000, 1000000];
  $('ticks').innerHTML = ticks
    .map(t => `<span style="left:${(countToSlider(t) / 10).toFixed(1)}%">${fmt(t)}</span>`)
    .join('');
}

// ---------- lanes ----------
function stageSize(lane) {
  const r = lane.stage.getBoundingClientRect();
  return { width: Math.max(16, Math.round(r.width)), height: Math.max(16, Math.round(r.height)) };
}

function buildLane(key) {
  const spec = LANES[key];
  const el = $('laneTpl').content.firstElementChild.cloneNode(true);
  el.style.setProperty('--accent', spec.accent);
  el.querySelector('h2').textContent = spec.name;
  el.querySelector('.detail').textContent = spec.detail;

  const lane = {
    key, spec, el,
    stage: el.querySelector('.stage'),
    overlay: el.querySelector('.overlay'),
    fpsEl: el.querySelector('.fps-num'),
    chip: el.querySelector('.chip'),
    frameEl: el.querySelector('.m-frame'),
    physicsEl: el.querySelector('.m-physics'),
    drawEl: el.querySelector('.m-draw'),
    graph: el.querySelector('.graph'),
    history: [],
    latest: null,
    samples: null,
    ready: false,
  };
  if (spec.loading) lane.overlay.textContent = spec.loading;
  $('lanes').appendChild(el);

  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', `${spec.name} simulation`);
  lane.stage.prepend(canvas);
  lane.canvas = canvas;

  lane.worker = new Worker(new URL('./lane-worker.js', import.meta.url), { type: 'module' });
  lane.worker.onmessage = e => onWorkerMessage(lane, e.data);
  lane.worker.onerror = e => showLaneError(lane, e.message || 'The simulation failed to start.');

  const offscreen = canvas.transferControlToOffscreen();
  const size = stageSize(lane);
  lane.worker.postMessage({
    type: 'start', backend: key, sim: state.sim, canvas: offscreen, palette: spec.palette,
    count: state.count, gravity: state.gravity, holes: state.holes,
    width: size.width, height: size.height,
  }, [offscreen]);

  lane.stage.addEventListener('click', e => addHole(lane, e));

  let resizeTimer;
  lane.observer = new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      lane.worker.postMessage({ type: 'resize', ...stageSize(lane) });
      placeHoles(lane);
    }, 120);
  });
  lane.observer.observe(lane.stage);

  if (state.paused) lane.worker.postMessage({ type: 'pause' });
  placeHoles(lane);
  return lane;
}

function showLaneError(lane, message) {
  lane.overlay.hidden = false;
  lane.overlay.textContent = `${lane.spec.name} couldn't start: ${message}`;
}

function onWorkerMessage(lane, m) {
  if (m.type === 'ready') {
    lane.ready = true;
    lane.overlay.hidden = true;
  } else if (m.type === 'error') {
    showLaneError(lane, m.message);
  } else if (m.type === 'stats') {
    if (m.count !== state.count && !state.benching) return; // stale, from before a change
    lane.latest = m;
    if (lane.samples) lane.samples.push(m.frame);
    lane.history.push(m.frame);
    if (lane.history.length > GRAPH_SAMPLES) lane.history.shift();
    renderStats(lane);
  }
}

const ms = v => (v >= 100 ? fmt(v) : v.toFixed(1)) + ' ms';

function renderStats(lane) {
  const s = lane.latest;
  if (!s) return;
  const fps = Math.min(60, s.fps); // timer jitter can read a touch over 60
  lane.fpsEl.textContent = fps < 10 ? fps.toFixed(1) : Math.round(fps);
  lane.fpsEl.classList.toggle('slow', s.fps < 55);
  lane.frameEl.textContent = ms(s.frame);
  lane.physicsEl.textContent = ms(s.physics);
  lane.drawEl.textContent = ms(s.draw);
  drawGraph(lane);
  updateChips();

  if (state.view === 'solo') {
    $('soloCount').textContent = fmt(s.count);
    $('soloFps').textContent = Math.round(Math.min(60, s.fps));
    $('soloMs').textContent = ms(s.frame);
  }
}

// Compare every lane's compute time with Rust's
function updateChips() {
  const rust = state.lanes.find(l => l.key === 'rust');
  for (const lane of state.lanes) {
    if (!lane.latest || !rust || !rust.latest) { lane.chip.textContent = ''; continue; }
    if (lane === rust) { lane.chip.textContent = state.lanes.length > 1 ? 'Baseline' : ''; continue; }
    const ratio = lane.latest.frame / rust.latest.frame;
    const r = ratio >= 10 ? Math.round(ratio) : ratio.toFixed(1);
    if (ratio >= 1.05) lane.chip.textContent = `About ${r}× slower than Rust`;
    else if (ratio <= 0.95) lane.chip.textContent = `About ${(1 / ratio).toFixed(1)}× faster than Rust`;
    else lane.chip.textContent = 'Level with Rust';
  }
}

function drawGraph(lane) {
  const c = lane.graph;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const top = 50; // ms at the top of the chart
  const y = v => h - (Math.min(v, top) / top) * (h - 4) - 2;

  g.strokeStyle = '#A7B2C4';
  g.setLineDash([5, 5]);
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, y(BUDGET_MS)); g.lineTo(w, y(BUDGET_MS));
  g.stroke();
  g.setLineDash([]);

  const pts = lane.history;
  if (pts.length < 2) return;
  g.strokeStyle = lane.spec.accent;
  g.lineWidth = 2;
  g.beginPath();
  pts.forEach((v, i) => {
    const x = (i / (GRAPH_SAMPLES - 1)) * w;
    i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v));
  });
  g.stroke();
}

// ---------- black holes ----------
function toWorld(lane, e) {
  const r = lane.stage.getBoundingClientRect();
  const scale = Math.min(r.width, r.height) * 0.42;
  return [(e.clientX - r.left - r.width / 2) / scale, (e.clientY - r.top - r.height / 2) / scale];
}

function addHole(lane, e) {
  if (state.benching) return;
  state.holes.push(toWorld(lane, e));
  if (state.holes.length > MAX_HOLES) state.holes.shift();
  sendAll({ type: 'holes', holes: state.holes });
  state.lanes.forEach(placeHoles);
}

function placeHoles(lane) {
  lane.stage.querySelectorAll('.hole').forEach(n => n.remove());
  const r = lane.stage.getBoundingClientRect();
  const scale = Math.min(r.width, r.height) * 0.42;
  for (const [x, y] of state.holes) {
    const dot = document.createElement('span');
    dot.className = state.sim === 'flock' ? 'hole hawk' : 'hole';
    dot.style.left = `${r.width / 2 + x * scale}px`;
    dot.style.top = `${r.height / 2 + y * scale}px`;
    lane.stage.appendChild(dot);
  }
}

// ---------- views ----------
function sendAll(msg) { state.lanes.forEach(l => l.worker.postMessage(msg)); }

function setView(view) {
  if (!VIEWS[view] || state.benching) return;
  state.lanes.forEach(l => { l.observer.disconnect(); l.worker.terminate(); });
  state.lanes = [];
  $('lanes').innerHTML = '';

  state.view = view;
  const v = VIEWS[view];
  state.count = v.count;
  document.body.classList.toggle('solo', view === 'solo');
  $('tagline').textContent = v.tagline;
  $('lanes').dataset.count = v.lanes.length;
  $('countLabel').textContent = view === 'solo' ? 'Particles' : 'Particles, in every lane';
  document.querySelectorAll('.views button').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.view === view)));
  syncCount();
  syncSim();


  state.lanes = v.lanes.map(buildLane);
}

function syncSim() {
  const info = SIMS[state.sim];
  $('sim').value = state.sim;
  $('gravityLabel').textContent = info.gravity;
  $('soloWhat').textContent = info.what;
  $('clickHint').textContent = info.click;
  const hash = `#${state.view}/${state.sim}`;
  if (location.hash !== hash) history.replaceState(null, '', hash);
}

function setSim(sim) {
  if (!SIMS[sim] || state.benching) return;
  state.sim = sim;
  state.holes = [];
  syncSim();
  state.lanes.forEach(l => { l.history = []; l.latest = null; placeHoles(l); });
  sendAll({ type: 'sim', sim, holes: [] });
}

function syncCount() {
  $('count').value = countToSlider(state.count);
  $('countOut').textContent = fmt(state.count);
  $('soloCount').textContent = fmt(state.count);
}

function setCount(n) {
  state.count = n;
  syncCount();
  state.lanes.forEach(l => { l.history = []; l.latest = null; });
  sendAll({ type: 'count', count: n });
}

// ---------- benchmark ----------
const sleep = t => new Promise(r => setTimeout(r, t));

async function measure(lane, n) {
  lane.worker.postMessage({ type: 'count', count: n });
  await sleep(800); // let it settle
  lane.samples = [];
  await sleep(1500);
  const s = lane.samples;
  lane.samples = null;
  if (!s.length) return Infinity;
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]; // median frame time
}

async function findLimits() {
  const ready = state.lanes.filter(l => l.ready);
  if (!ready.length) return;
  state.benching = true;
  const btn = $('bench');
  btn.disabled = true;
  const original = state.count;
  const results = [];

  // Test one lane at a time, with the others paused, so they don't
  // compete for the CPU.
  sendAll({ type: 'pause' });
  for (const lane of ready) {
    lane.worker.postMessage({ type: 'resume' });
    let n = 5000, best = 0;
    while (n <= MAX_COUNT) {
      $('resultText').textContent = `Testing ${lane.spec.name} with ${fmt(n)} particles…`;
      const frame = await measure(lane, n);
      if (frame > BUDGET_MS) break;
      best = n;
      n = niceCount(n * 1.3);
    }
    results.push({ lane, best, capped: n > MAX_COUNT });
    lane.worker.postMessage({ type: 'pause' });
  }

  state.benching = false;
  btn.disabled = false;
  setCount(original);
  if (!state.paused) sendAll({ type: 'resume' });

  $('resultText').innerHTML = results.map(({ lane, best, capped }) => {
    const label = `<span class="r-${lane.key}">${lane.spec.name}</span>`;
    if (!best) return `${label} couldn't hold 60 fps even at 5,000 particles.`;
    return `${label} held 60 fps up to ${fmt(best)}${capped ? '+' : ''} particles.`;
  }).join(' ') + ' <span class="note">Your result depends on your computer.</span>';
}

// ---------- controls ----------
function wire() {
  document.querySelectorAll('.views button').forEach(b =>
    b.addEventListener('click', () => setView(b.dataset.view)));

  $('pause').addEventListener('click', () => {
    state.paused = !state.paused;
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
    sendAll({ type: state.paused ? 'pause' : 'resume' });
  });

  $('reset').addEventListener('click', () => {
    if (state.benching) return;
    state.holes = [];
    state.lanes.forEach(l => { l.history = []; placeHoles(l); });
    sendAll({ type: 'reset' });
  });

  const count = $('count');
  count.addEventListener('input', () => { $('countOut').textContent = fmt(sliderToCount(+count.value)); });
  count.addEventListener('change', () => { if (!state.benching) setCount(sliderToCount(+count.value)); });

  const grav = $('gravity');
  grav.value = gravityToSlider(state.gravity);
  const showGravity = () => { $('gravityOut').textContent = sliderToGravity(+grav.value).toFixed(1) + '×'; };
  showGravity();
  grav.addEventListener('input', () => {
    showGravity();
    state.gravity = sliderToGravity(+grav.value);
    sendAll({ type: 'gravity', value: state.gravity });
  });

  $('sim').addEventListener('change', e => setSim(e.target.value));
  $('bench').addEventListener('click', findLimits);
  window.addEventListener('hashchange', () => {
    const [view, sim] = location.hash.slice(1).split('/');
    if (SIMS[sim] && sim !== state.sim) setSim(sim);
    if (VIEWS[view] && view !== state.view) setView(view);
  });
}

// ---------- start ----------
function supported() {
  return 'Worker' in window && 'OffscreenCanvas' in window &&
    'transferControlToOffscreen' in HTMLCanvasElement.prototype && 'WebAssembly' in window;
}

if (!supported()) {
  document.querySelector('main').innerHTML =
    '<p id="unsupported">This demo needs a recent browser (Chrome, Edge, Firefox, or Safari 16.4 or later), because each language runs in its own background thread and draws with OffscreenCanvas.</p>';
} else {
  drawTicks();
  wire();
  const [view, sim] = location.hash.slice(1).split('/');
  if (SIMS[sim]) state.sim = sim;
  setView(VIEWS[view] ? view : 'side');
}
