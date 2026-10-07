// The Rust lane: loads swarm.wasm and talks to it through plain
// exported functions, reading arrays straight out of its memory.

export async function create() {
  const url = new URL('../swarm.wasm', import.meta.url);
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(fetch(url), {}));
  } catch {
    // Fallback if the server sends the wrong content type for .wasm
    const bytes = await (await fetch(url)).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  }
  const ex = instance.exports;
  const mem = ex.memory;
  let w = 0, h = 0;

  // Views must be made fresh after any call that can grow memory
  const f32 = (ptr, len) => new Float32Array(mem.buffer, ptr, len);

  return {
    init(st) {
      const n = st.x.length;
      ex.init(n, st.mode);
      for (const k of ['x', 'y', 'vx', 'vy', 'ox', 'oy', 'svx', 'svy', 'age', 'life']) {
        if (st[k].length) f32(ex[`ptr_${k}`](), st[k].length).set(st[k]);
      }
    },

    setSize(W, H, lut) {
      w = W; h = H;
      ex.resize(w, h);
      new Uint32Array(mem.buffer, ex.ptr_lut(), 256).set(lut);
    },

    setParams(p) {
      ex.set_gravity(p.g);
      ex.clear_holes();
      for (const [hx, hy] of p.holes) ex.add_hole(hx, hy);
    },

    step(dt) { ex.step(dt); },

    draw() {
      const ptr = ex.draw();
      return new Uint8ClampedArray(mem.buffer, ptr, w * h * 4);
    },
  };
}
