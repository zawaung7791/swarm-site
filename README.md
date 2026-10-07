# Swarm

Three particle simulations (a spiral galaxy, fireworks, and a flock of
birds), each written in Rust (WebAssembly), JavaScript and Python with NumPy
(via Pyodide), raced side by side in the browser. Each language runs in its
own Web Worker and draws to its own canvas.

- Spiral galaxy: gravity from a central mass. Click to drop black holes.
- Fireworks: bursts with gravity, air drag and fading sparks. Black holes
  swallow sparks.
- Bird flock: each bird matches the heading of the birds around it, using a
  grid of neighbourhood cells so it scales to millions. Click to send in a
  hawk.

There's also a WebGPU lane, which runs all three simulations as compute
shaders on the graphics chip, including the drawing: straight after
moving, each particle adds itself to its pixel's count with an atomic
operation, and a final pass colours the pixels straight onto the canvas.
Each frame goes to the GPU as one batch, and the GPU timestamps its own
passes, so the lane's Physics figure includes the pixel counting. It goes up to 16 million
particles (the CPU lanes stop at 4 million). It needs a browser with
WebGPU in workers, such as a recent Chrome or Edge.

Views: Side by side (Rust and JavaScript), Three-way race (plus NumPy),
CPU vs GPU (Rust and WebGPU), and full-screen Rust only or WebGPU only.

The Rust lane uses WebAssembly SIMD in all three simulations, handling four
particles per instruction. The flock's breeze uses a fast polynomial sine
(accurate to about 0.1%), and all three languages use that same
approximation, so the race stays like for like.

Links can open a particular view and simulation, for example
https://swarm.zawaung.site/#race/flock or #gpu/galaxy

Live at https://swarm.zawaung.site

## Folder layout

```
swarm-site/
├── Dockerfile
├── docker-compose.yml
├── nginx.conf                 nginx inside the container
├── swarm.zawaung.site.conf    nginx on the VPS (host)
├── rust/
│   ├── Cargo.toml
│   └── src/lib.rs             the Rust lane
└── web/
    ├── index.html
    ├── app.js                 page logic, controls, benchmark
    ├── lane-worker.js         runs one lane in a background thread
    ├── sims.js                shared starting particles and colours
    └── backends/
        ├── rust.js            loads swarm.wasm
        ├── js.js              the JavaScript lane
        ├── gpu.js             the WebGPU lane (shaders included)
        └── numpy.js           the Python + NumPy lane
```

## Deploy

1. DNS: add an A record for `swarm` pointing at the VPS's IP address.

2. Build and start the container:

   ```bash
   cd swarm-site
   docker compose up -d --build
   curl -I http://127.0.0.1:8086/
   ```

3. nginx on the VPS:

   ```bash
   sudo cp swarm.zawaung.site.conf /etc/nginx/sites-available/swarm.zawaung.site
   sudo ln -s /etc/nginx/sites-available/swarm.zawaung.site /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   sudo certbot --nginx -d swarm.zawaung.site
   ```

## Updating

Edit the files, then run `docker compose up -d --build` again.

## Notes

- Port 8086 must be free on the VPS. If it isn't, change it in both
  `docker-compose.yml` and `swarm.zawaung.site.conf`.
- The NumPy lane downloads Pyodide and NumPy (about 10 MB) from the
  jsDelivr CDN the first time; browsers cache it after that.
- Needs a recent browser: Chrome, Edge, Firefox, or Safari 16.4 or later.
