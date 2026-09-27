// Non-house parcels: 畑 fields, 月極駐車場 gravel car parks, 空き地 vacant lots and small 公園 parks.
// field / parking / vacant are adapted from Sakuragaoka Station src/world/houses.js plot() (MIT, see
// vendor/sakuragaoka/LICENSE), changed to work in a rotated lot frame F (local x in [x0, x1],
// z in [z0, z1]) with ground heights from H.gy.

export function plotParcel(H, F, r, x0, x1, z0, z1, kind) {
  const { M, P } = H;
  const gy = (x, z) => H.gy(F, x, z);
  if (kind === 'field') { // 畑: soil with green rows, a few stakes, sometimes a shed or a vinyl tunnel
    H.groundRect(F, x0, z0, x1, z1, M.soil, '#c9b89c', 0.02, 1.5);
    const along = (x1 - x0) > (z1 - z0);
    const n = Math.floor(((along ? z1 - z0 : x1 - x0) - 1) / 0.9);
    for (let i = 0; i < n; i++) {
      const t = 0.6 + i * 0.9;
      const col = r.pick(['#8fb86f', '#7aa564', '#a9c77c', '#6f9a5a']);
      const leafy = r() < 0.7;
      if (along) {
        const zz = z0 + t; H.groundRect(F, x0 + 0.5, zz - 0.18, x1 - 0.5, zz + 0.18, M.soil, '#b5a283', 0.05, 1.5);
        if (leafy) for (let x = x0 + 0.8; x < x1 - 0.6; x += 0.8 + r() * 0.3) F.raw(M.plain, col, H.blobRaw(Math.floor(r() * 6), 0), x, gy(x, zz) + 0.12, zz, { sx: 0.5, sy: 0.22, sz: 0.36 });
      } else {
        const xx = x0 + t; H.groundRect(F, xx - 0.18, z0 + 0.5, xx + 0.18, z1 - 0.5, M.soil, '#b5a283', 0.05, 1.5);
        if (leafy) for (let z = z0 + 0.8; z < z1 - 0.6; z += 0.8 + r() * 0.3) F.raw(M.plain, col, H.blobRaw(Math.floor(r() * 6), 0), xx, gy(xx, z) + 0.12, z, { sx: 0.36, sy: 0.22, sz: 0.5 });
      }
    }
    if (r() < 0.6) { P.shed(F.sub(x1 - 1.2, 0, z0 + 0.8, 0), 0, gy(x1 - 1.2, z0 + 0.8), 0, r, 1.4); H.col(F, x1 - 1.2, z0 + 0.8, 1.6, 1.0, 0, -1, 2.5); }
    if (r() < 0.5 && z1 - z0 > 6) { // ビニールハウス tunnel
      const gx = x0 + 2.5, gz0 = z0 + 1, gz1 = Math.min(z1 - 1, z0 + 7);
      const g = gy(gx, (gz0 + gz1) / 2);
      F.cyl(M.poly, null, 1.3, gz1 - gz0, gx, g, (gz0 + gz1) / 2, { rx: Math.PI / 2, seg: 10, open: true, noOutline: true, shadow: false });
      for (let z = gz0; z <= gz1 + 0.01; z += 1.2) F.cyl(M.plain, '#c9ccd1', 1.31, 0.03, gx, g, z, { rx: Math.PI / 2, seg: 10, open: true });
    }
    for (let k = 0; k < 4; k++) { const x = x0 + 1 + r() * (x1 - x0 - 2), z = z0 + 1 + r() * (z1 - z0 - 2); F.boxB(M.plain, '#b48a62', 0.03, 0.9, 0.03, x, gy(x, z), z); }
  } else if (kind === 'parking') { // 月極駐車場: gravel, rope lines, a sign, wheel stops
    H.groundRect(F, x0, z0, x1, z1, M.gravel, '#cfc9bd', 0.025, 1.5);
    const along = (x1 - x0) > (z1 - z0);
    const n = Math.max(1, Math.floor((along ? x1 - x0 : z1 - z0) / 2.6));
    for (let i = 0; i <= n; i++) {
      if (along) { const x = x0 + i * (x1 - x0) / n; H.groundRect(F, x - 0.05, z0 + 0.3, x + 0.05, z0 + 4.8, M.plain, '#ece8df', 0.035, 1); }
      else { const z = z0 + i * (z1 - z0) / n; H.groundRect(F, x0 + 0.3, z - 0.05, x0 + 4.8, z + 0.05, M.plain, '#ece8df', 0.035, 1); }
    }
    const px = x0 + 0.4, pz = z1 - 0.4, g = gy(px, pz);
    F.boxB(M.plain, '#8e949b', 0.06, 1.3, 0.06, px, g, pz);
    F.box(M.atlas, '#ffffff', 0.7, 0.26, 0.03, px, g + 1.3, pz + 0.04, { uv: { rect: H.A.rects.kotatsu_sign, white: H.A.white } });
    for (let k = 0; k < 3; k++) { const x = x0 + 1.5 + r() * (x1 - x0 - 3), z = z0 + 1 + r() * (z1 - z0 - 2); F.boxB(M.concrete, '#b9b8b2', 0.6, 0.12, 0.15, x, gy(x, z), z, { uv: { world: 2 } }); }
  } else if (kind === 'park') { // 児童公園: gravel square, lawn edge, trees, benches, a slide and a low fence
    H.groundRect(F, x0, z0, x1, z1, M.lawn, '#b4c98f', 0.02, 2);
    H.groundRect(F, x0 + 1.4, z0 + 1.4, x1 - 1.4, z1 - 1.4, M.gravel, '#d8cfbe', 0.03, 1.5);
    for (const [xa, xb, zz] of [[x0, x1, z0 + 0.1], [x0, x1, z1 - 0.1]]) H.meshPanel(F, xa + 0.2, xb - 0.2, zz, 0, 0.9, '#8e949b');
    const trees = 2 + Math.floor(r() * 3);
    for (let k = 0; k < trees; k++) {
      const x = x0 + 1.2 + r() * (x1 - x0 - 2.4), z = k % 2 ? z0 + 1.0 + r() * 0.6 : z1 - 1.0 - r() * 0.6;
      P.tree(F, x, gy(x, z), z, 1.6 + r() * 0.8, r, 'round'); H.colC(F, x, z, 0.18, -1, 5);
    }
    const bx = (x0 + x1) / 2 + (r() - 0.5) * 2, bz = z0 + 2.2;
    F.boxB(M.wood, '#a8805e', 1.6, 0.06, 0.4, bx, gy(bx, bz) + 0.4, bz);               // bench seat
    for (const s of [-1, 1]) F.boxB(M.concrete, '#b9b8b2', 0.1, 0.4, 0.36, bx + s * 0.65, gy(bx, bz), bz, { uv: { world: 2 } });
    const sx = (x0 + x1) / 2 + (r() < 0.5 ? -1 : 1) * (x1 - x0) * 0.2, sz = (z0 + z1) / 2;  // slide (すべり台)
    const g = gy(sx, sz), col = r.pick(['#e36b5e', '#f2c230', '#5aa0d8']);
    F.boxB(M.plain, '#c9ccd1', 1.0, 1.4, 1.0, sx, g, sz - 0.6, { skip: 'd' });
    F.box(M.plain, col, 0.6, 0.06, 2.4, sx, g + 0.75, sz + 0.95, { rx: -0.52 });
    F.boxB(M.plain, col, 1.1, 0.12, 1.1, sx, g + 1.4, sz - 0.6);
    H.col(F, sx, sz, 1.1, 3.0, 0, -1, 2);
  } else { // 空き地: weeds, a few bushes, sometimes a tree
    H.groundRect(F, x0, z0, x1, z1, M.lawn, '#b4c98f', 0.02, 2);
    for (let k = 0; k < 8; k++) { const x = x0 + 0.8 + r() * (x1 - x0 - 1.6), z = z0 + 0.8 + r() * (z1 - z0 - 1.6); P.bush(F, x, gy(x, z) - 0.05, z, 0.25 + r() * 0.3, r, { pal: 'young', n: 2, flowers: r() < 0.4 ? 'flowers' : null, fn: 4 }); }
    if (r() < 0.5) { const x = (x0 + x1) / 2, z = (z0 + z1) / 2; P.tree(F, x, gy(x, z), z, 1.3, r, 'round'); H.colC(F, x, z, 0.15, -1, 5); }
  }
}
