// マンション — a mid-rise apartment block for the tall KeiSim buildings (13–42 m), built with the
// Sakuragaoka house kit (GB/Frame boxes, house materials, anime glass, laundry) so it shares the
// houses' look. The body fills the KeiSim footprint exactly; the setback strip becomes a paved
// forecourt with planting.
//
// F: lot frame at the frontage centre (+Z toward the road); lot local x in [-w/2, w/2], z in [-depth, 0].
// b: the KeiSim building (h, setback, w = depth of the box).
const WALLS = ['#ece6da', '#e4d9c7', '#d8cab4', '#cdbba4', '#e9e6df', '#bfae9a', '#d9d4ca'];
const NAMES = [['メゾン', 'MAISON'], ['コーポ', 'CORPO'], ['ハイツ', 'HEIGHTS'], ['グラン', 'GRAND'], ['パレス', 'PALACE'], ['レジデンス', 'RESIDENCE']];
const PLACES = ['さくら', '若葉', '緑町', '朝日', '春日', '桜台', '柳', '白梅', 'ひかり', '中央'];

export function buildMansion(H, F, w, depth, b, r) {
  const { M, P, ctx } = H;
  const res = { gardenSpots: [], bikeSpots: [], wallTops: [] };
  const FH = 2.95, base = 0.45;
  // KeiSim's 16–42 m boxes, compressed to 4–10 storeys (a suburb, not a CBD)
  const floors = Math.max(4, Math.min(10, Math.round(4 + (b.h - 16) * 0.23)));
  const top = base + floors * FH;
  const bw = w - 0.3, bd = Math.max(6, b.w - 0.3);
  const z1 = -b.setback - 0.15, z0 = z1 - bd, cz = (z0 + z1) / 2;       // front / back faces
  const wall = r.pick(WALLS), band = '#f3f1ec', frame = r.pick(['#8e949b', '#6d6a66', '#b9bcbf']);
  const parapetGlass = r() < 0.45;
  const parapet = r.pick(['#f1eee7', '#e6e2da', '#cfd3d6', wall]);

  // ------------------------------------------------------------------ body + slab bands + roof
  F.boxB(M.concrete, '#bdbcb5', bw + 0.1, base + 0.3, bd + 0.1, 0, -0.3, cz, { uv: { world: 2 } });
  F.boxB(M.tile, wall, bw, top - base, bd, 0, base, cz, { uv: { world: 1.2 } });
  H.col(F, 0, cz, bw + 0.1, bd + 0.1, 0, -1, top + 2);
  for (let f = 1; f < floors; f++) F.box(M.plain, band, bw + 0.08, 0.16, bd + 0.08, 0, base + f * FH, cz, { skip: 'td' });
  F.boxB(M.plain, band, bw + 0.14, 0.9, bd + 0.14, 0, top, cz, { skip: 'd' });                 // roof parapet
  F.boxB(M.concrete, '#a9a79f', bw - 0.3, 0.05, bd - 0.3, 0, top + 0.5, cz, { uv: { world: 2 } }); // roof deck (inside the parapet)
  const px = (r() - 0.5) * bw * 0.4;
  F.boxB(M.tile, wall, 3.2, 2.6, 3.4, px, top, cz - bd * 0.15, { uv: { world: 1.2 } });          // stair / lift house
  F.boxB(M.plain, band, 3.4, 0.2, 3.6, px, top + 2.6, cz - bd * 0.15);
  F.cyl(M.plain, '#dcdad3', 0.9, 1.6, px + (px > 0 ? -3.2 : 3.2), top + 1.4, cz + bd * 0.1, { seg: 12 }); // water tank
  F.box(M.plain, '#9aa1a8', 0.05, 2.2, 0.05, px + 1.2, top + 2.6 + 1.1, cz - bd * 0.15);        // antenna mast

  // ------------------------------------------------------------------ front: balconies per unit
  const units = Math.max(2, Math.round(bw / 5.6)), uw = bw / units;
  const zb = z1 + 1.25;                                    // balcony outer edge
  for (let f = 1; f < floors; f++) {
    const y = base + f * FH;
    F.box(M.concrete, '#d9d6cf', bw, 0.18, 1.25, 0, y, z1 + 0.62, { uv: { world: 2 } });
    if (parapetGlass) {
      F.boxB(M.frost, null, bw - 0.1, 0.95, 0.03, 0, y + 0.09, zb - 0.04, { shadow: false });
      F.boxB(M.plain, parapet, bw, 0.12, 0.1, 0, y + 0.09, zb - 0.04);
    } else F.boxB(M.plain, parapet, bw, 1.02, 0.1, 0, y + 0.09, zb - 0.05);
    F.box(M.plain, frame, bw + 0.02, 0.05, 0.1, 0, y + 1.14, zb - 0.04);
    for (let u = 0; u < units; u++) {
      const ux = -bw / 2 + (u + 0.5) * uw;
      if (u > 0) F.boxB(M.plain, '#e8e4dc', 0.07, FH - 0.2, 1.2, -bw / 2 + u * uw, y + 0.09, z1 + 0.62);   // 隔て板
      const gw = Math.min(uw - 1.4, 2.6);
      F.box(M.plain, frame, gw + 0.1, 2.08, 0.04, ux - 0.2, y + 1.14, z1 + 0.02);
      F.box(M.glass, null, gw, 2.0, 0.03, ux - 0.2, y + 1.14, z1 + 0.05, { shadow: false });
      F.box(M.plain, frame, 0.04, 2.0, 0.05, ux - 0.2, y + 1.14, z1 + 0.06);
      F.boxB(M.plain, '#e9e8e2', 0.78, 0.56, 0.28, ux + uw / 2 - 0.6, y + 0.09, z1 + 0.3);                // AC outdoor unit
      F.box(M.plain, '#8b8f94', 0.5, 0.5, 0.02, ux + uw / 2 - 0.6, y + 0.37, z1 + 0.445, { shadow: false });
      if (r() < 0.32) H.laundry.line(F, ux - uw / 2 + 0.25, ux + uw / 2 - 1.05, y + 2.05, z1 + 0.85, r, r() < 0.5 ? 'balcony' : 'family');
    }
  }
  // ground floor: entrance hall (glass) + canopy + name plate, shutters elsewhere
  const ex = (r() - 0.5) * (bw - 5);
  F.box(M.plain, frame, 2.6, 2.5, 0.06, ex, base + 1.25, z1 + 0.03);
  F.box(M.glass, null, 2.5, 2.4, 0.04, ex, base + 1.25, z1 + 0.07, { shadow: false });
  F.box(M.lamp, '#fff4dc', 2.3, 2.2, 0.02, ex, base + 1.2, z1 - 0.3, { shadow: false });          // lit lobby behind the glass
  F.box(M.concrete, '#d9d6cf', 3.4, 0.2, 1.8, ex, base + 2.75, z1 + 0.9, { uv: { world: 2 } });
  const [pre, en] = r.pick(NAMES), place = r.pick(PLACES);
  const name = `${pre}${place}`;
  const signTex = ctx.tex.draw(512, 128, (g, W, Hh) => {
    g.fillStyle = '#f7f4ee'; g.fillRect(0, 0, W, Hh);
    g.fillStyle = '#3a3346'; g.textAlign = 'center'; g.textBaseline = 'middle';
    ctx.tex.fitText(g, name, W / 2, Hh * 0.42, W * 0.86, 64, '"Noto Serif JP", serif', 700);
    g.font = '500 20px "Noto Sans JP", sans-serif'; g.fillText(en, W / 2, Hh * 0.82);
  }, { key: 'mansion|' + name + en });
  F.box(ctx.mat.toon('#ffffff', { map: signTex, paint: 0.02 }), null, 2.2, 0.55, 0.06, ex, base + 3.3, z1 + 0.05, { shadow: false });
  for (const s of [-1, 1]) {                                                                       // closed shutters / wall panels
    const xa = s < 0 ? -bw / 2 + 0.3 : ex + 1.6, xb = s < 0 ? ex - 1.6 : bw / 2 - 0.3;
    if (xb - xa > 1.2) F.box(M.shutter, '#b9bcbf', xb - xa, 2.3, 0.05, (xa + xb) / 2, base + 1.25, z1 + 0.03, { uv: { world: 1 } });
  }

  // ------------------------------------------------------------------ sides + back: small windows, corridor
  for (const s of [-1, 1]) {
    for (let f = 1; f < floors; f++) {
      const y = base + f * FH + 1.3;
      for (const zz of [z1 - bd * 0.3, z1 - bd * 0.72]) {
        F.box(M.plain, frame, 0.05, 0.95, 0.85, s * (bw / 2 + 0.02), y, zz);
        F.box(M.frost, null, 0.03, 0.85, 0.75, s * (bw / 2 + 0.04), y, zz, { shadow: false });
      }
    }
  }
  for (let f = 1; f < floors; f++) {                                                                // open corridor (外廊下) on the back
    const y = base + f * FH;
    F.box(M.concrete, '#d9d6cf', bw, 0.18, 1.3, 0, y, z0 - 0.65, { uv: { world: 2 } });
    F.boxB(M.plain, band, bw, 1.05, 0.12, 0, y + 0.09, z0 - 1.24);
    for (let u = 0; u < units; u++) F.box(M.plain, '#8a8176', 0.85, 2.0, 0.05, -bw / 2 + (u + 0.5) * uw, y + 1.1, z0 - 0.03); // doors
  }

  // ------------------------------------------------------------------ forecourt (setback strip)
  H.groundRect(F, -w / 2 + 0.1, z1, w / 2 - 0.1, -0.05, M.paver, '#cfc9bd', 0.03, 1.5);
  if (b.setback >= 1.6) {
    for (let x = -w / 2 + 0.8; x < w / 2 - 0.6; x += 1.4 + r() * 0.8) {
      if (Math.abs(x - ex) < 2.0) continue;
      P.bush(F, x, H.gy(F, x, -0.7), -0.7, 0.35 + r() * 0.12, r, { pal: r() < 0.5 ? 'green' : 'dark', flowers: r() < 0.3 ? 'azalea' : null, fn: 5 });
    }
  }
  { const p = F.w(ex + 2.4, 0, -0.9); res.bikeSpots.push({ x: p.x, z: p.z, rotY: F.ry }); }
  return res;
}
