// 路面の作り込み — opt-in road detailing (?roads=detail) on top of roads.js: painted markings (diamonds before
// crosswalks, yellow speed-limit numerals, blue bicycle chevrons), coloured pavement (red-brown junctions, green
// 路側帯 on residential roads), manholes, hydrant and valve covers, gratings, patched and sealed asphalt, tactile
// paving, guard pipes, curve mirrors and road signs. Everything is placed from the KeiSim town export with seeded
// randomness and lies off the lanes or flat on the road: KeiSim's geometry is unchanged. The default look
// (?roads=plain, what the KeiView-trained driving models know) does not load this module at all.
import * as THREE from 'three';
import { SIDEWALK_Y, SURF } from '../town.js';
import { SEM } from '../ego.js';

const TAU = Math.PI * 2;
const YELLOW = '#f0b52e';            // same paint as the yellow centre lines (roads.js)

/** Flat quads on the ground with UVs into an atlas, merged into one geometry. */
class Quads {
  constructor() { this.P = []; this.U = []; this.I = []; }
  /** centre (x, z) at height y; (fx, fz) unit forward = texture up; w across, l along; r = [u0, v0, u1, v1] */
  add(x, y, z, fx, fz, w, l, r) {
    const rx = -fz, rz = fx, a = w / 2, b = l / 2, n = this.P.length / 3;
    for (const [s, t, u, v] of [[-a, -b, r[0], r[1]], [a, -b, r[2], r[1]], [a, b, r[2], r[3]], [-a, b, r[0], r[3]]]) {
      this.P.push(x + rx * s + fx * t, y, z + rz * s + fz * t); this.U.push(u, v);
    }
    this.I.push(n, n + 1, n + 2, n, n + 2, n + 3);   // counter-clockwise seen from above: faces up
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.P, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.P.map((_, i) => (i % 3 === 1 ? 1 : 0)), 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.U, 2));
    g.setIndex(this.I);
    return g;
  }
}

// ==================================================================== textures
/** Canvas atlas: cells {name: [u0, v0, u1, v1]} (texture v up), an opaque colour texture and an alpha mask.
 *  Canvas pixels that are fully transparent read back black: mipmapping would blend that black into thin paint
 *  (a white diamond turns dark grey in the distance), so the colours are grown into the transparent texels and
 *  the alpha goes to a separate mask (material alphaMap). */
function atlas(ctx, key, W, H, layout, paint) {
  const cells = {};
  const { canvas, g } = ctx.tex.canvas(W, H);
  g.clearRect(0, 0, W, H);
  for (const [name, x, y, w, h] of layout) { g.save(); paint[name](g, x, y, w, h); g.restore(); }
  for (const [name, x, y, w, h] of layout) cells[name] = [(x + 1) / W, 1 - (y + h - 1) / H, (x + w - 1) / W, 1 - (y + 1) / H];
  const m = ctx.tex.canvas(W, H);
  const img = g.getImageData && g.getImageData(0, 0, W, H);
  if (img && img.data && img.data.length === W * H * 4) {
    const d = img.data, n = W * H;
    const mask = m.g.createImageData(W, H), md = mask.data;
    const done = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const a = d[i * 4 + 3];
      md[i * 4] = md[i * 4 + 1] = md[i * 4 + 2] = a; md[i * 4 + 3] = 255;
      if (a > 0) {                          // un-premultiplied colour stays; mark as a seed for the dilation
        done[i] = 1;
      }
    }
    let front = [];
    for (let i = 0; i < n; i++) if (done[i]) front.push(i);
    for (let pass = 0; pass < 10 && front.length; pass++) {   // grow the colours 10 px into the transparent texels
      const next = [];
      for (const i of front) {
        const x = i % W, y = (i / W) | 0;
        for (const j of [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1]) {
          if (j < 0 || done[j]) continue;
          done[j] = 1; d[j * 4] = d[i * 4]; d[j * 4 + 1] = d[i * 4 + 1]; d[j * 4 + 2] = d[i * 4 + 2];
          next.push(j);
        }
      }
      front = next;
    }
    for (let i = 0; i < n; i++) d[i * 4 + 3] = 255;
    g.putImageData(img, 0, 0);
    m.g.putImageData(mask, 0, 0);
  }
  const tex = ctx.tex.finish(canvas, { anisotropy: 8 });
  const alpha = ctx.tex.finish(m.canvas, { anisotropy: 8, srgb: false });
  return { tex, alpha, cells };
}

function seeded(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

function roadAtlas(ctx) {
  const F = ctx.tex.FONTS;
  const disc = (g, cx, cy, r, fill) => { g.fillStyle = fill; g.beginPath(); g.arc(cx, cy, r, 0, TAU); g.fill(); };
  const ring = (g, cx, cy, r, w, stroke) => { g.lineWidth = w; g.strokeStyle = stroke; g.beginPath(); g.arc(cx, cy, r, 0, TAU); g.stroke(); };
  const text = (g, s, x, y, size, color, weight = 900, font = F.sans) => {
    g.font = `${weight} ${size}px ${font}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = color; g.fillText(s, x, y);
  };
  const cover = (g, x, y, w) => {         // round cast-iron cover with its frame; clips to the lid
    const cx = x + w / 2, cy = y + w / 2, r = w / 2 - 8;
    disc(g, cx, cy, r, '#6c6e74');
    ring(g, cx, cy, r - 5, 6, '#4d4f55');
    g.beginPath(); g.arc(cx, cy, r - 9, 0, TAU); g.clip();
    return [cx, cy, r - 9];
  };
  const P = {
    manholeA(g, x, y, w) {                // 亀甲 (hexagon) pattern, the common sewer lid
      const [cx, cy, r] = cover(g, x, y, w);
      disc(g, cx, cy, r, '#585a60');
      g.strokeStyle = 'rgba(160,162,170,0.85)'; g.lineWidth = 3;
      const s = 13, h = s * Math.sqrt(3);
      for (let i = -12; i <= 12; i++) for (let j = -12; j <= 12; j++) {
        const hx = cx + i * s * 1.5, hy = cy + j * h + (i & 1 ? h / 2 : 0);
        if (Math.hypot(hx - cx, hy - cy) > r + s) continue;
        g.beginPath();
        for (let k = 0; k <= 6; k++) { const a = k * Math.PI / 3; const px = hx + s * 0.86 * Math.cos(a), py = hy + s * 0.86 * Math.sin(a); if (k) g.lineTo(px, py); else g.moveTo(px, py); }
        g.stroke();
      }
      disc(g, cx, cy, 34, '#4c4e54'); ring(g, cx, cy, 34, 4, 'rgba(170,172,178,0.9)');
      text(g, '下水', cx, cy + 2, 28, 'rgba(178,180,186,0.95)');
    },
    manholeB(g, x, y, w) {                // a town's design lid: sakura over the river, in colour
      const [cx, cy, r] = cover(g, x, y, w);
      disc(g, cx, cy, r, '#7fb6d8');
      g.fillStyle = '#5d93bb';
      for (let k = 0; k < 4; k++) { g.beginPath(); g.ellipse(cx, cy + 40 + k * 14, r, 6, 0, 0, TAU); g.fill(); }
      g.strokeStyle = '#6b4a3a'; g.lineWidth = 9; g.lineCap = 'round';
      g.beginPath(); g.moveTo(cx - r, cy - 10); g.quadraticCurveTo(cx - 20, cy - 40, cx + r, cy - 70); g.stroke();
      g.beginPath(); g.moveTo(cx - 30, cy - 28); g.quadraticCurveTo(cx - 10, cy + 10, cx + 30, cy + 20); g.stroke();
      const R = seeded(77);
      for (let k = 0; k < 14; k++) {
        const bx = cx + (R() - 0.5) * r * 1.6, by = cy + (R() - 0.7) * r * 1.3, br = 9 + R() * 9;
        if (Math.hypot(bx - cx, by - cy) > r - br) continue;
        g.fillStyle = R() < 0.5 ? '#f4a6c0' : '#f7c3d4';
        for (let p = 0; p < 5; p++) { const a = p * TAU / 5 + R(); g.beginPath(); g.ellipse(bx + Math.cos(a) * br * 0.55, by + Math.sin(a) * br * 0.55, br * 0.5, br * 0.36, a, 0, TAU); g.fill(); }
        disc(g, bx, by, br * 0.22, '#e46b8f');
      }
      g.fillStyle = 'rgba(40,40,46,0.55)'; g.fillRect(cx - r, cy + r - 34, 2 * r, 34);
      text(g, 'げすいどう', cx, cy + r - 17, 20, '#eef3f6', 700);
    },
    manholeC(g, x, y, w) {                // concentric rings and ribs (water / gas)
      const [cx, cy, r] = cover(g, x, y, w);
      disc(g, cx, cy, r, '#5b5d63');
      g.strokeStyle = 'rgba(160,162,170,0.8)'; g.lineWidth = 4;
      for (const rr of [28, 56, 84]) { g.beginPath(); g.arc(cx, cy, rr, 0, TAU); g.stroke(); }
      for (let k = 0; k < 24; k++) {
        const a = k * TAU / 24; g.beginPath(); g.moveTo(cx + Math.cos(a) * 30, cy + Math.sin(a) * 30); g.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r); g.stroke();
      }
      text(g, '汚水', cx, cy + 2, 24, 'rgba(178,180,186,0.95)');
    },
    hydrant(g, x, y, w) {                 // 消火栓: yellow-painted surround, lid with raised letters
      const cx = x + w / 2, cy = y + w / 2;
      disc(g, cx, cy, w / 2 - 6, '#e9bd2c');
      disc(g, cx, cy, w / 2 - 34, '#5d5f65'); ring(g, cx, cy, w / 2 - 38, 5, '#45474c');
      text(g, '消火栓', cx, cy + 2, 40, 'rgba(186,188,194,0.95)');
    },
    valve(g, x, y, w) {                   // 仕切弁: small square lid
      g.fillStyle = '#6a6c72'; g.fillRect(x + 8, y + 8, w - 16, w - 16);
      g.fillStyle = '#56585e'; g.fillRect(x + 16, y + 16, w - 32, w - 32);
      text(g, '仕切弁', x + w / 2, y + w / 2 + 1, 24, 'rgba(180,182,188,0.95)', 700);
    },
    gas(g, x, y, w) {
      const cx = x + w / 2, cy = y + w / 2;
      disc(g, cx, cy, w / 2 - 10, '#6c6e74'); disc(g, cx, cy, w / 2 - 20, '#55575d');
      text(g, 'ガス', cx, cy + 1, 28, '#e5c23a', 700);
    },
    grate(g, x, y, w, h) {                // グレーチング over the gutter: bars across it
      g.fillStyle = '#4b4d52'; g.fillRect(x + 6, y + 6, w - 12, h - 12);
      g.fillStyle = '#17181b'; g.fillRect(x + 14, y + 14, w - 28, h - 28);
      g.fillStyle = '#80838a';
      for (let yy = y + 16; yy < y + h - 16; yy += 9) g.fillRect(x + 14, yy, w - 28, 4);
      g.fillRect(x + w / 2 - 2, y + 14, 4, h - 28);
    },
    diamond(g, x, y, w, h) {              // ◇ 横断歩道又は自転車横断帯あり (white)
      g.strokeStyle = '#f4f2ec'; g.lineWidth = 15; g.lineJoin = 'miter';
      g.beginPath(); g.moveTo(x + w / 2, y + 12); g.lineTo(x + w - 14, y + h / 2); g.lineTo(x + w / 2, y + h - 12); g.lineTo(x + 14, y + h / 2); g.closePath(); g.stroke();
    },
    chevron(g, x, y, w, h) {              // 矢羽根: blue bicycle guidance chevron, pointing forward
      g.fillStyle = '#2f6fd0';
      g.beginPath(); g.moveTo(x + 8, y + 150); g.lineTo(x + w / 2, y + 24); g.lineTo(x + w - 8, y + 150);
      g.lineTo(x + w - 8, y + 226); g.lineTo(x + w / 2, y + 100); g.lineTo(x + 8, y + 226); g.closePath(); g.fill();
    },
    dots(g, x, y, w) {                    // 点状ブロック (warning)
      g.fillStyle = '#e8b417'; g.fillRect(x + 2, y + 2, w - 4, w - 4);
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
        const px = x + 16 + i * 24, py = y + 16 + j * 24;
        disc(g, px + 2, py + 2, 8, '#b98c0c'); disc(g, px, py, 8, '#f8d548');
      }
    },
    bars(g, x, y, w) {                    // 線状ブロック (guiding): bars along the walking direction
      g.fillStyle = '#e8b417'; g.fillRect(x + 2, y + 2, w - 4, w - 4);
      for (let i = 0; i < 4; i++) {
        const px = x + 14 + i * 28;
        g.fillStyle = '#b98c0c'; g.fillRect(px + 3, y + 12, 16, w - 20);
        g.fillStyle = '#f8d548'; g.fillRect(px, y + 9, 16, w - 20);
      }
    },
    patch(g, x, y, w, h) {                // re-surfaced asphalt: darker, sealed seam
      const R = seeded(11), cx = x + w / 2, cy = y + h / 2, pts = [];
      for (let k = 0; k < 14; k++) { const a = k * TAU / 14, rr = 0.36 + R() * 0.08; pts.push([cx + Math.cos(a) * w * rr * 1.05, cy + Math.sin(a) * h * rr]); }
      const path = () => { g.beginPath(); pts.forEach(([px, py], i) => (i ? g.lineTo(px, py) : g.moveTo(px, py))); g.closePath(); };
      g.fillStyle = 'rgba(46,48,54,0.5)'; path(); g.fill();
      g.save(); path(); g.clip();
      for (let k = 0; k < 900; k++) { const v = R() < 0.5 ? 30 : 120; g.fillStyle = `rgba(${v},${v},${v + 4},0.25)`; g.fillRect(x + R() * w, y + R() * h, 2, 2); }
      g.restore();
      g.strokeStyle = 'rgba(24,24,28,0.7)'; g.lineWidth = 4; path(); g.stroke();
    },
    trench(g, x, y, w, h) {               // a utility trench filled in across the lane
      g.fillStyle = 'rgba(48,50,56,0.6)'; g.fillRect(x + 4, y + 10, w - 8, h - 20);
      g.strokeStyle = 'rgba(24,24,28,0.7)'; g.lineWidth = 4; g.strokeRect(x + 4, y + 10, w - 8, h - 20);
    },
    crack(g, x, y, w, h) {                // sealed cracks (black sealant)
      const R = seeded(5);
      g.strokeStyle = 'rgba(26,26,30,0.78)'; g.lineCap = 'round'; g.lineJoin = 'round';
      for (let k = 0; k < 3; k++) {
        let px = x + w * (0.25 + 0.25 * k), py = y + 10;
        g.lineWidth = 4 + R() * 2; g.beginPath(); g.moveTo(px, py);
        while (py < y + h - 12) {
          px = Math.min(x + w - 10, Math.max(x + 10, px + (R() - 0.5) * 34)); py += 14 + R() * 18; g.lineTo(px, py);
          if (R() < 0.12) { g.moveTo(px, py); g.lineTo(px + (R() - 0.5) * 60, py + 20 + R() * 20); g.moveTo(px, py); }
        }
        g.stroke();
      }
    },
  };
  const digits = (n) => (g, x, y, w, h) => {      // 最高速度 (規制標示): yellow numerals, long in the travel direction
    g.translate(x + w / 2, y + h / 2); g.scale(1, 2.75);
    g.font = `900 132px ${F.sans}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = YELLOW;
    g.fillText(n, 0, 6);
  };
  for (const n of ['20', '30', '40', '50']) P['n' + n] = digits(n);
  const layout = [
    ['manholeA', 0, 0, 256, 256], ['manholeB', 256, 0, 256, 256], ['manholeC', 512, 0, 256, 256], ['hydrant', 768, 0, 256, 256],
    ['valve', 1024, 0, 128, 128], ['gas', 1024, 128, 128, 128], ['grate', 1152, 0, 128, 256], ['diamond', 1280, 0, 128, 256],
    ['chevron', 1408, 0, 128, 256], ['dots', 1536, 0, 128, 128], ['bars', 1536, 128, 128, 128], ['patch', 1664, 0, 256, 256],
    ['n20', 0, 256, 192, 512], ['n30', 192, 256, 192, 512], ['n40', 384, 256, 192, 512], ['n50', 576, 256, 192, 512],
    ['crack', 768, 256, 256, 512], ['trench', 1024, 256, 256, 128],
  ];
  return atlas(ctx, 'kv_roaddeco', 2048, 1024, layout, P);
}

function signAtlas(ctx) {
  const F = ctx.tex.FONTS;
  const disc = (g, cx, cy, r, fill) => { g.fillStyle = fill; g.beginPath(); g.arc(cx, cy, r, 0, TAU); g.fill(); };
  const speed = (n) => (g, x, y, w) => {           // 最高速度 (規制標識 323): red ring, blue numerals
    const cx = x + w / 2, cy = y + w / 2;
    disc(g, cx, cy, w / 2 - 3, '#d8302b'); disc(g, cx, cy, w / 2 - 27, '#ffffff');
    g.save(); g.translate(cx, cy + 6); g.scale(0.82, 1);
    g.font = `700 128px ${F.sans}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = '#1f4f9e'; g.fillText(n, 0, 0);
    g.restore();
  };
  const P = {
    crosswalk(g, x, y, w) {               // 横断歩道 (指示標識 407-A)
      g.fillStyle = '#1d60b0'; ctx.tex.roundRect(g, x + 4, y + 4, w - 8, w - 8, 18); g.fill();
      g.strokeStyle = '#ffffff'; g.lineWidth = 6; ctx.tex.roundRect(g, x + 14, y + 14, w - 28, w - 28, 12); g.stroke();
      g.fillStyle = '#ffffff'; g.beginPath(); g.moveTo(x + w / 2, y + 34); g.lineTo(x + w - 30, y + w - 44); g.lineTo(x + 30, y + w - 44); g.closePath(); g.fill();
      g.fillStyle = '#222'; for (let k = 0; k < 4; k++) g.fillRect(x + 72 + k * 30, y + w - 72, 16, 16);
      disc(g, x + w / 2 + 4, y + 92, 12, '#222');
      g.strokeStyle = '#222'; g.lineWidth = 11; g.lineCap = 'round';
      g.beginPath(); g.moveTo(x + w / 2 + 2, y + 108); g.lineTo(x + w / 2 - 4, y + 150); g.stroke();
      g.beginPath(); g.moveTo(x + w / 2 - 4, y + 150); g.lineTo(x + w / 2 - 22, y + 176); g.moveTo(x + w / 2 - 4, y + 150); g.lineTo(x + w / 2 + 16, y + 176); g.stroke();
      g.beginPath(); g.moveTo(x + w / 2 - 20, y + 132); g.lineTo(x + w / 2 + 2, y + 116); g.lineTo(x + w / 2 + 22, y + 136); g.stroke();
    },
    noparking(g, x, y, w) {               // 駐車禁止 (規制標識 316)
      const cx = x + w / 2, cy = y + w / 2;
      disc(g, cx, cy, w / 2 - 3, '#d8302b'); disc(g, cx, cy, w / 2 - 26, '#1d5fae');
      g.save(); g.beginPath(); g.arc(cx, cy, w / 2 - 26, 0, TAU); g.clip();
      g.strokeStyle = '#d8302b'; g.lineWidth = 24; g.beginPath(); g.moveTo(cx - w / 2, cy - w / 2); g.lineTo(cx + w / 2, cy + w / 2); g.stroke();
      g.restore();
    },
    back(g, x, y, w) {
      const cx = x + w / 2, cy = y + w / 2;
      disc(g, cx, cy, w / 2 - 3, '#8f969c'); disc(g, cx, cy, w / 2 - 14, '#a3a9ae');
      g.fillStyle = '#80868c'; g.fillRect(cx - 14, y + 30, 28, w - 60);
    },
    backsq(g, x, y, w) {
      g.fillStyle = '#8f969c'; ctx.tex.roundRect(g, x + 2, y + 2, w - 4, w - 4, 14); g.fill();
      g.fillStyle = '#a3a9ae'; ctx.tex.roundRect(g, x + 12, y + 12, w - 24, w - 24, 8); g.fill();
      g.fillStyle = '#80868c'; g.fillRect(x + w / 2 - 14, y + 30, 28, w - 60);
    },
    mirror(g, x, y, w) {                  // カーブミラー: convex mirror in an orange frame
      const cx = x + w / 2, cy = y + w / 2;
      disc(g, cx, cy, w / 2 - 3, '#ef8a26');
      const gr = g.createRadialGradient(cx - 30, cy - 30, 10, cx, cy, w / 2 - 18);
      gr.addColorStop(0, '#f4f8fb'); gr.addColorStop(0.6, '#c3d3df'); gr.addColorStop(1, '#8ea5b6');
      g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, w / 2 - 18, 0, TAU); g.fill();
      g.fillStyle = 'rgba(120,130,120,0.35)'; g.fillRect(x + 22, cy + 22, w - 44, 40);
    },
  };
  for (const n of ['20', '30', '40', '50']) P['s' + n] = speed(n);
  const layout = [['s20', 0, 0, 256, 256], ['s30', 256, 0, 256, 256], ['s40', 512, 0, 256, 256], ['s50', 768, 0, 256, 256],
    ['crosswalk', 0, 256, 256, 256], ['noparking', 256, 256, 256, 256], ['back', 512, 256, 256, 256], ['mirror', 768, 256, 256, 256],
    ['backsq', 1024, 0, 256, 256]];
  return atlas(ctx, 'kv_roadsigns', 1280, 512, layout, P);
}

// ==================================================================== geometry helpers
const _up = new THREE.Vector3(0, 1, 0), _q = new THREE.Quaternion(), _m = new THREE.Matrix4();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _s = new THREE.Vector3(1, 1, 1);

/** Cylinder from point a to point b ([x, y, z]). */
function rod(a, b, r, seg = 6) {
  _a.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const len = _a.length();
  const g = new THREE.CylinderGeometry(r, r, len, seg, 1, false);
  _q.setFromUnitVectors(_up, _a.normalize());
  _m.compose(_b.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2), _q, _s);
  g.applyMatrix4(_m);
  return g;
}

/** Flat face (disc or square) centred at (x, y, z), front facing horizontal direction (dx, dz), UVs into cell r. */
function face(x, y, z, dx, dz, size, r, round = true, tilt = 0) {
  const g = round ? new THREE.CircleGeometry(size / 2, 28) : new THREE.PlaneGeometry(size, size);
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, r[0] + uv.getX(i) * (r[2] - r[0]), r[1] + uv.getY(i) * (r[3] - r[1]));
  g.rotateX(-tilt);
  g.rotateY(Math.atan2(dx, dz));
  g.translate(x, y, z);
  return g;
}

/** Convex hull of [x, z] points (monotone chain). */
function hull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

// ==================================================================== build
export function build(ctx) {
  const T = ctx.town;
  const R = ctx.rng('kv-roaddeco:' + T.data.seed + ':' + (T.data.style || ''));
  const LH = T.cfg.left_hand_traffic === false ? -1 : 1;       // lanes keep left (KeiSim default)
  const deco = roadAtlas(ctx), sign = signAtlas(ctx);
  const A = deco.cells, S = sign.cells;
  const marks = new Quads(), covers = new Quads(), soft = new Quads(), tactile = new Quads();
  const geos = { white: [], gray: [], orange: [], face: [], back: [] };
  const red = [], green = [];
  const stats = {};
  const count = (k, n = 1) => { stats[k] = (stats[k] || 0) + n; };

  const J = new Map(T.junctions.map((j) => [j.id, j]));
  const obstacles = [...T.trees.map((t) => t.p), ...T.signals.map((s) => s.poleP),
    ...((ctx.services.poles && ctx.services.poles.poles) || []).map((p) => [p.x, p.z])];
  // keep the preset camera spots clear (a sign right in front of the lens)
  for (const v of T.views) if (!v.fly) for (let k = 0; k <= 4; k++) obstacles.push([v.x - Math.sin(v.yaw * Math.PI / 180) * k, v.z - Math.cos(v.yaw * Math.PI / 180) * k]);
  const free = (x, z, r) => obstacles.every((o) => Math.hypot(o[0] - x, o[1] - z) > r);

  /** Frame at arc length s of road r, travelling a -> b (dir 1) or b -> a (dir -1): point, forward, left. */
  const frame = (r, s, dir) => { const f = r.at(s); return { x: f.x, z: f.z, fx: f.tx * dir, fz: f.tz * dir, lx: f.nx * dir * LH, lz: f.nz * dir * LH }; };
  /** Point `lat` m to the traffic side (left with left-hand traffic) of frame F. */
  const side = (F, lat) => [F.x + F.lx * lat, F.z + F.lz * lat];
  const kmh = (r) => Math.max(20, Math.min(50, Math.round(r.speed * 3.6 / 10) * 10));
  const sigEnd = (r, dir) => !!(J.get(dir > 0 ? r.b : r.a) || {}).signalized;   // junction ahead (travel dir)
  const sAt = (r, dir, d) => (dir > 0 ? d : r.len - d);                           // arc length d m after the start

  // ---------------------------------------------------------------- road surface: markings, covers, repairs
  for (const r of T.roads) {
    const narrow = r.kind === 'alley';
    for (const dir of [1, -1]) {
      const busy = [];                     // [d0, d1] distances along this lane holding lane-centre markings
      // speed numerals after the start of the road, in the lane
      if (r.len > 45) {
        const d = 14.5, F = frame(r, sAt(r, dir, d), dir), [x, z] = side(F, r.laneW / 2);
        marks.add(x, 0.009, z, F.fx, F.fz, Math.min(2.0, r.laneW - 0.9), 4.5, A['n' + kmh(r)]);
        busy.push([d - 3, d + 3]); count('speed numerals');
      }
      // diamonds 30 m and 50 m before the stop line of a signalised junction
      if (sigEnd(r, dir) && !narrow) {
        for (const before of [30, 50]) {
          const d = r.len - 5 - before;
          if (d < 23) continue;
          const F = frame(r, sAt(r, dir, d), dir), [x, z] = side(F, r.laneW / 2);
          marks.add(x, 0.009, z, F.fx, F.fz, Math.min(1.5, r.laneW - 1.4), 3.0, A.diamond);
          busy.push([d - 2.5, d + 2.5]); count('diamonds');
        }
      }
      // blue bicycle chevrons along the outer edge of the lane (roads with sidewalks)
      if (r.sidewalk > 0) {
        for (let d = 9 + R() * 4; d < r.len - 9; d += 10) {
          if (busy.some(([a, b]) => d > a - 1 && d < b + 1)) continue;
          const F = frame(r, sAt(r, dir, d), dir), [x, z] = side(F, r.laneW - 0.6);
          marks.add(x, 0.009, z, F.fx, F.fz, 0.75, 1.5, A.chevron); count('chevrons');
        }
      }
      // gratings over the gutter, on this direction's kerb side
      for (let d = 6 + R() * 8; d < r.len - 6; d += 16 + R() * 12) {
        const F = frame(r, sAt(r, dir, d), dir), [x, z] = side(F, r.hw - 0.21);
        covers.add(x, 0.006, z, F.fx, F.fz, 0.38, 0.9, A.grate); count('gratings');
      }
    }
    // manholes along the middle of the road, a hydrant and valve covers near the edges
    for (let s = 12 + R() * 14; s < r.len - 12; s += 28 + R() * 24) {
      const F = frame(r, s, 1), [x, z] = side(F, (R() - 0.5) * (narrow ? 0.6 : 1.4));
      const u = R(), cell = u < 0.62 ? A.manholeA : u < 0.86 ? A.manholeB : A.manholeC;
      covers.add(x, 0.006, z, F.fx, F.fz, 0.66, 0.66, cell); count('manholes');
    }
    if (r.len > 30 && R() < 0.55) {
      const F = frame(r, 10 + R() * (r.len - 20), 1), [x, z] = side(F, (R() < 0.5 ? 1 : -1) * (r.hw - 0.8));
      covers.add(x, 0.006, z, F.fx, F.fz, 0.72, 0.72, A.hydrant); count('hydrant covers');
    }
    for (let k = Math.floor(R() * 3); k > 0; k--) {
      const F = frame(r, 8 + R() * Math.max(1, r.len - 16), 1), [x, z] = side(F, (R() < 0.5 ? 1 : -1) * (r.laneW - 0.35));
      covers.add(x, 0.006, z, F.fx, F.fz, 0.3, 0.3, R() < 0.6 ? A.valve : A.gas); count('valve covers');
    }
    // patched and sealed asphalt
    for (let k = Math.floor(R() * 3.2); k > 0; k--) {
      const F = frame(r, 8 + R() * Math.max(1, r.len - 16), 1), [x, z] = side(F, (R() - 0.5) * 2 * Math.max(0.2, r.laneW - 1.3));
      const a = (R() - 0.5) * 0.3, c = Math.cos(a), sn = Math.sin(a);
      soft.add(x, 0.005, z, F.fx * c - F.fz * sn, F.fz * c + F.fx * sn, 1.0 + R() * 1.6, 1.2 + R() * 2.2, A.patch); count('patches');
    }
    if (r.len > 25 && R() < 0.4) {        // a trench across one lane
      const dir = R() < 0.5 ? 1 : -1, F = frame(r, 8 + R() * (r.len - 16), dir), [x, z] = side(F, r.laneW / 2);
      soft.add(x, 0.005, z, F.lx, F.lz, 0.8, r.laneW * 0.92, A.trench); count('trenches');
    }
    for (let k = 1 + Math.floor(R() * 3); k > 0; k--) {
      const F = frame(r, 6 + R() * Math.max(1, r.len - 12), 1), [x, z] = side(F, (R() - 0.5) * 2 * Math.max(0.3, r.laneW - 0.9));
      soft.add(x, 0.005, z, F.fx, F.fz, 1.4 + R(), 2.4 + R() * 1.6, A.crack); count('cracks');
    }
    // green 路側帯 (school route) on residential roads: over the gutter, out to the road edge
    if (r.kind === 'residential' && R() < 0.5) {
      for (const sgn of [1, -1]) {
        const ring = [], back = [];
        for (let s = 6; s <= r.len - 6 + 1e-6; s += 2) {
          const f = r.at(Math.min(s, r.len - 6));
          ring.push([f.x + f.nx * sgn * (r.laneW + 0.2), f.z + f.nz * sgn * (r.laneW + 0.2)]);
          back.push([f.x + f.nx * sgn * (r.hw - 0.04), f.z + f.nz * sgn * (r.hw - 0.04)]);
        }
        if (ring.length >= 2) { green.push({ outer: ring.concat(back.reverse()), holes: [] }); count('green belts'); }
      }
    }
  }

  // ---------------------------------------------------------------- junctions: coloured pavement, curve mirrors
  const armsOf = (j) => T.roads.filter((r) => r.a === j.id || r.b === j.id).map((r) => {
    const atA = r.a === j.id, f = r.at(atA ? 0 : r.len), d = atA ? 1 : -1;
    return { r, x: f.x, z: f.z, dx: f.tx * d, dz: f.tz * d };     // road end and direction away from the junction
  });
  for (const j of T.junctions) {
    const arms = armsOf(j);
    if (j.signalized && R() < 0.35) {     // red-brown pavement over the junction box (inside the gutters)
      const pts = [];
      for (const a of arms) { const w = a.r.hw - 0.45; pts.push([a.x - a.dz * w, a.z + a.dx * w], [a.x + a.dz * w, a.z - a.dx * w]); }
      if (pts.length >= 3) { red.push({ outer: hull(pts), holes: [] }); count('red junctions'); }
    }
    if (arms.length === 2) {              // a bend: curve mirror on its outer side
      const [p, q] = arms, dot = p.dx * q.dx + p.dz * q.dz;
      if (dot > Math.cos(150 * Math.PI / 180)) {
        let bx = p.dx + q.dx, bz = p.dz + q.dz; const bl = Math.hypot(bx, bz) || 1; bx /= bl; bz /= bl;   // into the bend
        const [jx, jz] = j.p;
        let edge = 0;                      // outermost road surface along -b
        for (const a of arms) for (const sg of [1, -1]) edge = Math.max(edge, -((a.x - a.dz * a.r.hw * sg - jx) * bx + (a.z + a.dx * a.r.hw * sg - jz) * bz));
        let d = edge + 0.7, x = jx - bx * d, z = jz - bz * d;
        for (let k = 0; k < 12 && (T.surfaceAt(x, z) === SURF.road || !free(x, z, 0.9)); k++) { d += 0.25; x = jx - bx * d; z = jz - bz * d; }
        if (T.surfaceAt(x, z) !== SURF.road && free(x, z, 0.6)) {
          const y0 = T.heightAt(x, z), top = y0 + 2.9;
          geos.orange.push(rod([x, y0, z], [x, top, z], 0.038, 8));
          for (const a of arms) {           // one mirror for each approach, facing it
            let mx = bx + a.dx * 0.9, mz = bz + a.dz * 0.9; const ml = Math.hypot(mx, mz) || 1; mx /= ml; mz /= ml;
            const cx = x + mx * 0.12, cz = z + mz * 0.12;
            geos.orange.push(rod([x, top - 0.35, z], [cx, top - 0.35, cz], 0.025, 6));
            geos.face.push(face(cx + mx * 0.03, top - 0.35, cz + mz * 0.03, mx, mz, 0.8, S.mirror, true, -0.1));   // tilted down
            geos.back.push(face(cx - mx * 0.01, top - 0.35, cz - mz * 0.01, -mx, -mz, 0.84, S.back, true, 0.1));
          }
          ctx.physics.addCylinder(x, z, 0.12, y0, top);
          obstacles.push([x, z]); count('curve mirrors');
        }
      }
    }
  }

  // ---------------------------------------------------------------- road ends: tactile paving, guard pipes, signs
  const signAt = (F, latList, cell, round, h = 2.35) => {
    for (const ds of [0, 1.5, -1.5, 3, -3]) {
      for (const lat of latList) {
        const x = F.x + F.lx * lat + F.fx * ds, z = F.z + F.lz * lat + F.fz * ds;
        if (T.surfaceAt(x, z) === SURF.road || !free(x, z, 0.8)) continue;
        const y0 = T.heightAt(x, z);
        geos.gray.push(rod([x, y0, z], [x, y0 + h + 0.32, z], 0.03, 8));
        geos.face.push(face(x - F.fx * 0.045, y0 + h, z - F.fz * 0.045, -F.fx, -F.fz, 0.6, cell, round));
        geos.back.push(face(x - F.fx * 0.035, y0 + h, z - F.fz * 0.035, F.fx, F.fz, 0.6, round ? S.back : S.backsq, round));
        ctx.physics.addCylinder(x, z, 0.08, y0, y0 + h + 0.3);
        obstacles.push([x, z]);
        return true;
      }
    }
    return false;
  };
  const roadside = (r) => (r.sidewalk > 0 ? [r.hw + 0.45, r.hw + 0.8] : [r.hw + 0.15, r.hw + 0.3]);
  for (const r of T.roads) {
    for (const dir of [1, -1]) {
      const ahead = sigEnd(r, dir);
      // speed limit sign after the start of the road, on the traffic side
      if (r.len > 40) {
        const F = frame(r, sAt(r, dir, 9), dir);
        if (signAt(F, roadside(r), S['s' + kmh(r)], true)) count('speed signs');
      }
      if (ahead && r.len > 30) {
        const dEnd = r.len;                // distance of the junction from this lane's start
        // 横断歩道 sign before the crosswalk
        if (R() < 0.7) {
          const F = frame(r, sAt(r, dir, dEnd - 5.6), dir);
          if (signAt(F, roadside(r), S.crosswalk, false)) count('crosswalk signs');
        }
        if (r.sidewalk > 0) {
          for (const sg of [1, -1]) {      // both sides of the road end
            // warning blocks at the crosswalk end (2 tiles deep), guiding blocks back along the sidewalk
            for (let d = dEnd - 4.05; d <= dEnd - 0.95; d += 0.3) {
              const F = frame(r, sAt(r, dir, d), dir);
              for (const k of [0, 1]) {
                const lat = sg * (r.hw + 0.38 + 0.3 * k), x = F.x + F.lx * lat, z = F.z + F.lz * lat;
                if (T.surfaceAt(x, z) === SURF.sidewalk) { tactile.add(x, SIDEWALK_Y + 0.006, z, F.fx, F.fz, 0.3, 0.3, A.dots); count('tactile tiles'); }
              }
            }
            const mid = sg * (r.hw + r.sidewalk / 2);
            for (let d = dEnd - 4.4; d > Math.max(dEnd - 16, 6); d -= 0.3) {
              const F = frame(r, sAt(r, dir, d), dir), x = F.x + F.lx * mid, z = F.z + F.lz * mid;
              if (T.surfaceAt(x, z) !== SURF.sidewalk || !free(x, z, 0.45)) continue;
              tactile.add(x, SIDEWALK_Y + 0.006, z, F.fx, F.fz, 0.3, 0.3, A.bars); count('tactile tiles');
            }
            // guard pipes along the kerb, from just behind the crosswalk
            if (R() < 0.55) {
              const len = 8 + R() * 8, lat = sg * (r.hw + 0.32);
              let prev = null;
              for (let d = dEnd - 5.2; d >= Math.max(dEnd - 5.2 - len, 4); d -= 2.0) {
                const F = frame(r, sAt(r, dir, d), dir), x = F.x + F.lx * lat, z = F.z + F.lz * lat;
                if (T.surfaceAt(x, z) !== SURF.sidewalk || !free(x, z, 0.55)) { prev = null; continue; }
                const y = T.heightAt(x, z);
                geos.white.push(rod([x, y, z], [x, y + 0.8, z], 0.03, 6));
                if (prev) {
                  for (const h of [0.42, 0.76]) geos.white.push(rod([prev[0], prev[2] + h, prev[1]], [x, y + h, z], 0.022, 5));
                  const cx = (prev[0] + x) / 2, cz = (prev[1] + z) / 2;
                  ctx.physics.addBox(cx, cz, 0.08, Math.hypot(x - prev[0], z - prev[1]), Math.atan2(x - prev[0], z - prev[1]), y, y + 0.85);
                  count('guard pipe spans');
                }
                prev = [x, z, y];
              }
            }
          }
        }
      }
      // no-parking sign halfway along some streets
      if (r.len > 60 && r.kind !== 'main' && R() < 0.3) {
        const F = frame(r, sAt(r, dir, r.len * (0.4 + R() * 0.2)), dir);
        if (signAt(F, roadside(r), S.noparking, true)) count('no-parking signs');
      }
    }
  }

  // ---------------------------------------------------------------- meshes
  const decal = (q, mat, name, sem, order) => {
    if (!q.I.length) return;
    const m = new THREE.Mesh(q.geometry(), mat);
    m.name = name; m.receiveShadow = true; m.castShadow = false; m.renderOrder = order; m.userData.semClass = sem;
    ctx.noOutline(m); ctx.noBatch(m); ctx.addStatic(m);
  };
  const map = deco.tex, alphaMap = deco.alpha;
  decal(soft, ctx.mat.decal('#fdfdfd', { map, alphaMap, paint: 0.02 }), 'road-repairs', SEM.road, -1);
  decal(covers, ctx.mat.decal('#fefefe', { map, alphaMap, alphaTest: 0.45, paint: 0.04 }), 'road-covers', SEM.road, -1);
  decal(marks, ctx.mat.decal('#ffffff', { map, alphaMap, alphaTest: 0.45, paint: 0.03 }), 'road-markings', SEM.marking, 1);
  decal(tactile, ctx.mat.decal('#fcfcfc', { map, alphaMap, alphaTest: 0.45, paint: 0.03 }), 'tactile-paving', SEM.sidewalk, 0);
  const flat = (shapes, y, mat, name) => {
    if (!shapes.length) return;
    const g = ctx.geo.mergeGeometries(shapes.map((s) => {
      const sh = new THREE.Shape(s.outer.map(([x, z]) => new THREE.Vector2(x, -z)));
      const gg = new THREE.ShapeGeometry(sh, 1); gg.rotateX(-Math.PI / 2); gg.translate(0, y, 0); return gg;
    }), false);
    const m = new THREE.Mesh(g, mat);
    m.name = name; m.receiveShadow = true; m.userData.semClass = SEM.road;
    ctx.noOutline(m); ctx.noBatch(m); ctx.addStatic(m);
  };
  flat(red, 0.0045, ctx.mat.toon('#b8664c', { polygonOffset: -1, paint: 0.06 }), 'coloured-pavement');
  flat(green, 0.006, ctx.mat.toon('#4e9a5c', { polygonOffset: -1.5, paint: 0.06 }), 'green-belt');
  const solid = (list, mat, name) => {
    if (!list.length) return;
    const g = ctx.geo.mergeGeometries(list.map((x) => (x.index ? x.toNonIndexed() : x)), false);
    list.forEach((x) => x.dispose());
    const m = new THREE.Mesh(g, mat);
    m.name = name; m.castShadow = true; m.receiveShadow = true; m.userData.semClass = SEM.pole;
    ctx.noBatch(m); ctx.addStatic(m);
  };
  solid(geos.white, ctx.mat.toon('#f1f1ec', { paint: 0.03 }), 'guard-pipes');
  solid(geos.gray, ctx.mat.toon('#a2a8ae', { paint: 0.03 }), 'sign-poles');
  solid(geos.orange, ctx.mat.toon('#ee8a2a', { paint: 0.04 }), 'mirror-poles');
  solid(geos.face, ctx.mat.toon('#ffffff', { map: sign.tex, paint: 0.02 }), 'sign-faces');
  solid(geos.back, ctx.mat.toon('#ffffff', { map: sign.tex, paint: 0.02 }), 'sign-backs');   // same material (cached)
  ctx.services.roaddeco = stats;
}
