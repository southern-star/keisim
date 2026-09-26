// 地面 — one large grass plane under the whole scene (the KeiSim town is flat at y = 0).
import * as THREE from 'three';

export function build(ctx) {
  const R = ctx.rng('kv-ground');
  const SIZE = 1800, TILE = 6;
  const grass = ctx.tex.draw(512, 512, (g, w, h) => {
    g.fillStyle = '#ffffff'; g.fillRect(0, 0, w, h);
    // soft blotches (hand-painted wash) + short tuft strokes; values stay near white, the
    // material colour carries the hue
    for (let i = 0; i < 70; i++) {
      const x = R() * w, y = R() * h, r = 18 + R() * 60, a = 0.05 + R() * 0.07;
      g.fillStyle = R() < 0.5 ? `rgba(120,150,90,${a})` : `rgba(255,255,230,${a})`;
      g.beginPath(); g.ellipse(x, y, r, r * (0.5 + R() * 0.5), R() * 3, 0, 7); g.fill();
      for (const dx of [-w, w]) { g.beginPath(); g.ellipse(x + dx, y, r, r * 0.7, 0, 0, 7); g.fill(); }
    }
    g.lineCap = 'round';
    for (let i = 0; i < 900; i++) {
      const x = R() * w, y = R() * h, l = 3 + R() * 5;
      g.strokeStyle = R() < 0.6 ? 'rgba(90,125,70,0.22)' : 'rgba(255,255,235,0.25)';
      g.lineWidth = 1 + R();
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + (R() - 0.5) * 2, y - l); g.stroke();
    }
  }, { key: 'kv_grass', repeat: [SIZE / TILE, SIZE / TILE] });
  const mat = ctx.mat.toon('#a8c98a', { map: grass, paint: 0.1 });
  // 25 m cells, not one quad: near-plane clipping of two 1.8 km triangles loses enough depth
  // precision (seen on SwiftShader) for the grass 3 cm below to show through the road.
  const geo = new THREE.PlaneGeometry(SIZE, SIZE, SIZE / 25, SIZE / 25);
  geo.rotateX(-Math.PI / 2);
  const m = new THREE.Mesh(geo, mat);
  m.position.y = -0.03;
  m.receiveShadow = true; m.castShadow = false;
  m.name = 'ground';
  ctx.noBatch(m);
  ctx.addStatic(m);
}
