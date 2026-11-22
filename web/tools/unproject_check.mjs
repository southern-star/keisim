// Compares web/src/unproject.js with keisim/render/rays.py on values Python wrote (tests/test_recon.py writes the
// file and runs this; node only, no browser):
//   node tools/unproject_check.mjs fixture.json
// The fixture: {rig, pose: [x, y, yaw], encode: {r, q}, cams: [{index, step, q, world, proj}],
//   stitch: {index, atlas, nearest, planar}} (images base64, uint16 little-endian, top row first).
// Passes when the codes agree, the world points differ by less than 1 mm, the projections by less than 1e-6 pixel and
// the stitched cylinder depth by at most one code. Prints one line per comparison and exits 1 on a failure.
import fs from 'node:fs';
import { RIGS, cylinderFaces } from '../src/rig.js';
import { cameraRays, decodeDepth, encodeDepth, levelFrame, makeDepthStitch, project, unproject } from '../src/unproject.js';

const fx = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const u16 = (b64) => { const b = Buffer.from(b64, 'base64'); return new Uint16Array(b.buffer, b.byteOffset, b.length / 2); };
let failed = 0;
const report = (name, ok, text) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${text}`); if (!ok) failed++; };

{ // the 16 bit code, both ways
  let bad = 0, worst = 0;
  fx.encode.r.forEach((r, i) => {
    const q = encodeDepth(r);
    if (q !== fx.encode.q[i]) bad++;
    if (q > 0) worst = Math.max(worst, Math.abs(decodeDepth(q) - fx.encode.back[i]) / fx.encode.back[i]);
  });
  report('encode', bad === 0 && worst < 1e-12, `${fx.encode.r.length} distances, ${bad} codes differ, decoded within ${worst.toExponential(1)}`);
}

const rig = RIGS[fx.rig];
for (const c of fx.cams) {
  const cam = rig[c.index], frame = levelFrame(cam, ...fx.pose);
  const rays = cameraRays(cam, c.step), q = u16(c.q);
  const r = Float64Array.from(q, decodeDepth), P = unproject(r, rays, frame);
  let worst = 0;
  for (let i = 0; i < P.length; i++) worst = Math.max(worst, Math.abs(P[i] - c.world[i]));
  report(`${cam.name} step ${c.step} world`, P.length === c.world.length && worst < 1e-3, `${q.length} pixels, largest difference ${(worst * 1000).toExponential(2)} mm`);
  let pw = 0, okBad = 0;
  c.proj.pts.forEach((p, i) => {
    const g = project(cam, frame, p);
    if (g.ok !== c.proj.ok[i]) okBad++;
    if (g.ok) pw = Math.max(pw, Math.abs(g.u - c.proj.uv[i][0]), Math.abs(g.v - c.proj.uv[i][1]), Math.abs(g.r - c.proj.r[i]));
  });
  report(`${cam.name} step ${c.step} project`, okBad === 0 && pw < 1e-6, `${c.proj.pts.length} points, largest difference ${pw.toExponential(2)} px, ${okBad} in/out differ`);
}

{ // the cylinder's depth from its faces
  const cam = rig[fx.stitch.index], cyl = cylinderFaces(cam), atlas = u16(fx.stitch.atlas);
  for (const mode of ['nearest', 'planar']) {
    const want = u16(fx.stitch[mode]), got = makeDepthStitch(cyl, mode)(atlas, new Uint16Array(cam.width * cam.height));
    let worst = 0, n = 0;
    for (let i = 0; i < want.length; i++) { const d = Math.abs(got[i] - want[i]); if (d) n++; worst = Math.max(worst, d); }
    report(`${cam.name} stitch ${mode}`, got.length === want.length && worst <= (mode === 'nearest' ? 0 : 1), `${want.length} pixels, ${n} differ, by at most ${worst} codes`);
  }
}
process.exit(failed ? 1 : 0);
