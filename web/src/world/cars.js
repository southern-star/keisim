// 車 (detailed) — Japanese street cars in KeiView's cel style, for the views people look at (src/world/actors.js
// shows them near the camera when ctx.carDetail is on; the KeiPilot model frame keeps the plain cars it was
// trained on). Every body is a side profile extruded across the width with rounded edges: a lower body up to the
// belt line and a greenhouse of dark tinted glass. The type follows the size KeiSim gives a car (and, for kei
// trucks, its colour): kei tall wagon, kei hatch, kei truck, compact, sedan, SUV, minivan, small truck. Models face
// local +Z with their origin on the ground at the centre, like actors.js.
import * as THREE from 'three';

/** Side profile [z (forward), y (up)] extruded across x in [-w/2, w/2], edges rounded by r. The bevel grows the
 *  outline by its size, so the profile is first shrunk by that much about its bounding box (keeps L and H). */
function profileBody(pts, w, r) {
  const bs = r * 0.6;
  const zs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [z0, z1, y0, y1] = [Math.min(...zs), Math.max(...zs), Math.min(...ys), Math.max(...ys)];
  const cz = (z0 + z1) / 2, cy = (y0 + y1) / 2;
  const kz = Math.max(0.5, (z1 - z0 - 2 * bs) / Math.max(z1 - z0, 1e-6)), ky = Math.max(0.5, (y1 - y0 - 2 * bs) / Math.max(y1 - y0, 1e-6));
  const shape = new THREE.Shape(pts.map(([z, y]) => new THREE.Vector2(cz + (z - cz) * kz, cy + (y - cy) * ky)));
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.01, w - 2 * r), bevelEnabled: r > 0, bevelThickness: r, bevelSize: bs,
    bevelSegments: 3, curveSegments: 4,
  });
  g.rotateY(-Math.PI / 2);                    // shape x -> +z (forward), extrusion -> -x
  g.translate((w - 2 * r) / 2, 0, 0);          // centre the width
  return g;
}

/** A thin slab through 4 corner points (counter-clockwise seen from its outside), `thick` deep towards the outside. */
function panel(c, thick = 0.012) {
  const [a, b, d, e] = c.map((p) => new THREE.Vector3(...p));
  const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(e, a)).normalize().multiplyScalar(thick);
  const verts = [a, b, d, e, a.clone().add(n), b.clone().add(n), d.clone().add(n), e.clone().add(n)];
  const idx = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 5, 4, 0, 1, 5, 1, 6, 5, 1, 2, 6, 2, 7, 6, 2, 3, 7, 3, 4, 7, 3, 0, 4];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts.flatMap((v) => [v.x, v.y, v.z]), 3));
  g.setIndex(idx);
  const flat = g.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}

/** Bottom edge of a body profile around a wheel arch of radius ra over a wheel (radius wr) at zc, rear to front;
 *  a bottom edge above the axle (gc > wr) starts the arc where it meets it. */
function arch(zc, wr, ra, gc) {
  const a0 = gc < wr ? 0 : Math.asin(Math.min(1, (gc - wr) / ra));
  const out = gc < wr ? [[zc - ra, gc]] : [];
  for (let k = 0; k <= 8; k++) {
    const th = Math.PI - a0 - (Math.PI - 2 * a0) * k / 8;
    out.push([zc + ra * Math.cos(th), wr + ra * Math.sin(th)]);
  }
  if (gc < wr) out.push([zc + ra, gc]);
  return out;
}

/** Stable pseudo-random number in [0, 1) from a car's size (KeiSim draws the size at random for every car). */
const variant = (L, H) => ((Math.round(L * 1000) * 7919 + Math.round(H * 1000) * 104729) % 1000) / 1000;

/** Body type from KeiSim's size; a kei car is sometimes a kei truck, most often when it is white or silver. */
export function carType(kind, L, W, H, rgb = [0, 0, 0]) {
  if (kind === 1) return 'truck';
  if (L < 3.6) {
    const pale = Math.min(...rgb) > 150 && Math.max(...rgb) - Math.min(...rgb) < 40;
    if (variant(L, H) < (pale ? 0.5 : 0.1)) return 'keitruck';
    return H > 1.68 ? 'keiwagon' : 'keihatch';
  }
  if (H > 1.68) return H < 1.86 ? 'suv' : 'minivan';
  return L < 4.35 ? 'compact' : 'sedan';
}

/** Key points of a body type for a car of L x H: lower body profile (to the belt line), greenhouse [windscreen base,
 *  windscreen top, rear glass top, rear glass base], ground clearance, wheel radius, axles, door lengths [front, rear],
 *  and the styling: sliding rear doors, front quarter glass, rear hatch, tail lamps, wheels, SUV cladding and rails. */
function design(type, L, H) {
  const hl = L / 2;
  switch (type) {
    case 'keiwagon': {                          // N-BOX style: tall box, short hood, upright rear, sliding rear doors
      const gc = 0.17, belt = Math.min(0.98, H * 0.55);
      return {
        gc, wr: 0.28, axF: hl - 0.62, axR: -hl + 0.55, doors: [0.85, 0.75],
        slide: true, quarterF: true, hatch: true, tail: 'tall', rim: 'cap',
        body: [[-hl, gc + 0.12], [-hl + 0.05, gc], [hl - 0.08, gc], [hl, gc + 0.18], [hl - 0.02, belt - 0.2], [hl - 0.25, belt - 0.05],
          [hl - 0.55, belt], [-hl + 0.04, belt], [-hl, belt - 0.15]],
        green: [[hl - 0.55, belt], [hl - 0.95, H - 0.04], [-hl + 0.1, H - 0.04], [-hl + 0.02, belt]],
      };
    }
    case 'keihatch': {                          // Wagon R / Alto style: short hood, steep hatch
      const gc = 0.17, belt = H * 0.58;
      return {
        gc, wr: 0.27, axF: hl - 0.6, axR: -hl + 0.55, doors: [0.9, 0.75], hatch: true, tail: 'wrap', rim: 'cap',
        body: [[-hl, gc + 0.1], [-hl + 0.05, gc], [hl - 0.1, gc], [hl, gc + 0.16], [hl - 0.04, belt - 0.15], [hl - 0.5, belt - 0.02],
          [hl - 0.8, belt], [-hl + 0.06, belt], [-hl, belt - 0.12]],
        green: [[hl - 0.8, belt], [hl - 1.4, H - 0.03], [-hl + 0.3, H - 0.03], [-hl + 0.06, belt]],
      };
    }
    case 'compact': {                           // Fit / Yaris style hatchback
      const gc = 0.17, belt = H * 0.6;
      return {
        gc, wr: 0.3, axF: hl - 0.75, axR: -hl + 0.65, doors: [1.0, 0.85], hatch: true, tail: 'wrap', rim: 'alloy',
        body: [[-hl, gc + 0.12], [-hl + 0.06, gc], [hl - 0.12, gc], [hl, gc + 0.2], [hl - 0.05, belt - 0.18], [hl - 0.6, belt - 0.03],
          [hl - 1.0, belt], [-hl + 0.1, belt], [-hl, belt - 0.12]],
        green: [[hl - 1.0, belt], [hl - 1.8, H - 0.03], [-hl + 0.45, H - 0.03], [-hl + 0.12, belt]],
      };
    }
    case 'sedan': {                             // Corolla style: hood, raked glass, short trunk deck
      const gc = 0.17, belt = H * 0.62;
      return {
        gc, wr: 0.31, axF: hl - 0.85, axR: -hl + 0.95, doors: [1.05, 0.9], tail: 'wrap', rim: 'alloy',
        body: [[-hl, gc + 0.14], [-hl + 0.08, gc], [hl - 0.12, gc], [hl, gc + 0.2], [hl - 0.06, belt - 0.16], [hl - 0.75, belt - 0.02],
          [hl - 1.15, belt], [-hl + 1.0, belt], [-hl + 0.15, belt + 0.03], [-hl, belt - 0.1]],
        green: [[hl - 1.15, belt], [hl - 2.0, H - 0.03], [-hl + 1.6, H - 0.03], [-hl + 0.95, belt]],
      };
    }
    case 'suv': {                               // RAV4 / CX-5 style: high ground clearance, big wheels, black cladding
      const gc = 0.22, belt = H * 0.56;
      return {
        gc, wr: 0.36, axF: hl - 0.9, axR: -hl + 0.98, doors: [1.1, 0.95], hatch: true, tail: 'slim', rim: 'alloy',
        clad: true, rails: true,
        body: [[-hl, gc + 0.16], [-hl + 0.08, gc], [hl - 0.14, gc], [hl, gc + 0.24], [hl - 0.03, belt - 0.2], [hl - 0.55, belt - 0.04],
          [hl - 1.1, belt], [-hl + 0.1, belt], [-hl, belt - 0.14]],
        green: [[hl - 1.1, belt], [hl - 1.95, H - 0.08], [-hl + 0.35, H - 0.08], [-hl + 0.1, belt]],
      };
    }
    default: {                                  // minivan (Serena style): short sloped hood, long roof, upright rear
      const gc = 0.17, belt = H * 0.52;
      return {
        gc, wr: 0.32, axF: hl - 0.85, axR: -hl + 0.8, doors: [1.0, 0.9],
        slide: true, quarterF: true, hatch: true, tail: 'tall', rim: 'cap',
        body: [[-hl, gc + 0.14], [-hl + 0.06, gc], [hl - 0.12, gc], [hl, gc + 0.22], [hl - 0.05, belt - 0.22], [hl - 0.5, belt - 0.05],
          [hl - 0.95, belt], [-hl + 0.05, belt], [-hl, belt - 0.15]],
        green: [[hl - 0.95, belt], [hl - 1.65, H - 0.04], [-hl + 0.12, H - 0.04], [-hl + 0.03, belt]],
      };
    }
  }
}

/** Geometry helpers that collect [geometry, materialKey] parts. */
function kit(parts) {
  const add = (g, m) => { parts.push([g, m]); return g; };
  const box = (w, h, d, m, x, y, z, ry = 0) => {
    const g = new THREE.BoxGeometry(w, h, d);
    if (ry) g.rotateY(ry);
    g.translate(x, y, z);
    return add(g, m);
  };
  /** Disc of radius r and thickness t with its axis across the car (x). */
  const disc = (r, t, x, y, z, m, seg = 16) => {
    const g = new THREE.CylinderGeometry(r, r, t, seg); g.rotateZ(Math.PI / 2); g.translate(x, y, z); return add(g, m);
  };
  /** Flat ring (or arc from the front round over the top) facing outwards on side s. */
  const ring = (s, r0, r1, x, y, z, m, arc = Math.PI * 2, seg = 16) => {
    const g = new THREE.RingGeometry(r0, r1, seg, 1, 0, arc); g.rotateY(s * Math.PI / 2); g.translate(x, y, z); return add(g, m);
  };
  /** Tyre and wheel on side s; style 'alloy' (five dark openings), 'cap' (hub cap ring) or 'steel' (trucks). */
  function wheel(s, x, z, r, w, style) {
    disc(r, w, x, r, z, 'tire', 20);
    const xf = x + s * (w / 2 + 0.006);
    if (style === 'steel') {
      disc(r * 0.6, 0.012, xf, r, z, 'hub', 14);
      disc(r * 0.24, 0.02, xf + s * 0.008, r, z, 'trim', 8);
      return;
    }
    disc(r * 0.66, 0.012, xf, r, z, 'hub', 16);
    if (style === 'cap') ring(s, r * 0.4, r * 0.46, xf + s * 0.008, r, z, 'trim');
    else {
      for (let k = 0; k < 5; k++) {
        const th = k * Math.PI * 0.4 + 0.3;
        const g = new THREE.BoxGeometry(0.008, r * 0.3, r * 0.17); g.rotateX(th);
        g.translate(xf + s * 0.006, r + r * 0.42 * Math.cos(th), z + r * 0.42 * Math.sin(th)); add(g, 'trim');
      }
    }
    disc(r * 0.13, 0.016, xf + s * 0.006, r, z, 'trim', 8);
  }
  /** Straight strip of width wdt from (z0, y0) to (z1, y1) in the plane x, t thick across x: pillars, seams, mouldings. */
  function strip(x, z0, y0, z1, y1, wdt, m, t = 0.02) {
    const dz = z1 - z0, dy = y1 - y0;
    const g = new THREE.BoxGeometry(t, Math.hypot(dz, dy), wdt);
    g.rotateX(Math.atan2(dz, dy));
    g.translate(x, (y0 + y1) / 2, (z0 + z1) / 2);
    return add(g, m);
  }
  /** Box lying on a sloped face [z0, y0, z1, y1, out] (base at the belt, top at the roof; out 1: windscreen, -1: rear
   *  glass): centred at x and at fraction f up the face, wx across, h along it (default: all of it), lifted off it and
   *  turned by tilt within it. Not added (the caller picks the list). */
  function onFace(face, x, wx, f = 0.5, h = null, lift = 0.004, tilt = 0) {
    const [z0, y0, z1, y1, out] = face;
    const dz = z1 - z0, dy = y1 - y0, l = Math.hypot(dz, dy);
    const g = new THREE.BoxGeometry(wx, h ?? l, 0.02);
    if (tilt) g.rotateZ(tilt);
    g.rotateX(Math.atan2(dz, dy));
    g.translate(x, y0 + dy * f - out * dz / l * lift, z0 + dz * f + out * dy / l * lift);
    return g;
  }
  return { add, box, disc, ring, wheel, strip, onFace };
}

/**
 * Detailed car: returns {type, parts: [[geometry, materialKey], ...], tails: [geometry...]} with materialKey one of
 * body, glass, interior, trim, tire, hub, head, plateW, plateY, cargo (tails are separate: they switch material when
 * braking). Geometries are in the car frame; actors.js merges them per material.
 */
export function buildCar(kind, L, W, H, rgb) {
  const type = carType(kind, L, W, H, rgb);
  const parts = [], tails = [];
  const k = kit(parts);
  if (type === 'truck') buildTruck(L, W, H, k, tails);
  else if (type === 'keitruck') buildKeiTruck(L, W, H, k, tails);
  else buildBody(design(type, L, H), L, W, k, tails);
  return { type, parts, tails };
}

function buildBody(d, L, W, k, tails) {
  const hl = L / 2, hw = W / 2, gw = W - 0.16, ra = d.wr + 0.05;
  const [[wsBot, belt], [wsTop, yTop], [rearTop], [rearBot]] = d.green;
  // lower body with the wheel arches cut out of its bottom edge (profile points 1 -> 2 run rear to front)
  k.add(profileBody([d.body[0], d.body[1], ...arch(d.axR, d.wr, ra, d.gc), ...arch(d.axF, d.wr, ra, d.gc), ...d.body.slice(2)], W, 0.1), 'body');
  // greenhouse, narrower than the body (tumblehome): tinted glass over a dark interior, under a body-coloured roof
  k.add(profileBody(d.green, gw, 0.07), 'glass');
  k.add(profileBody(d.green.map(([z, y]) => [z, y - 0.02]), gw - 0.1, 0.04), 'interior');
  k.add(profileBody([[wsTop + 0.02, yTop - 0.06], [wsTop, yTop + 0.015], [rearTop, yTop + 0.015], [rearTop + 0.02, yTop - 0.06]], gw + 0.01, 0.02), 'body');

  // window frames: body-coloured A and C pillars on the sides and on the edges of the windscreen and rear glass,
  // black B pillar and door sashes, a black moulding along the bottom of the side glass
  const topAt = (z) => (z > wsTop ? belt + (yTop - belt) * (wsBot - z) / (wsBot - wsTop)
    : z < rearTop ? belt + (yTop - belt) * (z - rearBot) / (rearTop - rearBot) : yTop);
  const front = [wsBot, belt, wsTop, yTop, 1], rear = [rearBot, belt, rearTop, yTop, -1];
  const zf = d.axF - ra - 0.03, zb = zf - d.doors[0], zr = zb - d.doors[1];      // door edges: front, B pillar, rear
  for (const s of [-1, 1]) {
    const xg = s * (gw / 2 + 0.006);
    k.strip(xg, wsBot, belt, wsTop, yTop, 0.09, 'body');
    k.strip(xg, rearBot, belt, rearTop, yTop, 0.12, 'body');
    k.strip(xg, zb, belt, zb, topAt(zb), 0.08, 'trim');
    if (zr > rearBot + 0.12) k.strip(xg, zr, belt, zr, topAt(zr), 0.06, 'trim');
    if (d.quarterF) k.strip(xg, zf, belt, zf, topAt(zf), 0.05, 'trim');
    k.strip(s * (gw / 2 + 0.01), rearBot + 0.04, belt + 0.012, wsBot - 0.04, belt + 0.012, 0.024, 'trim', 0.014);
    k.add(k.onFace(front, s * (gw / 2 - 0.045), 0.09), 'body');
    k.add(k.onFace(rear, s * (gw / 2 - 0.07), 0.14), 'body');
  }

  // sides: door seams, handles, mirrors, wheels with a dark fender lining (SUV: wide cladding along the sill too)
  const archTop = (z, zc) => (Math.abs(z - zc) < ra ? d.wr + Math.sqrt(ra * ra - (z - zc) ** 2) : d.gc);
  for (const s of [-1, 1]) {
    const xs = s * (hw + 0.002);
    for (const z of [zf, zb, zr]) k.strip(xs, z, Math.max(d.gc + 0.08, archTop(z, d.axR) + 0.05), z, belt - 0.05, 0.014, 'trim', 0.006);
    k.box(0.012, 0.03, 0.15, 'trim', s * (hw + 0.006), belt - 0.1, zb + 0.16);
    k.box(0.012, 0.03, 0.15, 'trim', s * (hw + 0.006), belt - 0.1, d.slide ? zb - 0.14 : zr + 0.16);
    if (d.slide) k.strip(s * (hw + 0.004), -hl + 0.15, belt - 0.07, zr - 0.02, belt - 0.07, 0.02, 'trim', 0.01);   // door rail
    const zm = d.quarterF ? zf - 0.1 : wsBot - 0.06;
    k.box(0.06, 0.09, 0.16, 'body', s * (hw + 0.05), belt + 0.07, zm);
    k.box(0.1, 0.035, 0.07, 'trim', s * (hw - 0.01), belt + 0.03, zm);
    for (const z of [d.axF, d.axR]) {
      k.wheel(s, s * (hw - 0.16), z, d.wr, 0.2, d.rim);
      k.ring(s, ra - 0.005, ra + (d.clad ? 0.085 : 0.035), s * (hw + 0.003), d.wr, z, 'trim', Math.PI, 14);
    }
    if (d.clad) k.box(0.012, 0.12, d.axF - d.axR - 2 * ra, 'trim', s * (hw + 0.004), d.gc + 0.09, (d.axF + d.axR) / 2);
  }
  if (d.rails) {                                                      // roof rails
    const len = wsTop - rearTop - 0.2, zc = (wsTop + rearTop) / 2;
    for (const s of [-1, 1]) {
      k.box(0.04, 0.035, len, 'trim', s * (gw / 2 - 0.1), yTop + 0.05, zc);
      for (const e of [-1, 1]) k.box(0.05, 0.04, 0.08, 'trim', s * (gw / 2 - 0.1), yTop + 0.03, zc + e * (len / 2 - 0.04));
    }
  }

  // front: grille, headlamps, lower intake, plate
  const yHood = d.body[4][1];
  k.box(W * 0.42, 0.11, 0.05, 'trim', 0, yHood - 0.09, hl - 0.02);
  k.box(W * 0.5, 0.09, 0.05, 'trim', 0, d.gc + 0.22, hl - 0.01);
  if (d.clad) k.box(W - 0.2, 0.12, 0.06, 'trim', 0, d.gc + 0.1, hl - 0.06);
  for (const s of [-1, 1]) {
    const head = new THREE.BoxGeometry(0.34, 0.1, 0.06); head.rotateY(s * 0.25); head.translate(s * (hw - 0.26), yHood - 0.06, hl - 0.03);
    k.add(head, 'head');
  }
  const plate = L < 3.6 ? 'plateY' : 'plateW';
  k.box(0.34, 0.17, 0.02, plate, 0, d.gc + 0.26, hl + 0.012);
  // rear: tail lamps wrapping round the corners, high-mount stop lamp, valance, plate; hatchbacks a spoiler and wiper
  for (const s of [-1, 1]) {
    const [tw, th, tx, ty] = d.tail === 'tall' ? [0.13, 0.36, hw - 0.1, belt - 0.2]
      : d.tail === 'slim' ? [0.44, 0.085, hw - 0.23, belt - 0.07] : [0.32, 0.13, hw - 0.17, belt - 0.11];
    const g = new THREE.BoxGeometry(tw, th, 0.12); g.translate(s * tx, ty, -hl + 0.045); tails.push(g);
  }
  const lr = Math.hypot(rearTop - rearBot, yTop - belt);
  tails.push(k.onFace(rear, 0, 0.3, 1 - 0.07 / lr, 0.04, 0.008));
  k.box(W - 0.3, 0.06, 0.05, 'trim', 0, d.gc + 0.05, -hl + 0.02);
  k.box(W - 0.36, 0.012, 0.01, 'trim', 0, d.gc + 0.43, -hl - 0.003);                    // bumper / hatch seam
  if (d.clad) k.box(W - 0.2, 0.12, 0.06, 'trim', 0, d.gc + 0.1, -hl + 0.04);
  k.box(0.34, 0.17, 0.02, plate, 0, d.gc + 0.3, -hl - 0.012);
  if (d.hatch) {
    k.box(gw - 0.06, 0.045, 0.16, 'body', 0, yTop - 0.005, rearTop - 0.05);
    k.add(k.onFace(rear, -0.06, 0.42, 0.14, 0.018, 0.01, 0.12), 'trim');
  }
}

/** 軽トラ (Carry / Hijet style): cab over the front wheels, flat bed with drop sides, guard frame behind the cab. */
function buildKeiTruck(L, W, H, k, tails) {
  const hl = L / 2, hw = W / 2, wr = 0.27, ra = 0.31, gw = W - 0.12;
  const axF = hl - 0.6, axR = axF - 1.9;
  const cab0 = hl - 1.32, belt = 1.02, yTop = H - 0.03, fy = 0.64;
  // cab: lower body cut round the front wheel, glass above the belt, body-coloured roof and pillars
  k.add(profileBody([[cab0, 0.32], ...arch(axF, wr, ra, 0.32), [hl - 0.04, 0.32], [hl, 0.42], [hl, belt - 0.07], [hl - 0.08, belt],
    [cab0, belt]], W, 0.08), 'body');
  const green = [[hl - 0.08, belt], [hl - 0.3, yTop], [cab0 + 0.05, yTop], [cab0, belt]];
  k.add(profileBody(green, gw, 0.06), 'glass');
  k.add(profileBody(green.map(([z, y]) => [z, y - 0.02]), gw - 0.1, 0.04), 'interior');
  k.add(profileBody([[hl - 0.28, yTop - 0.05], [hl - 0.3, yTop + 0.015], [cab0 + 0.05, yTop + 0.015], [cab0 + 0.06, yTop - 0.05]], gw + 0.01, 0.02), 'body');
  const front = [hl - 0.08, belt, hl - 0.3, yTop, 1], back = [cab0, belt, cab0 + 0.05, yTop, -1];
  for (const s of [-1, 1]) {
    const xg = s * (gw / 2 + 0.006), xs = s * (hw + 0.002);
    k.strip(xg, hl - 0.08, belt, hl - 0.3, yTop, 0.08, 'body');
    k.strip(xg, cab0, belt, cab0 + 0.05, yTop, 0.22, 'body');
    k.strip(s * (gw / 2 + 0.01), cab0 + 0.1, belt + 0.012, hl - 0.12, belt + 0.012, 0.024, 'trim', 0.014);
    k.add(k.onFace(front, s * (gw / 2 - 0.045), 0.09), 'body');
    k.add(k.onFace(back, s * (gw / 2 - 0.13), 0.26), 'body');
    k.strip(xs, hl - 0.22, 0.62, hl - 0.22, belt - 0.04, 0.014, 'trim', 0.006);           // door
    k.strip(xs, cab0 + 0.22, 0.4, cab0 + 0.22, belt - 0.04, 0.014, 'trim', 0.006);
    k.box(0.012, 0.03, 0.13, 'trim', s * (hw + 0.006), belt - 0.1, cab0 + 0.34);
    k.box(0.05, 0.15, 0.1, 'trim', s * (hw + 0.09), belt + 0.12, hl - 0.2);              // mirror on a stalk
    k.box(0.1, 0.02, 0.02, 'trim', s * (hw + 0.035), belt + 0.06, hl - 0.2);
    for (const z of [axF, axR]) k.wheel(s, s * (hw - 0.13), z, wr, 0.17, 'steel');
    k.ring(s, ra - 0.005, ra + 0.03, s * (hw + 0.003), wr, axF, 'trim', Math.PI, 14);
    k.box(0.22, 0.2, 0.01, 'trim', s * (hw - 0.15), 0.36, axR - wr - 0.12);               // mud flap
  }
  // bed: floor and drop sides in body colour, chassis below, guard frame (鳥居) behind the cab
  const bed1 = cab0 - 0.06, bedL = bed1 + hl, bz = (bed1 - hl) / 2;
  k.box(W - 0.02, 0.05, bedL, 'body', 0, fy, bz);
  for (const s of [-1, 1]) k.box(0.035, 0.3, bedL, 'body', s * (hw - 0.0175), fy + 0.175, bz);
  k.box(W, 0.3, 0.035, 'body', 0, fy + 0.175, -hl + 0.0175);
  k.box(W, 0.3, 0.035, 'body', 0, fy + 0.175, bed1 - 0.0175);
  for (const s of [-1, 1]) k.box(0.05, 0.08, 0.03, 'trim', s * (hw - 0.07), fy + 0.25, -hl - 0.005);     // gate latches
  k.box(W * 0.62, 0.14, bedL + 0.3, 'trim', 0, 0.5, bz + 0.15);
  const gz = bed1 - 0.03, gTop = H + 0.04;
  for (const s of [-1, 1]) k.box(0.05, gTop - fy, 0.05, 'trim', s * (hw - 0.05), (gTop + fy) / 2, gz);
  k.box(W - 0.04, 0.05, 0.07, 'trim', 0, gTop - 0.025, gz);
  for (let i = 1; i <= 3; i++) k.box(W - 0.14, 0.02, 0.02, 'trim', 0, fy + 0.3 + i * (gTop - fy - 0.3) / 4, gz);
  // front and rear
  k.box(W - 0.04, 0.13, 0.08, 'trim', 0, 0.4, hl - 0.01);
  k.box(0.5, 0.05, 0.02, 'trim', 0, 0.7, hl + 0.006);
  for (const s of [-1, 1]) k.box(0.24, 0.15, 0.04, 'head', s * (hw - 0.2), 0.76, hl + 0.005);
  k.box(0.34, 0.17, 0.02, 'plateY', 0, 0.43, hl + 0.045);
  k.box(W - 0.06, 0.09, 0.06, 'trim', 0, 0.44, -hl + 0.05);
  for (const s of [-1, 1]) { const g = new THREE.BoxGeometry(0.2, 0.12, 0.05); g.translate(s * (hw - 0.16), 0.5, -hl + 0.02); tails.push(g); }
  k.box(0.34, 0.17, 0.02, 'plateY', 0, 0.5, -hl + 0.01);
}

/** Small truck (2 t, box body): cab, chassis, cargo box with its rear doors. */
function buildTruck(L, W, H, k, tails) {
  const hl = L / 2, hw = W / 2;
  const cabL = Math.min(1.9, L * 0.3), cabH = Math.min(H * 0.82, 2.45), y0 = 0.45;
  const zc = hl - cabL / 2;
  k.add(profileBody([[hl - cabL, y0], [hl, y0], [hl, cabH - 0.66], [hl - 0.22, cabH - 0.05], [hl - 0.3, cabH], [hl - cabL, cabH]], W, 0.08), 'body');
  k.add(panel([[-(hw - 0.08), cabH - 0.6, hl - 0.018], [hw - 0.08, cabH - 0.6, hl - 0.018], [hw - 0.1, cabH - 0.1, hl - 0.198], [-(hw - 0.1), cabH - 0.1, hl - 0.198]], 0.01), 'glass');
  for (const s of [-1, 1]) {
    k.box(0.012, 0.42, cabL * 0.5, 'glass', s * (hw + 0.004), cabH - 0.42, zc + 0.05);
    k.strip(s * (hw + 0.002), zc - cabL * 0.3, 0.55, zc - cabL * 0.3, cabH - 0.15, 0.014, 'trim', 0.006);
    k.box(0.012, 0.03, 0.14, 'trim', s * (hw + 0.006), cabH - 0.75, zc - cabL * 0.3 + 0.14);
    k.box(0.08, 0.22, 0.05, 'trim', s * (hw + 0.12), cabH - 0.55, hl - 0.3);           // mirrors
  }
  k.box(W * 0.94, 0.25, L * 0.96, 'trim', 0, 0.42, 0);
  k.box(W * 1.01, H - 0.6, L - cabL - 0.12, 'cargo', 0, 0.6 + (H - 0.6) / 2, -(cabL + 0.12) / 2);
  k.box(0.012, H - 0.75, 0.01, 'trim', 0, 0.6 + (H - 0.6) / 2, -hl - 0.005);              // rear doors and lock bars
  for (const s of [-1, 1]) k.box(0.022, H - 0.9, 0.025, 'hub', s * 0.28, 0.6 + (H - 0.6) / 2, -hl - 0.012);
  for (const z of [hl - cabL * 0.55, -hl + 1.1, -hl + 2.0]) for (const s of [-1, 1]) k.wheel(s, s * (hw - 0.2), z, 0.42, 0.26, 'steel');
  for (const s of [-1, 1]) {
    k.box(0.26, 0.14, 0.04, 'head', s * (hw - 0.25), 0.75, hl + 0.01);
    const g = new THREE.BoxGeometry(0.22, 0.14, 0.04); g.translate(s * (hw - 0.2), 0.62, -hl - 0.01); tails.push(g);
  }
  k.box(W - 0.1, 0.18, 0.1, 'trim', 0, 0.55, hl + 0.02);
  k.box(W * 0.5, 0.2, 0.03, 'trim', 0, 0.98, hl + 0.005);                                  // grille
  k.box(W - 0.2, 0.1, 0.08, 'trim', 0, 0.45, -hl + 0.1);                                   // underrun bar
  k.box(0.33, 0.16, 0.02, 'plateW', 0, 0.5, -hl - 0.02);
}

/** Car window glass: dark tint that shows the sky where the pane faces up (windscreens, rear glass) and at grazing
 *  angles, with the diagonal highlight streaks of KeiView's window glass. Cached in M.cache as a 'glass|' material,
 *  like the vendor glass, so ego mode labels it. */
export function carGlass(M) {
  const key = 'glass|car';
  if (M.cache.has(key)) return M.cache.get(key);
  const m = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uTint: { value: M.color('#1b2430') }, uSkyTop: { value: M.color(M.palette.skyMid) }, uSkyLow: { value: M.color('#e9eef2') },
      uOpacity: { value: 0.8 },
    }]),
    vertexShader: /* glsl */`
      #include <common>
      #include <fog_pars_vertex>
      varying vec3 vW; varying vec3 vN;
      void main(){
        vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz;
        vN = normalize(mat3(modelMatrix) * normal);
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <common>
      #include <fog_pars_fragment>
      uniform vec3 uTint; uniform vec3 uSkyTop; uniform vec3 uSkyLow; uniform float uOpacity;
      varying vec3 vW; varying vec3 vN;
      void main(){
        vec3 V = normalize(cameraPosition - vW);
        vec3 N = normalize(vN); if (dot(N, V) < 0.0) N = -N;
        float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
        vec3 R = reflect(-V, N);
        vec3 sky = mix(uSkyLow, uSkyTop, smoothstep(-0.1, 0.6, R.y));
        float k = 0.06 + 0.5 * fres + 0.5 * smoothstep(-0.05, 0.5, R.y);
        vec3 col = mix(uTint, sky, clamp(k, 0.0, 0.85));
        float s = dot(vW, normalize(vec3(0.62, 0.78, 0.62))) * 0.9;
        float band = abs(fract(s * 0.5) - 0.5);
        float streak = smoothstep(0.035, 0.0, abs(band - 0.18)) * 0.9 + smoothstep(0.012, 0.0, abs(band - 0.25)) * 0.6;
        col += vec3(1.0) * streak * 0.45;
        gl_FragColor = vec4(col, clamp(uOpacity + fres * 0.3 + streak * 0.3, 0.0, 1.0));
        #include <fog_fragment>
      }`,
    transparent: true, depthWrite: false, fog: true,
  });
  M.cache.set(key, m);
  return m;
}
