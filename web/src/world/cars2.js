// 車 (redesign) — the detailed cars of ?cardetail=2: lofted bodies (src/world/carbody.js) in place of cars.js'
// extruded side profiles, with faces picked per car (src/world/carface.js). Same inputs, same box (L x W x H), same
// body types by KeiSim's sizes (carType of cars.js), same material keys, so src/world/actors.js merges and shows them
// exactly like the cars of ?cardetail=1.
// Stage P1 of CarRedesign_Handoff.md: all eight body types are lofted (the trucks: their cabs; bed and box are boxes).
import * as THREE from 'three';
import { carType } from './cars.js';
import { mesher, shell, surface, patch, planEdge, lin, roundPoly, fillStations, clamp } from './carbody.js';
import { carHash, roll, chooseFace, frontFace, rearFace, cabFace, truckTails, plateAt } from './carface.js';

/**
 * Side profiles. The six cars: the table of cars.js design() (kept in step by hand: cars.js stays as it is while both
 * exist). body: [rear chamfer, rear bottom, front bottom, nose tip, bonnet edge, ..., rear edge of the deck, rear
 * chamfer top]; green: [windscreen base, windscreen top, rear glass top, rear glass base]. The two trucks: their cabs
 * in the same form (cab: true), from the measures of cars.js buildKeiTruck / buildTruck.
 */
function design(type, L, H) {
  const hl = L / 2;
  const car = (o) => {
    const ra = o.wr + 0.05, zf = o.zf ?? o.axF - ra - 0.03, zb = zf - o.doors[0], zr = zb - o.doors[1];   // zf: the front door's front edge
    return { ...o, ra, arches: [o.axF, o.axR], wheels: [o.axF, o.axR], tw: 0.2, tx: 0.165, zf, zb, zr, seams: [zf, zb, zr],
      handles: [zb + 0.1, o.slide ? zb - 0.25 : zr + 0.1] };
  };
  switch (type) {
    case 'keiwagon': {                          // N-BOX style: tall box, short hood, upright rear, sliding rear doors
      const gc = 0.17, belt = Math.min(0.98, H * 0.55);
      return car({
        gc, wr: 0.28, axF: hl - 0.62, axR: -hl + 0.55, doors: [0.85, 0.75],
        slide: true, quarterF: true, hatch: true, rim: 'cap', gh: 0.11,
        body: [[-hl, gc + 0.12], [-hl + 0.05, gc], [hl - 0.08, gc], [hl, gc + 0.18], [hl - 0.02, belt - 0.2], [hl - 0.25, belt - 0.05],
          [hl - 0.55, belt], [-hl + 0.04, belt], [-hl, belt - 0.15]],
        green: [[hl - 0.55, belt], [hl - 0.95, H - 0.04], [-hl + 0.1, H - 0.04], [-hl + 0.02, belt]],
      });
    }
    case 'keihatch': {                          // Wagon R / Alto style: short hood, steep hatch
      const gc = 0.17, belt = H * 0.58;
      return car({
        gc, wr: 0.27, axF: hl - 0.6, axR: -hl + 0.55, doors: [0.9, 0.75], hatch: true, rim: 'cap', gh: 0.1,
        body: [[-hl, gc + 0.1], [-hl + 0.05, gc], [hl - 0.1, gc], [hl, gc + 0.16], [hl - 0.04, belt - 0.15], [hl - 0.5, belt - 0.02],
          [hl - 0.8, belt], [-hl + 0.06, belt], [-hl, belt - 0.12]],
        green: [[hl - 0.8, belt], [hl - 1.4, H - 0.03], [-hl + 0.3, H - 0.03], [-hl + 0.06, belt]],
      });
    }
    case 'compact': {                           // Fit / Yaris style hatchback
      const gc = 0.17, belt = H * 0.6;
      return car({
        gc, wr: 0.3, axF: hl - 0.75, axR: -hl + 0.65, doors: [1.0, 0.85], hatch: true, rim: 'alloy',
        body: [[-hl, gc + 0.12], [-hl + 0.06, gc], [hl - 0.12, gc], [hl, gc + 0.2], [hl - 0.05, belt - 0.18], [hl - 0.6, belt - 0.03],
          [hl - 1.0, belt], [-hl + 0.1, belt], [-hl, belt - 0.12]],
        green: [[hl - 1.0, belt], [hl - 1.8, H - 0.03], [-hl + 0.45, H - 0.03], [-hl + 0.12, belt]],
      });
    }
    case 'sedan': {                             // Corolla style: hood, raked glass, short trunk deck
      const gc = 0.17, belt = H * 0.62;
      return car({
        gc, wr: 0.31, axF: hl - 0.85, axR: -hl + 0.95, doors: [1.05, 0.9], rim: 'alloy',
        body: [[-hl, gc + 0.14], [-hl + 0.08, gc], [hl - 0.12, gc], [hl, gc + 0.2], [hl - 0.06, belt - 0.16], [hl - 0.75, belt - 0.02],
          [hl - 1.15, belt], [-hl + 1.0, belt], [-hl + 0.15, belt + 0.03], [-hl, belt - 0.1]],
        green: [[hl - 1.15, belt], [hl - 2.0, H - 0.03], [-hl + 1.6, H - 0.03], [-hl + 0.95, belt]],
      });
    }
    case 'suv': {                               // RAV4 / CX-5 style: high ground clearance, big wheels, black cladding
      const gc = 0.22, belt = H * 0.56;
      return car({
        gc, wr: 0.36, axF: hl - 0.9, axR: -hl + 0.98, doors: [1.1, 0.95], hatch: true, rim: 'alloy', gh: 0.13,
        clad: true, rails: true,
        body: [[-hl, gc + 0.16], [-hl + 0.08, gc], [hl - 0.14, gc], [hl, gc + 0.24], [hl - 0.03, belt - 0.2], [hl - 0.55, belt - 0.04],
          [hl - 1.1, belt], [-hl + 0.1, belt], [-hl, belt - 0.14]],
        green: [[hl - 1.1, belt], [hl - 1.95, H - 0.08], [-hl + 0.35, H - 0.08], [-hl + 0.1, belt]],
      });
    }
    case 'minivan': {                           // Serena style: short sloped hood, long roof, upright rear
      const gc = 0.17, belt = H * 0.52;
      return car({
        gc, wr: 0.32, axF: hl - 0.85, axR: -hl + 0.8, doors: [1.0, 0.9],
        slide: true, quarterF: true, hatch: true, rim: 'cap', gh: 0.15, rows3: true,
        body: [[-hl, gc + 0.14], [-hl + 0.06, gc], [hl - 0.12, gc], [hl, gc + 0.22], [hl - 0.05, belt - 0.22], [hl - 0.5, belt - 0.05],
          [hl - 0.95, belt], [-hl + 0.05, belt], [-hl, belt - 0.15]],
        green: [[hl - 0.95, belt], [hl - 1.65, H - 0.04], [-hl + 0.12, H - 0.04], [-hl + 0.03, belt]],
      });
    }
    // ---- the types that share a size class with the eight of cars.js (SUBTYPES below); their profiles are new
    case 'keisedan': {                          // a kei with a longer nose and raked glass at both ends
      const gc = 0.16, belt = H * 0.56;
      return car({
        gc, wr: 0.27, axF: hl - 0.62, axR: -hl + 0.55, doors: [0.92, 0.72], hatch: true, gh: 0.09,
        body: [[-hl, gc + 0.12], [-hl + 0.05, gc], [hl - 0.1, gc], [hl, gc + 0.16], [hl - 0.05, belt - 0.14], [hl - 0.55, belt - 0.02],
          [hl - 0.9, belt], [-hl + 0.1, belt], [-hl, belt - 0.12]],
        green: [[hl - 0.9, belt], [hl - 1.6, H - 0.03], [-hl + 0.62, H - 0.03], [-hl + 0.14, belt]],
      });
    }
    case 'keisuv': {                            // a square kei on tall tyres with unpainted fenders (both kei heights)
      const gc = 0.2, belt = Math.min(1.0, H * 0.57), top = H - (H > 1.68 ? 0.08 : 0.03);
      return car({
        gc, wr: 0.29, axF: hl - 0.6, axR: -hl + 0.55, doors: [0.88, 0.76], hatch: true, clad: true, rails: H > 1.68, gh: 0.12,
        body: [[-hl, gc + 0.14], [-hl + 0.05, gc], [hl - 0.09, gc], [hl, gc + 0.2], [hl - 0.015, belt - 0.14], [hl - 0.3, belt - 0.03],
          [hl - 0.62, belt], [-hl + 0.04, belt], [-hl, belt - 0.14]],
        green: [[hl - 0.62, belt], [hl - 0.9, top], [-hl + 0.14, top], [-hl + 0.04, belt]],
      });
    }
    case 'keivan': {                            // a kei van: hardly a nose, the driver over the front wheels, upright rear
      const gc = 0.18, belt = Math.min(0.98, H * 0.54);
      return car({
        gc, wr: 0.27, axF: hl - 0.62, axR: -hl + 0.6, zf: hl - 0.36, doors: [0.95, 1.0], slide: true, hatch: true, gh: 0.08,
        body: [[-hl, gc + 0.12], [-hl + 0.04, gc], [hl - 0.07, gc], [hl, gc + 0.16], [hl - 0.01, belt - 0.1], [hl - 0.12, belt - 0.02],
          [hl - 0.3, belt], [-hl + 0.03, belt], [-hl, belt - 0.14]],
        green: [[hl - 0.3, belt], [hl - 0.72, H - 0.04], [-hl + 0.08, H - 0.04], [-hl + 0.02, belt]],
      });
    }
    case 'tallwagon': {                         // a tall small wagon: upright screen, short high bonnet, square tail
      const gc = 0.17, belt = H * 0.58;
      return car({
        gc, wr: 0.3, axF: hl - 0.72, axR: -hl + 0.62, doors: [0.98, 0.88], hatch: true, quarterF: true, gh: 0.12,
        body: [[-hl, gc + 0.13], [-hl + 0.06, gc], [hl - 0.1, gc], [hl, gc + 0.2], [hl - 0.03, belt - 0.17], [hl - 0.45, belt - 0.035],
          [hl - 0.82, belt], [-hl + 0.06, belt], [-hl, belt - 0.14]],
        green: [[hl - 0.82, belt], [hl - 1.35, H - 0.03], [-hl + 0.22, H - 0.03], [-hl + 0.06, belt]],
      });
    }
    case 'smallsuv': {                          // a small SUV: the compact on bigger wheels, with cladding and rails
      const gc = 0.2, belt = H * 0.6;
      return car({
        gc, wr: 0.33, axF: hl - 0.76, axR: -hl + 0.68, doors: [1.0, 0.85], hatch: true, clad: true, rails: true, gh: 0.12,
        body: [[-hl, gc + 0.15], [-hl + 0.07, gc], [hl - 0.12, gc], [hl, gc + 0.23], [hl - 0.03, belt - 0.17], [hl - 0.55, belt - 0.03],
          [hl - 0.98, belt], [-hl + 0.1, belt], [-hl, belt - 0.13]],
        green: [[hl - 0.98, belt], [hl - 1.7, H - 0.08], [-hl + 0.38, H - 0.08], [-hl + 0.1, belt]],
      });
    }
    case 'wagon': {                             // an estate car: the saloon's front, a long roof and a raked tailgate
      const gc = 0.17, belt = H * 0.62;
      return car({
        gc, wr: 0.31, axF: hl - 0.85, axR: -hl + 0.95, doors: [1.05, 0.9], hatch: true,
        body: [[-hl, gc + 0.14], [-hl + 0.08, gc], [hl - 0.12, gc], [hl, gc + 0.2], [hl - 0.06, belt - 0.16], [hl - 0.75, belt - 0.02],
          [hl - 1.15, belt], [-hl + 0.1, belt], [-hl, belt - 0.12]],
        green: [[hl - 1.15, belt], [hl - 2.0, H - 0.03], [-hl + 0.6, H - 0.03], [-hl + 0.14, belt]],
      });
    }
    case 'hatch': {                             // a wedge-shaped hatchback: low nose, the roof's top well forward, high tail
      const gc = 0.16, belt = H * 0.63;
      return car({
        gc, wr: 0.32, axF: hl - 0.88, axR: -hl + 0.85, doors: [1.08, 0.88], hatch: true,
        body: [[-hl, gc + 0.2], [-hl + 0.08, gc], [hl - 0.14, gc], [hl, gc + 0.18], [hl - 0.08, belt - 0.2], [hl - 0.8, belt - 0.04],
          [hl - 1.2, belt], [-hl + 0.06, belt], [-hl, belt - 0.07]],
        green: [[hl - 1.2, belt], [hl - 2.05, H - 0.03], [-hl + 1.5, H - 0.03], [-hl + 0.32, belt]],
      });
    }
    case 'fourwd': {                            // a square off-roader: upright faces, flat bonnet (no spare wheel: it would pass L)
      const gc = 0.24, belt = H * 0.57;
      return car({
        gc, wr: 0.37, axF: hl - 0.85, axR: -hl + 0.95, doors: [1.05, 0.95], hatch: true, clad: true, rails: true, gh: 0.14,
        body: [[-hl, gc + 0.16], [-hl + 0.06, gc], [hl - 0.1, gc], [hl, gc + 0.26], [hl - 0.01, belt - 0.13], [hl - 0.25, belt - 0.02],
          [hl - 1.05, belt], [-hl + 0.05, belt], [-hl, belt - 0.14]],
        green: [[hl - 1.05, belt], [hl - 1.3, H - 0.08], [-hl + 0.14, H - 0.08], [-hl + 0.05, belt]],
      });
    }
    case 'talltaxi': {                          // a tall taxi: short bonnet, big glass, a sliding door
      const gc = 0.17, belt = H * 0.54;
      return car({
        gc, wr: 0.31, axF: hl - 0.85, axR: -hl + 0.85, doors: [1.0, 1.0], slide: true, hatch: true, gh: 0.13,
        body: [[-hl, gc + 0.14], [-hl + 0.06, gc], [hl - 0.12, gc], [hl, gc + 0.22], [hl - 0.04, belt - 0.2], [hl - 0.55, belt - 0.045],
          [hl - 0.95, belt], [-hl + 0.06, belt], [-hl, belt - 0.15]],
        green: [[hl - 0.95, belt], [hl - 1.5, H - 0.04], [-hl + 0.22, H - 0.04], [-hl + 0.06, belt]],
      });
    }
    case 'onebox': {                            // a one-box van: no nose, the driver over the front wheels
      const gc = 0.19, belt = H * 0.5;
      return car({
        gc, wr: 0.32, axF: hl - 0.78, axR: -hl + 0.95, zf: hl - 0.48, doors: [1.05, 1.25], slide: true, hatch: true, gh: 0.1, rows3: true,
        body: [[-hl, gc + 0.14], [-hl + 0.05, gc], [hl - 0.09, gc], [hl, gc + 0.2], [hl - 0.015, belt - 0.12], [hl - 0.16, belt - 0.025],
          [hl - 0.38, belt], [-hl + 0.04, belt], [-hl, belt - 0.16]],
        green: [[hl - 0.38, belt], [hl - 0.95, H - 0.04], [-hl + 0.1, H - 0.04], [-hl + 0.03, belt]],
      });
    }
    case 'keitruck': {                          // Carry / Hijet style cab over the front wheels
      const gc = 0.32, belt = 1.02, c0 = hl - 1.32, yTop = H - 0.05, axF = hl - 0.6;
      return {
        cab: true, gc, wr: 0.27, ra: 0.31, arches: [axF], wheels: [axF, axF - 1.9], tw: 0.17, tx: 0.13, rim: 'steel',
        seams: [hl - 0.3, c0 + 0.2], handles: [c0 + 0.24], c0, belt, top: H - 0.02,
        body: [[c0, gc + 0.03], [c0 + 0.03, gc], [hl - 0.04, gc], [hl, gc + 0.1], [hl, belt - 0.07], [hl - 0.08, belt], [c0 + 0.03, belt], [c0, belt - 0.03]],
        green: [[hl - 0.08, belt], [hl - 0.3, yTop], [c0 + 0.07, yTop], [c0 + 0.05, belt]],
      };
    }
    case 'truck': {                             // 2 t truck: a tall cab over the front wheels
      const cabL = Math.min(1.9, L * 0.3), cabH = Math.min(H * 0.82, 2.45), c0 = hl - cabL, gc = 0.45, belt = cabH - 0.64, zw = hl - cabL * 0.55;
      return {
        cab: true, gc, wr: 0.42, ra: 0.47, arches: [zw], wheels: [zw, -hl + 1.1, -hl + 2.0], tw: 0.26, tx: 0.2, rim: 'steel',
        seams: [hl - 0.32, c0 + 0.2 * cabL], handles: [c0 + 0.2 * cabL + 0.05], c0, belt, top: cabH, cabL, cabH,
        body: [[c0, gc + 0.03], [c0 + 0.03, gc], [hl - 0.05, gc], [hl, gc + 0.12], [hl, belt - 0.06], [hl - 0.05, belt], [c0 + 0.03, belt], [c0, belt - 0.03]],
        green: [[hl - 0.05, belt], [hl - 0.27, cabH - 0.03], [c0 + 0.08, cabH - 0.03], [c0 + 0.06, belt]],
      };
    }
    default: return null;
  }
}

/** What the loft adds to a side profile (metres): the plan, the section and the glass. */
const BASE = {
  rF: 0.36, rzF: 0.42, bowF: 0.06, taperF: 0.06, noseR: 0.09,      // nose: corner radius across / along, bow, taper, profile rounding
  rR: 0.30, rzR: 0.30, bowR: 0.04, taperR: 0.045, tailR: 0.07, deckR: 0.28,
  fender: 0.016, tuck: 0.035, sh: 0.11, lean: 0.036, drop: 0.028,  // fender swell, sill tuck, shoulder height / lean, deck edge drop
  shelf: 0.03, tumble: 0.115, rG: 0.13, bowWS: 0.045, bowRG: 0.03, crownRoof: 0.045,
  cW: 0.07, lip: 0.032,                                            // C pillar width at the belt, wheel arch lip
};
const STYLE = {
  compact: BASE,
  keihatch: { ...BASE, rF: 0.27, rzF: 0.3, bowF: 0.05, taperF: 0.045, rR: 0.25, rzR: 0.25, taperR: 0.035, deckR: 0.2, fender: 0.014, tuck: 0.03,
    tumble: 0.085, rG: 0.11, bowWS: 0.04, crownRoof: 0.04 },
  keiwagon: { ...BASE, rF: 0.25, rzF: 0.27, bowF: 0.045, taperF: 0.04, noseR: 0.07, rR: 0.25, rzR: 0.22, bowR: 0.035, taperR: 0.03, deckR: 0.14,
    fender: 0.013, tuck: 0.028, sh: 0.1, lean: 0.03, tumble: 0.06, rG: 0.1, bowWS: 0.035, bowRG: 0.025, crownRoof: 0.035 },
  sedan: { ...BASE, rF: 0.38, rzF: 0.46, taperF: 0.065, rR: 0.32, rzR: 0.34, taperR: 0.06, tumble: 0.125, cW: 0.26 },
  suv: { ...BASE, rF: 0.32, rzF: 0.38, bowF: 0.05, taperF: 0.05, rR: 0.28, rzR: 0.26, taperR: 0.04, fender: 0.024, tuck: 0.03, tumble: 0.1,
    cW: 0.16, lip: 0.085 },
  minivan: { ...BASE, rF: 0.3, rzF: 0.36, bowF: 0.05, taperF: 0.045, rR: 0.25, rzR: 0.22, taperR: 0.03, deckR: 0.18, tumble: 0.075, rG: 0.11,
    bowRG: 0.025, crownRoof: 0.04 },
  keitruck: { ...BASE, rF: 0.2, rzF: 0.2, bowF: 0.035, taperF: 0.03, noseR: 0.06, rR: 0.05, rzR: 0.05, bowR: 0, taperR: 0, tailR: 0.02, deckR: 0.04,
    fender: 0.01, tuck: 0.02, sh: 0.09, lean: 0.02, drop: 0.02, shelf: 0.02, tumble: 0.04, rG: 0.08, bowWS: 0.03, bowRG: 0, crownRoof: 0.03, cW: 0.2 },
  truck: { ...BASE, rF: 0.22, rzF: 0.2, bowF: 0.04, taperF: 0.025, noseR: 0.07, rR: 0.06, rzR: 0.06, bowR: 0, taperR: 0, tailR: 0.02, deckR: 0.04,
    fender: 0.012, tuck: 0.02, sh: 0.1, lean: 0.02, drop: 0.02, shelf: 0.025, tumble: 0.05, rG: 0.09, bowWS: 0.035, bowRG: 0, crownRoof: 0.035,
    cW: 0.25, lip: 0.04 },
};
Object.assign(STYLE, {
  keisedan: { ...STYLE.keihatch, tumble: 0.095 },
  keisuv: { ...STYLE.keiwagon, fender: 0.018, tumble: 0.055, cW: 0.1, lip: 0.07 },
  keivan: { ...STYLE.keiwagon, rzF: 0.25, bowF: 0.04, taperF: 0.035, taperR: 0.02, deckR: 0.08, noseR: 0.06, tumble: 0.045, cW: 0.12 },
  tallwagon: { ...BASE, rF: 0.3, rzF: 0.34, bowF: 0.05, taperF: 0.05, rR: 0.26, rzR: 0.24, taperR: 0.035, deckR: 0.2, tumble: 0.09, rG: 0.12, cW: 0.1 },
  smallsuv: { ...STYLE.suv, rF: 0.33, rzF: 0.4, fender: 0.022, lip: 0.075 },
  wagon: { ...STYLE.sedan, rR: 0.28, rzR: 0.26, taperR: 0.045, cW: 0.14 },
  hatch: { ...STYLE.sedan, taperF: 0.07, taperR: 0.055, cW: 0.3, tumble: 0.13 },
  fourwd: { ...STYLE.suv, rF: 0.26, rzF: 0.28, bowF: 0.04, taperF: 0.04, rR: 0.25, rzR: 0.22, taperR: 0.03, deckR: 0.1, noseR: 0.06, tumble: 0.06,
    bowWS: 0.035, rG: 0.1, crownRoof: 0.035, cW: 0.14 },
  talltaxi: { ...STYLE.minivan, cW: 0.12 },
  onebox: { ...STYLE.minivan, rF: 0.27, rzF: 0.28, bowF: 0.045, taperF: 0.04, taperR: 0.02, deckR: 0.08, noseR: 0.07, tumble: 0.055, cW: 0.12 },
  flatbed: STYLE.truck, reefer: STYLE.truck, delivery: STYLE.truck,
});

/**
 * Body types within KeiSim's size classes (the eight of cars.js carType): [type, share]. The class comes from the
 * size (and, for kei trucks, the colour); the type within it from the hash of the car.
 */
export const SUBTYPES = {
  keihatch: [['keihatch', 0.55], ['keisedan', 0.25], ['keisuv', 0.2]],
  keiwagon: [['keiwagon', 0.6], ['keivan', 0.25], ['keisuv', 0.15]],
  keitruck: [['keitruck', 1]],
  compact: [['compact', 0.55], ['tallwagon', 0.25], ['smallsuv', 0.2]],
  sedan: [['sedan', 0.45], ['wagon', 0.25], ['hatch', 0.3]],
  suv: [['suv', 0.6], ['fourwd', 0.2], ['talltaxi', 0.2]],
  minivan: [['minivan', 0.65], ['onebox', 0.35]],
  truck: [['truck', 0.45], ['flatbed', 0.25], ['reefer', 0.15], ['delivery', 0.15]],
};
const TRUCKS = ['truck', 'flatbed', 'reefer', 'delivery'];
/** Colours a taxi comes in: white, black, dark blue. */
const taxiColour = (rgb) => {
  const lo = Math.min(...rgb), hi = Math.max(...rgb);
  return (lo > 190 && hi - lo < 25) || hi < 75 || (rgb[2] > rgb[0] + 40 && rgb[2] > rgb[1] + 25 && rgb[2] < 200);
};
/** { base (the class, cars.js carType), type } of a car. A tall taxi only comes in a taxi's colours: there it takes
 *  more than its share of the class, elsewhere none, which adds up to the table's share over KeiSim's colours. */
export function carType2(kind, L, W, H, rgb, id = 0, force = null) {
  const base = carType(kind, L, W, H, rgb);
  let list = SUBTYPES[base];
  if (force && list.some((e) => e[0] === force)) return { base, type: force };
  if (base === 'suv') list = taxiColour(rgb) ? [['suv', 0.3], ['fourwd', 0.15], ['talltaxi', 0.55]] : [['suv', 0.77], ['fourwd', 0.23]];
  let u = roll(carHash(id, L, W, H, rgb), 11);
  for (const [type, share] of list) { if (u < share) return { base, type }; u -= share; }
  return { base, type: list[list.length - 1][0] };
}


/** Wheels by body type (the first of a list comes twice as often as the others). */
const RIMS = {
  compact: ['spoke5', 'cap', 'spoke10', 'dish'], keihatch: ['cap', 'dish', 'spoke5'], keiwagon: ['cap', 'spoke5', 'dish', 'black'],
  sedan: ['spoke5', 'spoke10', 'dish', 'cap'], suv: ['spoke5', 'black', 'spoke10'], minivan: ['spoke10', 'cap', 'spoke5', 'dish'],
  keisedan: ['cap', 'dish', 'spoke5'], keisuv: ['black', 'spoke5', 'cap'], keivan: ['cap', 'cap', 'dish'], tallwagon: ['cap', 'spoke5', 'spoke10', 'dish'],
  smallsuv: ['spoke5', 'black', 'spoke10'], wagon: ['spoke10', 'spoke5', 'dish'], hatch: ['spoke5', 'spoke10', 'black'], fourwd: ['black', 'spoke5', 'cap'],
  talltaxi: ['cap', 'dish'], onebox: ['cap', 'dish', 'spoke5'],
};
/**
 * What the hash of a car picks besides its face: { wheel, roof and pillar (material keys: 'body', or 'plateW' /
 * 'trim' for a white / black roof, 'trim' for blacked-out pillars), plate (its number), driver, key }. A kei car has
 * a two-tone roof one time in five (white, or black on a pale car), an SUV a black roof 15 % of the time; a quarter
 * of the others have black pillars. opt: { wheel, roof, pillar, driver } force them (the gallery; driver: false for a
 * parked car).
 */
function chooseTrim(type, h, rgb, opt) {
  const o = opt || {}, list = RIMS[type];
  const k = list ? Math.floor(roll(h, 7) * (list.length + 1)) : 0;
  const wheel = o.wheel || (list ? list[Math.max(0, k - 1)] : 'steel');
  const kei = type.startsWith('kei') && type !== 'keitruck';
  const lo = Math.min(...rgb), hi = Math.max(...rgb), pale = lo > 150, dark = hi < 70;
  let roof = 'body', pillar = 'body';
  if (list) {
    const u = roll(h, 8);
    if (kei && u < 0.2) roof = pillar = pale ? 'trim' : 'plateW';
    else if (['suv', 'smallsuv', 'fourwd'].includes(type) && u < 0.15 && !dark) roof = pillar = 'trim';
    else if (roll(h, 9) < 0.25 && !dark) pillar = 'trim';
  }
  // working cars and small things: what the car is used for, a mark on its nose and tail, a roof carrier
  let work = null, mark = null, carrier = null, panel = false, load = 0;
  if (type === 'talltaxi') work = 'taxi';
  else if (type === 'sedan') {
    if (taxiColour(rgb) && roll(h, 12) < 0.15) work = 'taxi';
    else if (lo > 190 && hi - lo < 25 && roll(h, 13) < 0.03) work = 'patrol';
    else if (roll(h, 14) < 0.02) work = 'school';
  } else if (type === 'keivan') {
    if (rgb[0] > rgb[1] + 60 && rgb[0] > rgb[2] + 60) work = 'post';
    panel = roll(h, 18) < 0.4;
  } else if (type === 'flatbed') load = Math.floor(roll(h, 19) * 3);
  if (list && !work && roll(h, 15) < 0.06) mark = roll(h, 20) < 0.5 ? 'beginner' : 'elder';
  if (['suv', 'fourwd', 'keivan'].includes(type) && roll(h, 16) < 0.15) carrier = type === 'keivan' && roll(h, 17) < 0.5 ? 'ladder' : 'rack';
  if (o.work !== undefined) work = o.work;
  if (o.mark !== undefined) mark = o.mark;
  if (o.carrier !== undefined) carrier = o.carrier;
  if (o.panel !== undefined) panel = o.panel;
  if (o.load !== undefined) load = o.load;
  if (work === 'patrol') roof = pillar = 'trim';
  if (work === 'taxi' || work === 'patrol' || work === 'school') { mark = null; if (carrier) carrier = null; }
  if (o.roof) roof = pillar = o.roof;
  if (o.pillar) pillar = o.pillar;
  const plate = 1 + Math.floor(roll(h, 10) * 9998), driver = o.driver !== false;
  const extras = [work, mark, carrier, panel ? 'panel' : null, load ? `load${load}` : null].filter(Boolean).join('+');
  return { wheel, roof, pillar, plate, driver, work, mark, carrier, panel, load,
    key: `${wheel}|${roof}|${pillar}|${plate}|${extras}|${driver ? 'd' : '-'}` };
}

const mirror = (fn) => (a, b) => { const p = fn(a, b); return [-p[0], p[1], p[2]]; };

// ------------------------------------------------------------------------------------------ lower body
function lowerSpec(d, st, W) {
  const B = d.body, n = B.length, gc = d.gc, ra = d.ra, z1 = B[3][0], z0 = B[0][0];
  const deckY = lin(roundPoly(B.slice(4).map(([z, y]) => [z, y]).reverse(), st.deckR, 4));
  // the nose's profile z(y): chamfer, tip, bonnet edge and on along the bonnet while it rises; the tail likewise
  const fr = [[B[2][1], B[2][0]], [B[3][1], B[3][0]], [B[4][1], B[4][0]]];
  for (let i = 5; i < n - 1 && B[i][1] > fr[fr.length - 1][0] + 1e-6; i++) fr.push([B[i][1], B[i][0]]);
  const rr = [[B[1][1], B[1][0]], [B[0][1], B[0][0]], [B[n - 1][1], B[n - 1][0]]];
  for (let i = n - 2; i > 4 && B[i][1] > rr[rr.length - 1][0] + 1e-6; i--) rr.push([B[i][1], B[i][0]]);
  // the tips are rounded; the bonnet's front edge stays a crease (the outline pass draws it all the way round)
  const zF = lin(roundPoly(fr.slice(0, 3), st.noseR).concat(fr.slice(3))), zR = lin(roundPoly(rr.slice(0, 4), st.tailR).concat(rr.slice(4)));
  const top = (z) => deckY(z) - st.drop;
  const bottom = (z) => {
    let y = gc;
    for (const zc of d.arches) { const dz = Math.abs(z - zc); if (dz <= ra + 1e-9) y = Math.max(y, d.wr + Math.sqrt(Math.max(0, ra * ra - dz * dz))); }
    return y;
  };
  // the corners start clear of the wheel arches
  const aF = Math.max(...d.arches), aR = Math.min(...d.arches);
  const zsF = Math.max(aF + ra + 0.02, z1 - st.rzF), zsR = Math.min(aR - ra - 0.02, z0 + st.rzR);
  const keep = [zsF, zsR, ...B.slice(5, n - 1).map((p) => p[0]).filter((z) => z < zsF - 0.02 && z > zsR + 0.02)];
  for (const zc of d.arches) {
    for (let k = 0; k <= 8; k++) keep.push(zc + ra * Math.cos(Math.PI * k / 8));
    keep.push(zc + ra + 0.006, zc - ra - 0.006);
  }
  const stations = fillStations(keep.filter((z) => z <= zsF + 1e-9 && z >= zsR - 1e-9), 0.12).map((z) => ({ zb: z, z: () => z }));

  const base = W / 2 - st.fender, yFend = d.wr + ra + 0.06;
  const plan = (z) => base * (1 - st.taperF * clamp((z - aF) / Math.max(z1 - aF, 0.1), 0, 1) ** 2 - st.taperR * clamp((aR - z) / Math.max(aR - z0, 0.1), 0, 1) ** 2);
  const swell = (z) => Math.min(1, d.arches.reduce((a, zc) => a + Math.exp(-(((z - zc) / 0.4) ** 2)), 0));
  function hw(z, y) {
    const yT = top(z), yS = yT - st.sh;
    let w = plan(z) + st.fender * swell(z) * (y <= yFend ? 1 : Math.max(0.35, 1 - 0.65 * (y - yFend) / Math.max(yS - yFend, 0.05)));
    const tau = (y - gc) / Math.max(yS - gc, 0.1);
    if (tau < 0.3) w -= st.tuck * (1 - Math.max(tau, 0) / 0.3) ** 2;
    if (y > yS) w -= st.lean * Math.min(1.2, (y - yS) / st.sh) ** 1.5;
    return w;
  }
  const mono = (r, lo, hi) => { for (let k = 1; k < r.length; k++) r[k] = Math.min(hi, Math.max(r[k], r[k - 1], lo)); return r; };
  const tipF = B[3][1], tipR = B[0][1];
  const rows = (s) => {
    const z = s.zb, yB = bottom(z), yT = top(z), yS = yT - st.sh, a = tipR + (tipF - tipR) * (z - z0) / (z1 - z0) + 0.05;
    return mono([yB, gc + 0.06, a - 0.1, a, a + (yS - a) / 3, a + 2 * (yS - a) / 3, yS, yS + 0.55 * st.sh, yT], yB, yT);
  };
  const faceRows = (y0, tip, y1) => { const a = tip + 0.05, b = y1 - 0.09; return mono([y0, y0 + 0.06, tip - 0.05, a, a + (b - a) / 3, a + 2 * (b - a) / 3, b, y1 - 0.03, y1], y0, y1); };
  return {
    hw, stations, rows, deckY, top, bottom, zsF, zsR, ra, z0, z1,
    front: { z: zF, rows: faceRows(B[2][1], tipF, B[4][1]), r: st.rF, bow: st.bowF },
    rear: { z: zR, rows: faceRows(B[1][1], tipR, B[n - 2][1]), r: st.rR, bow: st.bowR },
    nF: 3, nC: 8,
  };
}

// ------------------------------------------------------------------------------------------ greenhouse
function greenSpec(d, st, edge) {
  const [[wsBot, belt], [wsTop, yTopD], [rearTop], [rb]] = d.green;
  const rearBot = Math.max(rb, d.body[d.body.length - 2][0] + 0.02);          // not behind the deck's rear edge
  const yRoof = yTopD + 0.03, yb = belt - 0.03, yE = yRoof - st.crownRoof;
  const zF = (y) => wsBot + (wsTop - wsBot) * (y - belt) / (yTopD - belt), zR = (y) => rearBot + (rearTop - rearBot) * (y - belt) / (yTopD - belt);
  const sF = (y) => zF(y) - st.bowWS - st.rG, sR = (y) => zR(y) + st.bowRG + st.rG;
  const knots = [[sF(yb), sF(yE)], [sR(yb), sR(yE)]], pillars = [], more = [];
  const upright = (z, half, key) => {                 // a pillar that stands upright, if the leaning ends leave room
    if (z == null || z + half > sF(yE) - 0.12 || z - half < sR(yE) + 0.12) return;
    knots.push([z + half, z + half], [z - half, z - half]); pillars.push({ z0: z - half, z1: z + half, key });
  };
  upright(d.zb, 0.04, 'trim'); upright(d.zr, 0.022, 'trim');
  if (d.quarterF && d.zf < sF(yb) - 0.1) {             // the divider of a front quarter glass leans with the screen
    more.push(d.zf - 0.02, d.zf + 0.02); pillars.push({ z0: d.zf - 0.02, z1: d.zf + 0.02, key: 'trim' });
  }
  const zC = sR(yb) + st.cW;                           // the rearmost pillar is body-coloured up to here
  if (zC < sF(yb) - 0.25) more.push(zC);
  const topOf = lin(knots);
  const stations = fillStations([...knots.map((k) => k[0]), ...more], 0.3).map((z) => {
    const zt = topOf(z);
    return { zb: z, zt, z: (y) => z + (zt - z) * (y - yb) / (yE - yb) };
  });
  const hw = (z, y) => {
    const t = clamp((y - belt) / (yE - belt), 0, 1), u = clamp((y - (yE - 0.075)) / 0.075, 0, 1);
    return edge(z) - st.shelf - st.tumble * t ** 1.2 - 0.045 * u * u;
  };
  const r = [yb, belt + 0.022, yE - 0.075, yE - 0.025, yE];
  const deckY = (z) => { const s = clamp((z - rearTop) / (wsTop - rearTop), 0, 1); return yE + st.crownRoof * (0.55 + 0.45 * (1 - (2 * s - 1) ** 2)); };
  return {
    hw, stations, rows: () => r.slice(), deckY, yb, yE, yRoof, belt, pillars, sF, sR, zC, wsTop, rearTop,
    front: { z: zF, rows: r.slice(), r: st.rG, bow: st.bowWS }, rear: { z: zR, rows: r.slice(), r: st.rG, bow: st.bowRG },
    nF: 3, nC: 4,
  };
}

// ------------------------------------------------------------------------------------------ small parts
const disc = (M, key, r, t, x, y, z, seg = 16) => { const g = new THREE.CylinderGeometry(r, r, t, seg); g.rotateZ(Math.PI / 2); g.translate(x, y, z); M.geometry(key, g); };
const box = (M, key, w, h, dd, x, y, z, rx = 0, ry = 0) => { const g = new THREE.BoxGeometry(w, h, dd); if (rx) g.rotateX(rx); if (ry) g.rotateY(ry); g.translate(x, y, z); M.geometry(key, g); };
/** A surface of revolution round the cross axis at (x, yc, z) on side s: prof [[radius, how far outwards], ...]. */
function lathe(M, key, s, x, yc, z, prof, seg = 16) {
  const P = [];
  for (const [r, dx] of prof) for (let i = 0; i < seg; i++) { const a = 2 * Math.PI * i / seg; P.push([x + s * dx, yc + r * Math.cos(a), z + r * Math.sin(a)]); }
  M.grid(P, seg, prof.length, { mat: key, wrap: true, flip: s < 0, group: (i, j) => `b${j}` });
}
/** A flat patch on the face of a wheel (plane x): pts [[y, z], ...] as a fan, facing outwards on side s. */
function onWheel(M, key, s, x, pts) {
  const tri = [];
  for (let k = 1; k + 1 < pts.length; k++) {
    const [a, b, c] = [pts[0], pts[k], pts[k + 1]];
    const nx = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const o = nx * s > 0 ? [a, b, c] : [a, c, b];
    for (const p of o) tri.push([x, p[0], p[1]]);
  }
  M.flat(key, tri);
}
export const WHEELS = ['cap', 'spoke5', 'spoke10', 'dish', 'black'];
/**
 * Tyre and wheel on side s. The tyre: tread, a shoulder that stands proud of the side wall, the inner side closed.
 * The wheel: 'cap' steel with a hub cap (a dark ring), 'spoke5' / 'spoke10' alloys (dark openings between the spokes),
 * 'dish' a smooth disc with small slots, 'black' a dark wheel with bright spokes, 'steel' as on the trucks.
 */
function wheel(M, s, x, z, r, w, style) {
  const h = w / 2, rr = r * 0.7;
  lathe(M, 'tire', s, x, r, z, [[0, -h], [r, -h], [r, h - 0.004], [r * 0.87, h + 0.012], [rr, h]]);
  const xf = x + s * (h - 0.012), up = xf + s * 0.004;
  const sector = (key, a0, a1, r0, r1) => onWheel(M, key, s, up, [[r + r0 * Math.cos(a0), z + r0 * Math.sin(a0)], [r + r1 * Math.cos(a0), z + r1 * Math.sin(a0)],
    [r + r1 * Math.cos(a1), z + r1 * Math.sin(a1)], [r + r0 * Math.cos(a1), z + r0 * Math.sin(a1)]]);
  const cap = (key, rc) => { const p = []; for (let k = 0; k < 8; k++) p.push([r + rc * Math.cos(k * Math.PI / 4), z + rc * Math.sin(k * Math.PI / 4)]); onWheel(M, key, s, up + s * 0.004, p); };
  const dark = style === 'black';
  lathe(M, dark ? 'trim' : 'hub', s, x, r, z, [[rr, h], [rr * 0.93, h - 0.012], [0, h - 0.012]]);
  if (style === 'steel') { cap('trim', r * 0.24); return; }
  const n = style === 'spoke10' ? 10 : style === 'dish' ? 8 : style === 'cap' ? 12 : 5;
  for (let k = 0; k < n; k++) {
    const a = 0.3 + 2 * Math.PI * k / n, da = 2 * Math.PI / n;
    if (style === 'cap') sector('trim', a, a + da * 1.02, rr * 0.58, rr * 0.68);                    // the ring round a hub cap
    else if (style === 'dish') sector('trim', a + da * 0.3, a + da * 0.7, rr * 0.72, rr * 0.86);    // slots near the rim
    else if (dark) sector('hub', a - da * 0.09, a + da * 0.09, rr * 0.2, rr * 0.9);                 // bright spokes
    else sector('trim', a + da * (n === 5 ? 0.16 : 0.2), a + da * (n === 5 ? 0.84 : 0.8), rr * 0.3, rr * 0.86);   // openings
  }
  cap(dark ? 'hub' : 'trim', r * 0.13);
}

const SKIN = ['body', 'glass', 'trim', 'interior', 'seat', 'head', 'hub', 'cargo', 'plateW', 'plateY', 'plateG'];
/** Scale what has been built so far (not the marked extras) to the box: half width, top, z0 .. z1. */
function fitTo(M, T, bx) {
  const bb = M.bounds(SKIN), bt = T.bounds();
  for (let k = 0; k < 3; k++) { bb.lo[k] = Math.min(bb.lo[k], bt.lo[k]); bb.hi[k] = Math.max(bb.hi[k], bt.hi[k]); }
  const sx = bx.hw / Math.max(bb.hi[0], -bb.lo[0]), sy = bx.top / bb.hi[1], sz = (bx.z1 - bx.z0) / (bb.hi[2] - bb.lo[2]);
  const f = ([x, y, z]) => [x * sx, y * sy, bx.z0 + (z - bb.lo[2]) * sz];
  M.warp(f, SKIN); T.warp(f);
  return [sx, sy, sz];
}

// ------------------------------------------------------------------------------------------ the lofted body
/** Lower body, greenhouse, interior and what lies on the sides, into M. Returns the surfaces for the faces. */
function loftBody(M, d, st, W, tr) {
  const hw = W / 2, ra = d.ra, belt = d.green[0][1];

  // ---- lower body: skin and deck
  const LS = lowerSpec(d, st, W), lo = shell(LS);
  lo.rimZ = lo.deck.rim.map((o, j) => lo.deck.P[j * lo.deck.nu + lo.nF][2]);     // centre line z of each deck line
  M.grid(lo.P, lo.nu, lo.nv, { mat: 'body', wrap: true, group: (i, j) => (j >= 6 ? 'shoulder' : 'side') });
  const edge = planEdge(lo);
  const GS = greenSpec(d, st, edge);
  // the deck, without what lies under the cabin (the interior covers it)
  const cab0 = GS.sR(GS.yb) + 0.12, cab1 = GS.sF(GS.yb) - 0.12;
  M.grid(lo.deck.P, lo.deck.nu, lo.deck.nv, {
    mat: (i, j) => (i > 0 && i < lo.deck.nu - 2 && lo.rimZ[j] < cab1 && lo.rimZ[j + 1] > cab0 ? null : 'body'), group: () => 'deck',
  });
  // floor: a flat underside between the sills, and round the nose and tail
  const floor = [];
  for (const o of lo.deck.rim) {
    const p = o.zone === 'side' ? [LS.hw(o.st.zb, d.gc + 0.01) - 0.012, d.gc, o.st.zb] : lo.point(o, 0);
    floor.push([-p[0], p[1], p[2]], [p[0], p[1], p[2]]);
  }
  M.grid(floor, 2, lo.deck.rim.length, { mat: 'interior', flip: true });
  // wheel houses: a dark liner inside each arch and a wall behind the wheel
  const xw = hw - d.tx - d.tw / 2 - 0.095;
  for (const zc of d.arches) for (const s of [-1, 1]) {
    const arc = [[zc + ra, d.gc]];
    for (let k = 0; k <= 8; k++) { const th = Math.PI * k / 8; arc.push([zc + ra * Math.cos(th), Math.max(d.gc, d.wr + ra * Math.sin(th))]); }
    arc.push([zc - ra, d.gc]);
    const liner = [];
    for (const [z, y] of arc) { const x = LS.hw(z, Math.max(y, d.wr)) - 0.004; liner.push([s * x, y, z]); }
    for (const [z, y] of arc) liner.push([s * xw, y, z]);
    M.grid(liner, arc.length, 2, { mat: 'interior', flip: s > 0 });
    const wall = [];
    for (let k = 0; k + 1 < arc.length; k++) {
      const a = [s * xw, arc[k][1], arc[k][0]], b = [s * xw, arc[k + 1][1], arc[k + 1][0]], c = [s * xw, d.gc, zc];
      wall.push(...(s > 0 ? [c, b, a] : [c, a, b]));
    }
    M.flat('interior', wall);
  }

  // ---- greenhouse: glass all round under a body-coloured roof; the corners are the pillars
  const gh = shell(GS);
  const pillarAt = (q) => {
    if (q.zone !== 'side') return null;
    const z0 = q.b.st.zb, z1 = q.a.st.zb;                           // the quad's span at the belt (z1 > z0)
    for (const p of GS.pillars) if (z0 >= p.z0 - 1e-3 && z1 <= p.z1 + 1e-3) return p.key;
    if (z1 <= GS.zC + 1e-3) return 'body';                           // the last stretch before the rear corner
    return null;
  };
  M.grid(gh.P, gh.nu, gh.nv, {
    wrap: true,
    mat: (i, j) => {
      const q = gh.quad(i), corner = q.zone === 'cornerF' || q.zone === 'cornerR';
      if (j >= 2) return tr.roof;
      if (corner) return tr.pillar;                                  // the corner pillars: body, roof colour or black
      if (j === 0) return 'trim';                                    // the moulding under the glass, all round
      const p = pillarAt(q);
      if (tr.panel && q.zone === 'side' && q.a.st.zb <= d.zb + 1e-3) return p === 'trim' ? 'body' : p || 'body';   // a panel van
      return p === 'body' ? tr.pillar : p || 'glass';
    },
    group: (i, j) => {
      const q = gh.quad(i), corner = q.zone === 'cornerF' || q.zone === 'cornerR';
      if (j >= 2 || corner) return 'gh';
      if (j === 0) return 'moulding';
      return pillarAt(q) ? `pillar${q.side}` : `glass:${q.zone}${q.side}`;
    },
  });
  M.grid(gh.deck.P, gh.deck.nu, gh.deck.nv, { mat: tr.roof, group: () => 'roof' });
  // ---- the interior, as the glass may be see-through (src/world/carglass.js): the pillars' inner faces, a lining
  // under the roof, a tub below the belt (its floor above the wheels), seats, dashboard and steering wheel
  M.grid(gh.P, gh.nu, gh.nv, {
    wrap: true, flip: true,
    mat: (i, j) => { const q = gh.quad(i); return j === 1 && (q.zone === 'cornerF' || q.zone === 'cornerR' || pillarAt(q)) ? 'interior' : null; },
  });
  const thin = GS.stations.filter((s, i) => i % 2 === 0 || i === GS.stations.length - 1);
  const yL = [GS.yb, GS.yE - 0.08];
  const lining = shell({
    nF: 2, nC: 3, hw: (z, y) => GS.hw(z, y) - 0.035, deckY: (z) => GS.deckY(z) - 0.04,
    stations: thin.map((s) => ({ zb: s.zb, z: s.z })), rows: () => yL,
    front: { z: (y) => GS.front.z(y) - 0.045, rows: yL, r: st.rG - 0.02, bow: st.bowWS },
    rear: { z: (y) => GS.rear.z(y) + 0.045, rows: yL, r: st.rG - 0.02, bow: st.bowRG },
  });
  M.grid(lining.deck.P, lining.deck.nu, lining.deck.nv, { mat: 'interior', flip: true });
  const yF = d.wr + ra + 0.03, yW = [yF, belt + 0.022], zbF = GS.front.z(GS.yb), zbR = GS.rear.z(GS.yb);
  const tubHw = (z) => GS.hw(z, GS.yb) - 0.035;
  const tub = shell({
    nF: 2, nC: 3, hw: tubHw, deckY: () => yW[1], stations: thin.map((s) => ({ zb: s.zb, z: () => s.zb })), rows: () => yW,
    front: { z: () => zbF - 0.045, rows: yW, r: st.rG - 0.02, bow: st.bowWS }, rear: { z: () => zbR + 0.045, rows: yW, r: st.rG - 0.02, bow: st.bowRG },
  });
  M.grid(tub.P, tub.nu, tub.nv, { mat: 'interior', wrap: true, flip: true });
  const tubFloor = [];
  for (const o of tub.deck.rim) { const p = tub.point(o, 0); tubFloor.push([-p[0], p[1], p[2]], [p[0], p[1], p[2]]); }
  M.grid(tubFloor, 2, tub.deck.rim.length, { mat: 'interior' });
  // seats: cushion ahead of zBack, backrest leaning back, headrests (their tops under the roof, whatever the floor's height)
  const zA = GS.sF(GS.yb), zZ = GS.sR(GS.yb), hwT = tubHw((zA + zZ) / 2);
  const seatTop = Math.min(belt + 0.42, GS.yE - 0.13), yC = yF + 0.1, ySh = seatTop - 0.17;
  const seat = (x, zBack, w, heads) => {
    box(M, 'seat', w, 0.1, 0.44, x, yF + 0.05, zBack + 0.24);
    box(M, 'seat', w, ySh - yC + 0.04, 0.1, x, (ySh + yC) / 2, zBack + 0.02, -0.16);
    for (const hx of heads) box(M, 'seat', 0.2, 0.15, 0.08, x + hx, ySh + 0.095, zBack - 0.03);
  };
  const pair = (zBack) => { for (const s of [-1, 1]) seat(s * hwT * 0.5, zBack, hwT * 0.72, [0]); };
  const bench = (zBack) => seat(0, zBack, 2 * hwT - 0.1, [-hwT * 0.5, hwT * 0.5]);
  if (tr.driver) {                                                   // a dark figure at the wheel, on the right
    const x = -hwT * 0.5, zDrv = d.cab ? zZ + 0.1 : d.zb + 0.12;
    box(M, 'interior', 0.36, ySh - yC + 0.02, 0.2, x, (ySh + yC) / 2 + 0.01, zDrv + 0.15, -0.1);
    const g = new THREE.SphereGeometry(0.1, 7, 4); g.translate(x, Math.min(ySh + 0.13, GS.yE - 0.2), zDrv + 0.12); M.geometry('interior', g);
    for (const e of [-1, 1]) box(M, 'interior', 0.07, 0.07, 0.34, x + e * 0.15, ySh - 0.1, zDrv + 0.33, 0.35);
  }
  if (d.cab) pair(zZ + 0.1);
  else {
    // the rear seat: over the rear axle, and its headrests inside a raked rear glass
    const z1 = d.zb + 0.12, z3 = Math.max(zZ + 0.2, d.wheels[1] + 0.1, GS.rear.z(seatTop) + st.bowRG + 0.16);
    pair(z1);
    if (d.rows3) pair((z1 + z3) / 2);
    bench(z3);
  }
  // dashboard under the windscreen and the steering wheel on the right (the car's -X)
  const zD = zA - 0.01, hD = Math.min(0.22, belt - yF);
  box(M, 'trim', 2 * hwT - 0.02, hD, 0.3, 0, belt - hD / 2, zD - 0.15);
  box(M, 'trim', 2 * (hwT - st.rG), hD, st.rG - 0.05, 0, belt - hD / 2, zD + (st.rG - 0.05) / 2);
  { const g = new THREE.TorusGeometry(0.16, 0.02, 3, 8); g.rotateX(0.4); g.translate(-hwT * 0.5, belt + 0.03, zD - 0.36); M.geometry('trim', g); }

  // ---- what lies on the sides
  const SL = surface(LS), SG = surface(GS);
  const yS = (z) => LS.top(z) - st.sh;
  for (const [s, side] of [[1, SL.side], [-1, mirror(SL.side)]]) {
    const out = [s, 0, 0];
    // wheel arch lips (wide and unpainted on an SUV)
    for (const zc of d.arches) {
      const a0 = d.gc > d.wr ? Math.asin(Math.min(1, (d.gc - d.wr) / ra)) : 0;
      patch(M, 'trim', (a, b) => side(zc + (ra + b) * Math.cos(a), d.wr + (ra + b) * Math.sin(a)),
        [[a0, -0.003], [Math.PI - a0, -0.003], [Math.PI - a0, st.lip], [a0, st.lip]], { na: 8, nb: 1, lift: 0.004, out, rim: false });   // 8: the arch's own points
    }
    if (d.clad) {                                                    // and a band along the sill between them
      const za = Math.min(...d.arches) + ra + st.lip - 0.01, zb = Math.max(...d.arches) - ra - st.lip + 0.01;
      patch(M, 'trim', side, [[za, d.gc + 0.004], [zb, d.gc + 0.004], [zb, d.gc + 0.13], [za, d.gc + 0.13]], { na: 10, nb: 2, lift: 0.004, out, rim: false });
    }
    // door seams and handles
    for (const z of d.seams) {
      const y0 = Math.max(d.gc + 0.08, LS.bottom(z) + 0.05), y1 = LS.top(z) - 0.015;
      patch(M, 'trim', side, [[z - 0.006, y0], [z + 0.006, y0], [z + 0.006, y1], [z - 0.006, y1]], { na: 1, nb: 7, lift: 0.004, out, rim: false });
    }
    for (const z of d.handles) {
      const y = yS(z) - 0.065;
      patch(M, 'trim', side, [[z, y], [z + 0.15, y], [z + 0.15, y + 0.03], [z, y + 0.03]], { na: 2, nb: 1, lift: 0.012, out });
    }
    if (d.slide) {                                                   // the rail of a sliding door
      const za = LS.zsR + 0.05, zb = d.zr - 0.03, y = (z) => yS(z) - 0.03;
      patch(M, 'trim', side, [[za, y(za) - 0.011], [zb, y(zb) - 0.011], [zb, y(zb) + 0.011], [za, y(za) + 0.011]], { na: 6, nb: 1, lift: 0.004, out, rim: false });
    }
    // what a working car carries on its sides: a taxi's band on the doors, a mail van's white band, a patrol car's
    // dark upper doors (a made-up livery: no letters, no badges)
    const bandOn = (key, z0, z1, ya, yb) => patch(M, key, side, [[z0, ya(z0)], [z1, ya(z1)], [z1, yb(z1)], [z0, yb(z0)]],
      { na: Math.ceil((z1 - z0) / 0.15), nb: 1, lift: 0.004, out, rim: false });
    if (tr.work === 'taxi') bandOn('plateY', d.zr + 0.02, d.zf - 0.02, (z) => yS(z) - 0.24, (z) => yS(z) - 0.17);
    if (tr.work === 'post') bandOn('plateW', LS.zsR + 0.03, d.zf - 0.02, (z) => yS(z) - 0.34, (z) => yS(z) - 0.25);
    if (tr.work === 'patrol') bandOn('trim', d.zr + 0.02, d.zf - 0.02, (z) => yS(z) - 0.3, (z) => yS(z) - 0.03);   // below the shoulder's roll: one flat row lies on the skin there
    // door mirrors: on a car the door's front corner, on a truck a tall one on a stalk (they may reach out of the box)
    if (d.cab) {
      const big = d.cabH != null, zm = LS.z1 - (big ? 0.3 : 0.2), ym = belt + (big ? 0.1 : 0.12);
      box(M, 'extra:trim', big ? 0.08 : 0.05, big ? 0.22 : 0.15, big ? 0.05 : 0.1, s * (hw + (big ? 0.12 : 0.09)), ym, zm);
      box(M, 'extra:trim', 0.12, 0.02, 0.02, s * (hw + 0.03), ym - 0.05, zm);
    } else {
      const zm = d.quarterF ? d.zf - 0.1 : GS.sF(belt) - 0.1, xm = edge(zm);
      box(M, 'extra:body', 0.07, 0.1, 0.17, s * (xm + 0.065), belt + 0.075, zm, 0, s * 0.12);
      box(M, 'extra:trim', 0.1, 0.035, 0.07, s * xm, belt + 0.035, zm);
    }
  }
  if (d.rails) {                                                     // roof rails: the highest thing on the car
    const len = GS.wsTop - GS.rearTop - 0.3, zc = (GS.wsTop + GS.rearTop) / 2, x = GS.hw(zc, GS.yE) - 0.1, yT = GS.yRoof + 0.05;
    for (const s of [-1, 1]) {
      box(M, 'trim', 0.04, 0.035, len, s * x, yT - 0.0175, zc);
      for (const e of [-1, 1]) box(M, 'trim', 0.05, 0.06, 0.08, s * x, yT - 0.05, zc + e * (len / 2 - 0.04));
    }
  }
  // ---- on the roof (all of it may reach out of the box): a taxi's lamp, a patrol car's red lamp, a driving school's
  // sign, a carrier with or without a ladder
  {
    const zc = (GS.wsTop + GS.rearTop) / 2, yR = GS.yRoof, xr = GS.hw(zc, GS.yE);
    if (tr.work === 'taxi') {
      box(M, 'extra:trim', 0.3, 0.025, 0.16, 0, yR + 0.005, GS.wsTop - 0.32);
      box(M, 'extra:head', 0.26, 0.1, 0.12, 0, yR + 0.065, GS.wsTop - 0.32);
    } else if (tr.work === 'patrol') {
      box(M, 'extra:trim', xr * 1.5, 0.03, 0.2, 0, yR + 0.005, zc + 0.15);
      box(M, 'extra:beacon', xr * 1.4, 0.09, 0.15, 0, yR + 0.06, zc + 0.15);
    } else if (tr.work === 'school') {
      box(M, 'extra:plateG', 0.6, 0.04, 0.1, 0, yR + 0.01, zc + 0.1);
      box(M, 'extra:plateW', 0.56, 0.15, 0.05, 0, yR + 0.1, zc + 0.1);
    }
    if (tr.carrier) {
      const len = Math.min(1.5, GS.wsTop - GS.rearTop - 0.35), x = xr - 0.07, y = yR + 0.09;
      for (const s of [-1, 1]) {                                       // side bars; feet where there are no rails to stand on
        box(M, 'extra:trim', 0.03, 0.03, len, s * x, y, zc);
        if (!d.rails) for (const e of [-1, 1]) box(M, 'extra:trim', 0.035, 0.1, 0.035, s * x, y - 0.05, zc + e * (len / 2 - 0.1));
      }
      for (const e of [-1, 0, 1]) box(M, 'extra:trim', 2 * x, 0.025, 0.04, 0, y, zc + e * (len / 2 - 0.1));
      if (tr.carrier === 'ladder') {
        for (const s of [-1, 1]) box(M, 'extra:hub', 0.03, 0.04, len + 0.25, 0.22 + s * 0.17, y + 0.035, zc - 0.05);
        for (let i = 0; i < 6; i++) box(M, 'extra:hub', 0.34, 0.02, 0.03, 0.22, y + 0.035, zc - 0.05 + (i - 2.5) * (len + 0.1) / 6);
      }
    }
  }
  return { LS, GS, SL, SG, edge };
}

// ------------------------------------------------------------------------------------------ the cars
function loftCar(type, d, st, L, W, H, face, tr) {
  const hl = L / 2, hw = W / 2, M = mesher(), T = mesher();        // T: the tail and stop lamps (their own mesh)
  const { GS, SL, SG } = loftBody(M, d, st, W, tr);
  const plateKey = L < 3.6 ? 'plateY' : 'plateW', plateNo = tr.plate;
  frontFace({ M, nose: SL.nose, W, gc: d.gc, yHood: d.body[4][1], gh: d.gh, clad: d.clad, r: st.rF, plateKey, plateNo, mark: tr.mark }, face);
  rearFace({ M, T, tail: SL.tail, gtail: SG.tail, W, gc: d.gc, yT: d.body[d.body.length - 2][1], yE: GS.yE, belt: GS.belt, hatch: d.hatch, clad: d.clad, rR: st.rR, plateKey, plateNo, mark: tr.mark }, face);
  // fit the body to its box: the corrections are small (the numbers above already aim at L x W x H)
  const fit = fitTo(M, T, { hw, top: H, z0: -hl, z1: hl });
  return finish(type, d, M, T, hw, fit, face, tr);
}

/** 軽トラ: lofted cab; flat bed with drop sides, guard frame behind the cab, frame below (as cars.js). */
function loftKeiTruck(type, d, st, L, W, H, face, tr) {
  const hl = L / 2, hw = W / 2, M = mesher(), T = mesher();
  const { SL } = loftBody(M, d, st, W, tr);
  cabFace({ M, nose: SL.nose, W, kei: true, r: st.rF, plateKey: 'plateY', plateNo: tr.plate }, face);
  fitTo(M, T, { hw, top: d.top, z0: d.c0, z1: hl });
  const fy = 0.64, axR = d.wheels[1];
  const bed1 = d.c0 - 0.06, bedL = bed1 + hl, bz = (bed1 - hl) / 2;
  box(M, 'body', W - 0.02, 0.05, bedL, 0, fy, bz);
  for (const s of [-1, 1]) box(M, 'body', 0.035, 0.3, bedL, s * (hw - 0.0175), fy + 0.175, bz);
  box(M, 'body', W, 0.3, 0.035, 0, fy + 0.175, -hl + 0.0275);
  box(M, 'body', W, 0.3, 0.035, 0, fy + 0.175, bed1 - 0.0175);
  for (const s of [-1, 1]) {
    box(M, 'trim', 0.05, 0.08, 0.02, s * (hw - 0.07), fy + 0.25, -hl + 0.01);              // gate latches
    box(M, 'trim', 0.22, 0.2, 0.01, s * (hw - 0.15), 0.36, axR - d.wr - 0.12);             // mud flaps
  }
  box(M, 'trim', W * 0.62, 0.14, bedL + 0.24, 0, 0.5, bz + 0.18);
  const gz = bed1 - 0.03, gTop = H;                                                         // guard frame (鳥居)
  for (const s of [-1, 1]) box(M, 'trim', 0.05, gTop - fy, 0.05, s * (hw - 0.05), (gTop + fy) / 2, gz);
  box(M, 'trim', W - 0.04, 0.05, 0.07, 0, gTop - 0.025, gz);
  for (let i = 1; i <= 3; i++) box(M, 'trim', W - 0.14, 0.02, 0.02, 0, fy + 0.3 + i * (gTop - fy - 0.3) / 4, gz);
  box(M, 'trim', W - 0.06, 0.09, 0.06, 0, 0.44, -hl + 0.05);
  truckTails({ push: (g) => T.geometry('tail', g) }, face.tail, hw - 0.16, 0.5, -hl, false);
  plateAt(M, 'plateY', 0.5, -hl + 0.008, 0, -1, tr.plate);
  const fit = fitTo(M, T, { hw, top: H, z0: -hl, z1: hl });
  return finish(type, d, M, T, hw, fit, face, tr);
}

/** 2 t truck: lofted cab and frame (as cars.js); behind the cab a box ('truck'), a box with a refrigeration unit
 *  ('reefer'), a box with a band in the cab's colour ('delivery') or a flat bed with drop sides and a load ('flatbed'). */
function loftTruck(type, d, st, L, W, H, face, tr) {
  const hl = L / 2, hw = W / 2, M = mesher(), T = mesher();
  const { SL } = loftBody(M, d, st, W, tr);
  cabFace({ M, nose: SL.nose, W, kei: false, r: st.rF, plateKey: 'plateG', plateNo: tr.plate }, face);
  fitTo(M, T, { hw, top: d.top, z0: d.c0, z1: hl });
  const zb0 = -hl + 0.03, zb1 = d.c0 - 0.12, yb = 0.6, ym = yb + (H - yb) / 2, zm = (zb0 + zb1) / 2, bl = zb1 - zb0;
  box(M, 'trim', W * 0.46, 0.25, L - 0.4, 0, 0.42, -0.1);                                   // frame, between the wheels
  const g0 = d.wheels[1] + d.wr + 0.15, g1 = d.wheels[0] - d.wr - 0.15;                     // side guards and a tank under the box
  for (const s of [-1, 1]) for (const y of [0.42, 0.54]) box(M, 'trim', 0.03, 0.05, g1 - g0, s * (hw - 0.04), y, (g0 + g1) / 2);
  box(M, 'hub', 0.34, 0.3, 0.7, hw - 0.3, 0.44, g0 + 0.5);
  if (type === 'flatbed') {
    const fy = 0.98, gate = 0.36;
    box(M, 'cargo', W, 0.1, bl, 0, fy - 0.05, zm);                                           // the bed and its drop sides
    for (const s of [-1, 1]) box(M, 'cargo', 0.04, gate, bl, s * (hw - 0.02), fy + gate / 2, zm);
    box(M, 'cargo', W, gate, 0.04, 0, fy + gate / 2, zb0 + 0.02);
    for (const s of [-1, 1]) for (const e of [0.25, 0.5, 0.75]) box(M, 'trim', 0.012, gate, 0.05, s * (hw + 0.0), fy + gate / 2, zb0 + bl * e);   // gate posts
    // the guard frame behind the cab: the highest thing on the truck
    for (const s of [-1, 1]) box(M, 'trim', 0.07, H - fy, 0.07, s * (hw - 0.035), (H + fy) / 2, zb1 - 0.035);
    box(M, 'trim', W, 0.07, 0.07, 0, H - 0.035, zb1 - 0.035);
    for (let i = 1; i <= 4; i++) box(M, 'trim', W - 0.14, 0.03, 0.03, 0, fy + gate + i * (H - fy - gate) / 5, zb1 - 0.035);
    if (tr.load === 1) {                                                                     // building materials: pipes and boards
      for (let i = 0; i < 4; i++) { const g = new THREE.CylinderGeometry(0.07, 0.07, bl - 0.5, 8); g.rotateX(Math.PI / 2); g.translate(-0.3 + i * 0.16, fy + 0.07 + (i % 2) * 0.12, zm - 0.1); M.geometry('hub', g); }
      box(M, 'body', W * 0.36, 0.3, bl * 0.4, hw * 0.45, fy + 0.15, zm + bl * 0.15);
      box(M, 'cargo', W * 0.34, 0.18, bl * 0.3, hw * 0.45, fy + 0.39, zm + bl * 0.15);
    } else if (tr.load === 2) {                                                              // a load under a sheet
      const g = new THREE.BoxGeometry(W - 0.2, 0.85, bl - 0.5, 1, 2, 3);
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) if (p.getY(i) > 0.2) { p.setX(i, p.getX(i) * 0.8); p.setZ(i, p.getZ(i) * 0.86); p.setY(i, p.getY(i) + 0.06 * Math.sin(p.getZ(i) * 2.3)); }
      g.computeVertexNormals(); g.translate(0, fy + 0.425, zm - 0.1);
      M.geometry('plateG', g);
    }
  } else {
    const wb = type === 'delivery' ? W - 0.012 : W;
    box(M, 'cargo', wb, H - yb, bl, 0, ym, zm);
    box(M, 'trim', 0.012, H - yb - 0.15, 0.01, 0, ym, zb0 - 0.004);                          // rear doors and lock bars
    for (const s of [-1, 1]) box(M, 'hub', 0.022, H - yb - 0.3, 0.025, s * 0.28, ym, zb0 - 0.012);
    if (type === 'reefer') {                                                                 // the refrigeration unit over the cab
      box(M, 'hub', W * 0.7, 0.5, 0.3, 0, H - 0.35, zb1 + 0.14);
      box(M, 'trim', W * 0.5, 0.3, 0.02, 0, H - 0.35, zb1 + 0.3);
    } else if (type === 'delivery') {                                                        // a band in the cab's colour along the box
      for (const s of [-1, 1]) box(M, 'body', 0.006, (H - yb) * 0.22, bl - 0.04, s * (hw - 0.003), yb + (H - yb) * 0.42, zm);
      box(M, 'body', wb - 0.3, (H - yb) * 0.22, 0.006, 0, yb + (H - yb) * 0.42, zb1 + 0.002);
    }
  }
  box(M, 'trim', W - 0.2, 0.1, 0.08, 0, 0.45, -hl + 0.1);                                   // underrun bar
  truckTails({ push: (g) => T.geometry('tail', g) }, face.tail, hw - 0.2, 0.62, -hl, true);
  plateAt(M, 'plateG', 0.5, -hl + 0.008, 0, -1, tr.plate);
  const fit = fitTo(M, T, { hw, top: H, z0: -hl, z1: hl });
  return finish(type, d, M, T, hw, fit, face, tr);
}

function finish(type, d, M, T, hw, fit, face, tr) {
  // wheels (after the fit: they stand on the ground as they are)
  for (const s of [-1, 1]) for (const z of d.wheels) wheel(M, s, s * (hw - d.tx), z, d.wr, d.tw, tr.wheel);
  // what may reach out of the box (door mirrors; later roof signs and carriers) is marked: [geometry, key, { extra }]
  const parts = M.parts().map(([g, k]) => (k.startsWith('extra:') ? [g, k.slice(6), { extra: true }] : [g, k]));
  const tails = T.parts().map(([g]) => g);
  return { type, base: tr.base, face: face.key, trim: tr.key, parts, tails, wheels: [], lamps: [], lofted: true, fit, triangles: M.triangles() + T.triangles() };
}

/**
 * Detailed car of the redesign: { type, face, parts: [[geometry, materialKey, { extra }?], ...], tails: [geometry...],
 * wheels, lamps } with the material keys of cars.js (extra: a part that may reach out of the L x W x H box, like a
 * door mirror; a roof lamp, sign or carrier). id varies what KeiSim's size and colour leave open: the body type within
 * the size class, the face, wheels, paint, plate, and what the car is used for (opt forces them: type, face, wheel, ...).
 */
export function buildCar2(kind, L, W, H, rgb, id = 0, opt = null) {
  const { base, type } = carType2(kind, L, W, H, rgb, id, opt && opt.type);
  const d = design(TRUCKS.includes(type) ? 'truck' : type, L, H), st = STYLE[type], h = carHash(id, L, W, H, rgb);
  const face = chooseFace(type, h, opt && opt.face), tr = chooseTrim(type, h, rgb, opt);
  tr.base = base;
  if (type === 'keitruck') return loftKeiTruck(type, d, st, L, W, H, face, tr);
  if (TRUCKS.includes(type)) return loftTruck(type, d, st, L, W, H, face, tr);
  return loftCar(type, d, st, L, W, H, face, tr);
}
