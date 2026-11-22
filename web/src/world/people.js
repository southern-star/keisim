// 歩行者 (detailed) — KeiSim's pedestrians built like the bicycle riders of bikes.js: tapered limbs, one vertex-coloured
// mesh per rigid body, legs and arms solved every frame from where the feet and hands should be. Stage P1 of the
// redesign (PedRedesign_Handoff.md): seven types of people (personType), each with its clothes, hair or hat, shoes,
// things to carry and (stage P2) its way of walking, all chosen from a hash of what KeiSim gives a pedestrian (id,
// height, shirt colour).
// The model faces local +Z (+X is its left) with its origin on the ground under the pelvis, like actors.js.
//
// The walk follows KeiSim's phase (keisim/traffic.py: 4.2 rad for every metre walked, so one cycle of two steps is
// 1.496 m). The left foot is forward when sin(phase) > 0, as in the plain pedestrian of actors.js. A foot on the
// ground moves back under the body exactly as fast as the body moves forward, so it does not slide: it lands on the
// heel, rolls flat, and leaves over the ball of the foot. People whose legs are too short for KeiSim's step (children)
// keep each foot on the ground for a shorter time and run, with both feet in the air between the steps.
// On top of that every type has its habits (HABITS: how much the hips bob, the arms swing, the trunk leans, the
// pelvis turns, the feet lift), and every person a little of their own. What a hand carries swings after it like a
// pendulum; a walking stick is planted with the other side's foot and stays where it stands; the hem of a skirt or
// coat is a skinned cone whose lower edge follows the legs.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { collector, twoBone, setBone } from './bikes.js';

export const PED_DETAIL_DIST = 45;            // m: the detailed pedestrians are for closer than this to the camera
export const STEP = 1 / 4.2;                  // metres walked per radian of phase (keisim/traffic.py)
const TAU = Math.PI * 2;
const D_WALK = 0.56;                          // share of the cycle a walking foot is on the ground
const TOE_UP = 0.30, HEEL_UP = 0.62;          // rad: the foot's pitch at heel strike and (at least) at toe-off
const U_FLAT = 0.14, U_HEEL = 0.50;           // stance progress where the foot is flat / where the heel starts to rise
const STAND_S = 0.3, WALK_S = 0.2;            // seconds to settle into the standing pose / to leave it
const HEADS = 7.46;                           // a person of head size 1 is this many heads tall
const CANE_D = 0.36;                          // share of the cycle a walking stick stands on the ground
const ITEM_LAG = 0.7;                         // rad of phase a thing in the hand swings behind the arm
const ITEM_SWING = { plastic: 1.1, tote: 0.8, school: 0.55, brief: 0.45, umbrella: 0.4 };   // how far it swings, against the arm's own swing
const SHOULDER_Y = 0.075;                     // the shoulder joints are this far (x s) under the top of the trunk

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (t) => { const x = clamp(t, 0, 1); return x * x * (3 - 2 * x); };
const wrap = (a) => a - TAU * Math.floor((a + Math.PI) / TAU);      // to [-pi, pi)

/** Stable 32-bit hash of what KeiSim gives a pedestrian: every choice of the model comes from it. */
export function personHash(id, h, shirt) {
  let x = (Math.imul((id | 0) + 1, 0x9E3779B1) ^ Math.imul(Math.round(h * 100), 0x85EBCA6B) ^ Math.imul(shirt.getHex() + 1, 0xC2B2AE35)) >>> 0;
  x ^= x >>> 15; x = Math.imul(x, 0x2C1B3C6D); x ^= x >>> 12; x = Math.imul(x, 0x297A2D39); x ^= x >>> 15;
  return x >>> 0;
}
/** The numbers 0 .. 1 a person's choices are drawn from, in a fixed order, from the hash. */
function stream(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---- the types (PedRedesign_Handoff.md 4.4): [name, shortest, tallest, weight]. Under 1.40 m everybody is a child.
export const PERSON_TYPES = ['kid', 'student', 'office', 'shopper', 'elder', 'worker', 'casual'];
export const TYPE_BANDS = [['student', 1.50, 1.72, 15], ['office', 1.55, 1.85, 20], ['shopper', 0, 9, 15], ['elder', 1.50, 1.68, 15],
  ['worker', 1.60, 1.85, 10], ['casual', 0, 9, 25]];
export const KID_BELOW = 1.40;
function typeAt(h, u) {
  if (h < KID_BELOW) return 'kid';
  const c = TYPE_BANDS.filter(([, lo, hi]) => h >= lo - 1e-9 && h <= hi + 1e-9);
  let x = u * c.reduce((n, b) => n + b[3], 0);
  for (const b of c) { x -= b[3]; if (x < 0) return b[0]; }
  return c[c.length - 1][0];
}

// ---- colours that do not come from KeiSim
const HAIR = ['#2b2630', '#3d2f2a', '#5a4130', '#1f1d24', '#6b5845'], GREY = ['#8c8c90', '#b6b6ba', '#d6d6d6', '#5f5b5d'];
const WHITE = '#f1f1ec', YELLOW = '#f0d020', DARK = '#22242a';
const SUIT = ['#1f2a44', '#2e3138', '#17181c', '#3a3f4b'], COAT = ['#b9a583', '#1f2a44', '#2a2b30', '#6b5a4a', '#7c8088'];
const BLAZER = ['#1c2540', '#2b2f3a', '#3b2a30'], KNIT = ['#1c2540', '#c9b993', '#7c8088'];
const PLEAT = ['#3a4050', '#1c2540', '#4a3d52', '#2f4a3a'], SLACKS = ['#3a4050', '#1c2540', '#2a2b30', '#5a5f6a'];
const WORK = ['#5b7a99', '#b8a989', '#2f3f5c', '#4f6b55', '#8a8f96'], UNIFORM = ['#e9e7de', '#3d5a80', '#4a4d55', '#5f7f5a', '#c9d3dc'];
const HIVIS = ['#d6e23a', '#f08a24'], HELMET = ['#f2f2ee', '#f0d040', '#f2f2ee'], SILVER = '#e2e4e8';
const VEST = ['#9a8f78', '#6f7a66', '#7d6f84', '#8b6b5a', '#5d6b7c'], INNER = ['#f1f1ec', '#d9d4c7', '#c6d2df', '#2c2d33', '#e3c9c0'];
const HATS = ['#2c2d33', '#e8e2d0', '#3a4a66', '#8a2f3a', '#6f7a66', '#c8b89a'];
const BAG = ['#2a2c33', '#5a4636', '#3a4a66', '#8a2f3a', '#c8b89a', '#3f5f4a'], BRIEF = ['#1f2024', '#3c2a20', '#2a3040'];
const RANDO = ['#b02a30', '#1c1d22', '#2a4f9a', '#8a5a3a', '#d98aa0', '#6aa0c8'], ECO = ['#e8e2d0', '#6f9a78', '#d98a4a', '#5a7fb0', '#c85a6a'];
const PLASTIC = '#eceae4', CARDBOARD = '#c7a678', TAPE = '#a88a5e', UMBRELLA = ['#1f2024', '#2a3f6a', '#e8e8e8', '#7a2a35', '#3f6a55'];
const CANE = ['#6b4a2e', '#2a2a2e', '#b9bcc2'];
const SNEAKER = ['#e8e6e0', '#2c2d33', '#3a4a66', '#b04040', '#7a7d84'], LEATHER = ['#1e1a18', '#4a3426', '#2a2422'], BOOT = ['#3a2c22', '#1f1f24', '#6b5238'];
const TIGHTS = ['#26262c', '#3a3a44'];
const hex = (c) => (c ? new THREE.Color(c).getHexString() : '-');
const shade = (c, k) => new THREE.Color(c).multiplyScalar(k);

/**
 * What a person looks like: the type and every part, drawn from the hash. cols: [shirt, pants, skin] of KeiSim.
 * Children, shoppers, elderly and casual people wear KeiSim's shirt colour on top and its pants colour below;
 * students, office workers and workers wear the colours of their type, with the shirt colour as an accent (tie,
 * ribbon, stripe on the bag, cap or helmet band). Sides: +1 the person's left, -1 their right.
 */
function lookOf(h, cols, hash) {
  const [shirt, pants, skin] = cols;
  const R = stream(hash), type = typeAt(h, R());
  const of = (a) => a[Math.floor(R() * a.length) % a.length];
  const chance = (p) => R() < p;
  const pick = (o) => { const ks = Object.keys(o); let u = R() * ks.reduce((n, k) => n + o[k], 0); for (const k of ks) { u -= o[k]; if (u < 0) return k; } return ks[ks.length - 1]; };
  const L = { type, heads: 7.2, kS: 1, kG: 1, loose: 1, hair: 'short', hairCol: of(HAIR), hat: null, hatCol: null, band: null, glasses: false, mask: false,
    phones: false, cord: false, top: 'tee', topCol: shirt, armCol: null, inner: null, tie: null, ribbon: null, collar: null, hood: false,
    bands: false, stripe: null, flap: 0.06, bottom: 'long', botCol: pants, legCol: skin, socks: null, shoes: 'sneaker', shoeCol: SNEAKER[0],
    back: null, backCol: null, sling: null, hand: null, phone: 0, box: false };
  // heads tall: children 5 .. 6, grown-ups 7 .. 7.5; the three measures of the build
  L.heads = h < KID_BELOW ? clamp(lerp(5.4, 6.0, (h - 1.0) / 0.35) + 0.16 * (R() - 0.5), 5.05, 6) : h < 1.5 ? lerp(6.2, 6.9, (h - 1.4) / 0.1) : 7 + 0.5 * R();
  L.kS = 0.96 + 0.1 * R(); L.kG = 0.94 + 0.14 * R();
  const mainSide = chance(0.72) ? -1 : 1;
  const anyHair = { short: 34, bob: 18, long: 16, pony: 18, bun: 14 };
  let carry = 'none';
  switch (type) {
    case 'kid': {                                            // short sleeves, shorts or a skirt, school satchel, yellow hat
      L.kS *= 0.93;
      L.hair = pick({ short: 40, bob: 22, pony: 14, twin: 16, bun: 8 });
      L.bottom = chance(L.hair === 'short' ? 0.12 : 0.6) ? 'skirt' : 'shorts';
      const hat = pick({ none: 30, cap: 35, tulip: 35 });
      if (hat !== 'none') { L.hat = hat; L.hatCol = YELLOW; }
      if (chance(0.85)) { L.back = 'randoseru'; L.backCol = RANDO[L.bottom === 'skirt' ? pick({ 0: 50, 4: 20, 5: 15, 3: 15 }) : pick({ 1: 55, 2: 25, 3: 20 })]; }
      L.socks = WHITE; L.shoeCol = of(SNEAKER);
      break;
    }
    case 'student': {                                        // blazer or stand-up collar, skirt or slacks, school bag, phone
      L.hair = pick({ short: 38, bob: 22, long: 14, pony: 22, bun: 4 });
      const gakuran = L.hair === 'short' && chance(0.45), summer = chance(0.25);
      if (gakuran) { L.top = 'jacket'; L.topCol = '#17181c'; L.collar = 'stand'; L.botCol = '#17181c'; L.flap = 0.1; }
      else {
        const skirt = chance(L.hair === 'short' ? 0.1 : 0.8);
        L.top = summer ? 'vest' : 'jacket'; L.topCol = summer ? of(KNIT) : of(BLAZER); L.armCol = summer ? WHITE : null; L.inner = WHITE; L.collar = WHITE;
        L.flap = summer ? 0.07 : 0.115;
        L.bottom = skirt ? 'skirt' : 'long'; L.botCol = skirt ? of(PLEAT) : of(SLACKS);
        if (skirt) { L.ribbon = shirt; L.legCol = chance(0.35) ? of(TIGHTS) : skin; L.socks = L.legCol === skin ? (chance(0.5) ? '#1c2540' : WHITE) : null; } else L.tie = shirt;
      }
      L.shoes = chance(0.6) ? 'leather' : 'sneaker'; L.shoeCol = of(L.shoes === 'leather' ? LEATHER : SNEAKER);
      carry = pick({ school: 40, sling: 35, rucksack: 15, none: 10 });
      if (chance(0.2)) L.phone = -mainSide;
      L.phones = chance(0.08);
      break;
    }
    case 'office': {                                         // suit, white shirt, the accent as a tie or scarf, business bag
      L.hair = pick({ short: 52, bob: 16, long: 10, pony: 12, bun: 10 });
      const suit = of(SUIT), skirt = L.hair !== 'short' && chance(0.5), coat = chance(0.15);
      L.top = coat ? 'coat' : 'jacket'; L.topCol = coat ? of(COAT) : suit; L.inner = of(['#f2f2ee', '#f2f2ee', '#cfdcec']); L.collar = L.inner; L.flap = 0.115;
      L.bottom = skirt && !coat ? 'skirt' : 'long'; L.botCol = suit;
      if (skirt) { if (chance(0.5)) L.ribbon = shirt; L.legCol = chance(0.5) ? of(TIGHTS) : skin; L.kS *= 0.97; } else { L.tie = shirt; L.kS *= 1.03; }
      L.shoes = 'leather'; L.shoeCol = of(LEATHER);
      carry = pick({ brief: 62, sling: 14, umbrella: 10, none: 14 });
      if (chance(0.12)) L.phone = -mainSide;
      L.glasses = chance(0.25);
      break;
    }
    case 'shopper': {                                        // everyday clothes, a shopping bag in one hand
      L.hair = pick(anyHair);
      L.top = pick({ shirt: 28, tee: 24, vest: 16, parka: 16, coat: 16 });
      L.bottom = pick({ long: 52, longskirt: 20, skirt: 16, shorts: 12 });
      if (L.top === 'vest') { L.armCol = of(INNER); L.inner = L.armCol; }
      const hat = pick({ none: 76, brim: 14, cap: 10 });
      if (hat !== 'none') { L.hat = hat; L.hatCol = of(HATS); }
      carry = chance(0.5) ? 'tote' : 'plastic';
      if (chance(0.22)) L.sling = { side: -mainSide, col: of(BAG) };
      L.shoes = pick({ sneaker: 60, leather: 15, boots: 25 });
      L.glasses = chance(0.15); L.mask = chance(0.12);
      break;
    }
    case 'elder': {                                          // a vest over the shirt, loose trousers, a stick or a hat
      L.hairCol = of(GREY); L.hair = pick({ short: 55, bob: 20, bun: 25 });
      if (chance(0.7)) { L.top = 'vest'; L.topCol = of(VEST); L.armCol = shirt; L.inner = shirt; } else L.top = 'shirt';
      L.collar = shirt; L.flap = 0.075;
      L.bottom = L.hair !== 'short' && chance(0.35) ? 'longskirt' : 'long'; L.loose = 1.16;
      const hat = pick({ none: 52, brim: 26, cap: 22 });
      if (hat !== 'none') { L.hat = hat; L.hatCol = of(HATS); }
      carry = pick({ cane: 45, tote: 15, sling: 15, none: 25 });
      L.kS *= 0.95; L.kG *= 1.08;
      L.shoes = chance(0.7) ? 'sneaker' : 'leather';
      L.glasses = chance(0.35); L.mask = chance(0.12);
      break;
    }
    case 'worker': {                                         // delivery: uniform, cap, a parcel; site: work clothes, safety vest, helmet
      L.hair = pick({ short: 62, bob: 10, pony: 22, bun: 6 });
      if (chance(0.45)) {
        L.top = chance(0.5) ? 'tee' : 'shirt'; L.topCol = of(UNIFORM); L.stripe = shirt; L.collar = L.top === 'shirt' ? L.topCol : null;
        L.botCol = of(SLACKS); L.flap = 0.05;
        if (chance(0.85)) { L.hat = 'cap'; L.hatCol = shirt; }
        carry = pick({ box: 60, tote: 15, none: 25 });
        L.shoes = 'sneaker';
      } else {
        const wc = of(WORK);
        L.top = 'shirt'; L.topCol = wc; L.botCol = chance(0.7) ? wc : of(WORK); L.loose = 1.1; L.collar = wc; L.flap = 0.07;
        if (chance(0.7)) { L.top = 'vest'; L.topCol = of(HIVIS); L.armCol = wc; L.inner = wc; L.bands = true; }
        if (chance(0.75)) { L.hat = 'helmet'; L.hatCol = of(HELMET); L.band = shirt; } else { L.hat = 'cap'; L.hatCol = shirt; }
        carry = pick({ none: 62, tote: 23, box: 15 });
        L.shoes = 'boots'; L.kS *= 1.04;
      }
      L.glasses = chance(0.1); L.mask = chance(0.1);
      break;
    }
    default: {                                               // casual: hoodie, shirt, T-shirt, coat, dress, shorts
      L.hair = pick(anyHair);
      const top = pick({ parka: 25, shirt: 20, tee: 20, coat: 15, dress: 10, vest: 10 });
      L.bottom = pick({ long: 60, shorts: 14, skirt: 13, longskirt: 13 });
      if (top === 'dress') { L.top = chance(0.5) ? 'tee' : 'shirt'; L.bottom = chance(0.5) ? 'skirt' : 'longskirt'; L.botCol = shirt; L.flap = 0; }
      else L.top = top;
      if (L.top === 'vest') { L.armCol = of(INNER); L.inner = null; if (L.bottom === 'shorts') L.bottom = 'long'; }
      if (L.top === 'parka') { L.hood = true; L.flap = 0.085; if (chance(0.2)) { L.hat = 'hood'; L.hatCol = shirt; } }
      if (!L.hat) { const hat = pick({ none: 82, cap: 13, brim: 5 }); if (hat !== 'none') { L.hat = hat; L.hatCol = of(HATS); } }
      carry = pick({ rucksack: 28, sling: 15, phone: 14, umbrella: 5, none: 38 });
      L.phones = chance(0.15); L.cord = chance(0.4);
      L.shoes = pick({ sneaker: 65, leather: 10, boots: 25 });
      L.glasses = chance(0.15); L.mask = chance(0.1);
    }
  }
  // the thing carried
  const bagCol = of(BAG), thingCol = R();
  const from = (a) => a[Math.floor(thingCol * a.length) % a.length];
  if (carry === 'school') L.hand = { kind: 'school', side: mainSide, col: from(['#1c2540', '#2a2c33', '#3b2a30']), accent: shirt };
  else if (carry === 'brief') L.hand = { kind: 'brief', side: mainSide, col: from(BRIEF) };
  else if (carry === 'tote') L.hand = { kind: 'tote', side: mainSide, col: type === 'worker' ? from(['#3a4a66', '#4a4d55', '#8a2f3a']) : from(ECO) };
  else if (carry === 'plastic') L.hand = { kind: 'plastic', side: mainSide, col: PLASTIC };
  else if (carry === 'cane') L.hand = { kind: 'cane', side: mainSide, col: from(CANE) };
  else if (carry === 'umbrella') L.hand = { kind: 'umbrella', side: mainSide, col: from(UMBRELLA) };
  else if (carry === 'sling') L.sling = { side: mainSide, col: bagCol };
  else if (carry === 'rucksack') { L.back = 'rucksack'; L.backCol = bagCol; }
  else if (carry === 'phone') L.phone = mainSide;
  else if (carry === 'box') L.box = true;
  // what does not go together
  if (L.top === 'coat') { L.flap = 0; if (L.bottom !== 'long') L.bottom = 'long'; if (!L.collar) L.collar = L.topCol; }
  if (L.top === 'shirt' && !L.collar) L.collar = L.topCol;
  if (L.bottom === 'skirt' || L.bottom === 'longskirt') { if (L.legCol === skin && type !== 'kid' && type !== 'student' && chance(0.3)) L.legCol = of(TIGHTS); }
  else if (L.bottom === 'long') { L.legCol = L.botCol; L.socks = null; }
  if (L.shoes === 'boots') { L.shoeCol = of(BOOT); L.socks = null; } else if (L.shoes === 'leather') L.shoeCol = of(LEATHER); else if (type !== 'kid') L.shoeCol = of(SNEAKER);
  if (L.hat && L.hat !== 'cap' && (L.hair === 'bun' || L.hair === 'twin')) L.hair = 'bob';
  if (L.hat === 'cap' && L.hair === 'bun') L.hair = 'pony';
  if (L.hat && L.hat !== 'cap') L.phones = false;
  if (!L.phones || (L.hand && (L.hand.kind === 'cane' || L.hand.kind === 'umbrella'))) L.cord = false;
  if (L.box) { L.phone = 0; L.sling = null; }
  if (L.phone && L.hand && L.hand.side === L.phone) L.phone = -L.phone;
  if (L.back && L.hair === 'long') L.hair = 'pony';
  L.headgear = L.hat || L.hair;
  L.things = [L.back, L.sling && 'sling', L.hand && L.hand.kind, L.phone && 'phone', L.box && 'box', L.phones && 'headphones'].filter(Boolean);
  // everything that makes two people look different (not their height)
  L.key = [type, L.hair, hex(L.hairCol), L.hat, hex(L.hatCol), hex(L.band), +L.glasses, +L.mask, +L.phones, +L.cord, L.top, hex(L.topCol), hex(L.armCol),
    hex(L.inner), hex(L.tie), hex(L.ribbon), hex(L.collar === 'stand' ? null : L.collar), +L.hood, +L.bands, hex(L.stripe), L.bottom, hex(L.botCol), hex(L.legCol),
    hex(L.socks), L.shoes, hex(L.shoeCol), L.back, hex(L.backCol), L.sling ? L.sling.side + hex(L.sling.col) : '-', L.hand ? L.hand.kind + L.hand.side + hex(L.hand.col) : '-',
    L.phone, +L.box, hex(skin)].join('|');
  return L;
}
/** The look of the person makePerson builds from the same inputs (type, parts, key: see lookOf). */
export function personLook(h, cols, id) { return lookOf(h, cols, personHash(id, h, cols[0])); }
export function personType(h, cols, id) { return typeAt(h, stream(personHash(id, h, cols[0]))()); }

/** A person's measures: the limb lengths of bikes.js riderFit (thigh = shin 0.245 h, trunk 0.30 h, upper arm 0.172 h,
 *  forearm 0.165 h), the foot (the ankle above the sole; heel, ball and toe along it) and the hip joints' height when
 *  standing. s scales the widths (1 at 1.70 m), as in the rider. kH: the head's size (1: 7.46 heads tall); a bigger
 *  head takes its extra height from the trunk, so that the person stays h tall. kS: the shoulders' width. */
export function personFit(h, kH = 1, kS = 1) {
  const leg = 0.245 * h, s = 0.30 * h / 0.51, lt = 0.30 * h - 0.23 * s * (kH - 1), ankle = 0.05 * h;
  return { h, leg, lt, s, hs: s * kH, headH: 0.228 * s * kH, upper: 0.172 * h, fore: 0.165 * h, ankle, heel: 0.035 * h, ball: 0.085 * h, toe: 0.11 * h,
    hipW: 0.055 * h, track: 0.05 * h, shoulderW: 0.115 * h * kS, stand: ankle + 0.995 * 2 * leg };
}

/** The walk that fits a person's legs: D the share of the cycle a foot is on the ground, A half the way it travels
 *  under the body there, delta where that way is centred, H0 / bob the hip joints' height, heelUp the foot's pitch at
 *  toe-off, tw the pelvis' turn, run 0 (walk) .. 1 (run). hb: the person's habits (habitOf): bob, tw and crouch change
 *  where the hips are, so the walk is fitted with them.
 *  KeiSim's step (0.748 m) is the same for everybody. Long legs walk it easily (D 0.56). Shorter ones first roll
 *  further over the toes, turn the pelvis more and carry the hips lower (effort 0 .. 1) to keep D at 0.52; legs
 *  too short even for that keep their feet on the ground for less than half the cycle: they run. */
export function gaitOf(fit, hb = null) {
  const { h, leg, ankle, heel, ball, stand, hipW } = fit;
  const kBob = hb ? hb.bob : 1, kTw = hb ? hb.tw : 1, crouch = hb ? hb.crouch : 0;
  const reach = 0.99 * 2 * leg, half = Math.PI * STEP;                // the leg at full stretch, one step (0.748 m)
  // the ankle at heel strike: its height, and how far it is from where the flat foot's ankle would be
  const yS = heel * Math.sin(TOE_UP) + ankle * Math.cos(TOE_UP), cF = heel * (1 - Math.cos(TOE_UP)) + ankle * Math.sin(TOE_UP);
  function at(effort, run, D) {
    const heelUp = HEEL_UP + 0.24 * effort, tw = (0.10 + 0.08 * effort) * kTw;
    const yO = ball * Math.sin(heelUp) + ankle * Math.cos(heelUp), cR = ball * (1 - Math.cos(heelUp)) + ankle * Math.sin(heelUp);   // at toe-off
    const bob = 0.012 * h * (1 - 2 * run) * kBob;                      // walking: highest over the standing leg; running: lowest
    // (the pelvis lifts the standing side 0.002 h; hips that bob more are carried lower, so that the leg still reaches)
    const H0 = stand - (0.004 + 0.021 * effort + 0.03 * run + crouch) * h - 0.012 * h * Math.max(1, kBob);
    // how far the ankle reaches in front of the hips when the foot lands (Rf) and behind them when it leaves (Rr).
    // The hips are highest (running: lowest) when the standing foot is under them, which is `lead` rad of phase
    // before the middle of its time on the ground, because that time is centred behind the hips (delta < 0).
    let delta = 0, lead = 0, Rf = 0, Rr = 0;
    for (let it = 0; it < 4; it++) {
      lead = delta / STEP;
      const Hs = H0 + bob * Math.cos(2 * (D * Math.PI + lead)), Ho = H0 + bob * Math.cos(2 * (D * Math.PI - lead));
      Rf = Math.sqrt(Math.max(reach * reach - (Hs - yS) ** 2, 0)); Rr = Math.sqrt(Math.max(reach * reach - (Ho - yO) ** 2, 0));
      delta = (Rf + cF - cR - Rr) / 2;
    }
    return { heelUp, tw, bob, H0, yO, delta, lead: delta / STEP, Dmax: 0.97 * (Rf + Rr + cF + cR + 2 * hipW * Math.sin(tw)) / 2 / half };
  }
  let effort = 0, run = 0, D = Math.min(D_WALK, at(0, 0, D_WALK).Dmax);
  if (D < 0.52) {
    let lo = 0, hi = 1;
    for (let it = 0; it < 20; it++) { effort = (lo + hi) / 2; if (at(effort, 0, 0.52).Dmax < 0.52) lo = effort; else hi = effort; }
    effort = hi; D = Math.min(0.52, at(effort, 0, 0.52).Dmax);
    for (let it = 0; it < 12 && D < 0.5; it++) {                       // running: settle D with the hips that go with it
      run = clamp((0.5 - D) / 0.08, 0, 1);
      D = lerp(D, Math.min(0.5, at(1, run, D).Dmax), 0.6);
    }
    run = clamp((0.5 - D) / 0.08, 0, 1);
  }
  const g = at(effort, run, D);
  return { D, A: D * half, delta: g.delta, lead: g.lead, H0: g.H0, bob: g.bob, run, effort, heelUp: g.heelUp, tw: g.tw, yS, yO: g.yO };
}

// ---- ways of walking (PedRedesign_Handoff.md 4.3 / 4.4): factors on the walk's measures, by type
//   bob     the hips' rise and fall            arm    the free arms' swing (and the trunk's turn against the pelvis)
//   lean    rad the trunk leans forward (+)    tw     the pelvis' turn with the leg that swings forward
//   lift    how high the swinging foot lifts   reach  how far it swings out ahead before it lands (a long stride's look:
//           the step itself is KeiSim's, the same for everybody)
//   crouch  how much lower the hips are carried (x h)       drop   rad the shoulder of a hand with a load hangs lower
export const HABITS = {
  kid: { bob: 1.3, arm: 1.35, lean: 0, tw: 1.1, lift: 1.3, reach: 0, crouch: 0, drop: 0.02 },          // bouncing, arms swinging wide
  student: { bob: 1, arm: 1, lean: 0, tw: 1, lift: 1, reach: 0, crouch: 0, drop: 0.02 },
  office: { bob: 0.9, arm: 1.12, lean: -0.022, tw: 1.25, lift: 0.95, reach: 1, crouch: 0, drop: 0.015 },  // long strides, upright
  shopper: { bob: 1, arm: 0.95, lean: 0.005, tw: 0.95, lift: 1, reach: 0, crouch: 0, drop: 0.055 },       // the shoulder of the hand with the bag hangs
  elder: { bob: 0.55, arm: 0.5, lean: 0.11, tw: 0.7, lift: 0.6, reach: 0, crouch: 0.005, drop: 0.02 },    // stooped, little bob, arms quiet
  worker: { bob: 1, arm: 1, lean: 0, tw: 1, lift: 1, reach: 0, crouch: 0, drop: 0.02 },
  casual: { bob: 1, arm: 1, lean: 0, tw: 1, lift: 1, reach: 0, crouch: 0, drop: 0.02 },
};
/** A person's habits: their type's, with a little of their own (bob, arm and tw within 10 %, the lean within 0.6 deg). */
function habitOf(type, hash) {
  const R = stream((hash ^ 0xA53C9E17) >>> 0), b = HABITS[type], v = () => 0.9 + 0.2 * R();
  return { ...b, bob: b.bob * v(), arm: b.arm * v(), tw: b.tw * v(), lean: b.lean + 0.02 * (R() - 0.5), lift: b.lift * (0.95 + 0.1 * R()) };
}
/** The habits of the person makePerson builds from the same inputs. */
export function personHabit(h, cols, id) { const hash = personHash(id, h, cols[0]); return habitOf(typeAt(h, stream(hash)()), hash); }

// ---- shapes
const ell = (rx, ry, rz, x, y, z, w = 12, hh = 8) => new THREE.SphereGeometry(1, w, hh).scale(rx, ry, rz).translate(x, y, z);
const ball = (r, y = 0, w = 8, hh = 5) => new THREE.SphereGeometry(r, w, hh).translate(0, y, 0);
/** An open tube along +Y from y0: radius r0 at its start, r1 at its end. */
const tube = (len, r0, r1, y0 = 0, seg = 8) => new THREE.CylinderGeometry(r1, r0, len, seg, 1, true).translate(0, y0 + len / 2, 0);
const cyl = (r0, r1, len, seg, x, y, z) => new THREE.CylinderGeometry(r1, r0, len, seg).translate(x, y, z);
const box = (w, hh, d, x = 0, y = 0, z = 0) => new THREE.BoxGeometry(w, hh, d).translate(x, y, z);
const rbox = (w, hh, d, r, x = 0, y = 0, z = 0) => new RoundedBoxGeometry(w, hh, d, 1, r).translate(x, y, z);
/** A sheet of (nu + 1) x (nv + 1) points f(u, v) -> [x, y, z] with smooth normals, facing away from the Y axis.
 *  loop: u runs once round and the two ends are one seam. */
function sheet(nu, nv, f, loop = false) {
  const pos = [], idx = [];
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) pos.push(...f(i / nu, j / nv));
  const at = (i, j) => j * (nu + 1) + i;
  const p = (k) => [pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]];
  // which way round: the first cell's normal against the direction away from the axis
  const a = p(at(0, 0)), b = p(at(1, 0)), c = p(at(0, 1));
  const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const flip = nx * (a[0] + b[0] + c[0]) + nz * (a[2] + b[2] + c[2]) < 0;
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const q = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
    if (flip) idx.push(q[0], q[2], q[1], q[0], q[3], q[2]); else idx.push(q[0], q[1], q[2], q[0], q[2], q[3]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  if (loop) {
    const n = g.attributes.normal;
    for (let j = 0; j <= nv; j++) {
      const i0 = at(0, j), i1 = at(nu, j);
      const x = n.getX(i0) + n.getX(i1), y = n.getY(i0) + n.getY(i1), z = n.getZ(i0) + n.getZ(i1), l = Math.hypot(x, y, z) || 1;
      n.setXYZ(i0, x / l, y / l, z / l); n.setXYZ(i1, x / l, y / l, z / l);
    }
  }
  return g;
}

/**
 * One pedestrian. cols: [shirt, pants, skin] as THREE.Color; mat: a white vertex-coloured material (the riders' one).
 * Returns { g, fine, pose, type, look, habit, fit, gait, bodies }: g faces local +Z with its origin on the ground; fine:
 * meshes of thin parts (walking stick, the ends of an umbrella, a headphone cord) without outlines, which the caller
 * hides far from the camera; type: one of PERSON_TYPES; pose({ phase, moving, dt, stand }) moves the bodies:
 *   phase   KeiSim's walk phase (rad)
 *   moving  false while the person stands; left out: standing when the phase is the one of the call before
 *   dt      seconds since the call before (left out: 0.1 s), for settling into and out of the standing pose
 *   stand   true / false: the standing / walking pose at once (stills)
 * The first call has nothing to compare with and gives the walking pose.
 */
export function makePerson(ctx, mat, h, cols, id) {
  const [shirt, , skin] = cols;
  const hash = personHash(id, h, shirt), L = lookOf(h, cols, hash);
  const hb = habitOf(L.type, hash), fit = personFit(h, HEADS / L.heads, L.kS), gait = gaitOf(fit, hb);
  const { s, hs, leg, lt, upper, fore, ankle, heel, toe } = fit, kS = L.kS, kG = L.kG;
  const P = collector(mat), fine = [];
  const long = L.top !== 'tee', thick = L.top === 'jacket' || L.top === 'coat' || L.top === 'parka';
  const puff = thick ? 1.05 : L.top === 'vest' ? 1.03 : 1, armCol = L.armCol || L.topCol;
  const skirt = L.bottom === 'skirt' || L.bottom === 'longskirt', coat = L.top === 'coat';

  // ---- trunk: a tube of oval section under the shoulders' ellipsoid (its lower end is inside the pelvis)
  const T = lt - 0.07 * s, rT = 0.158 * s * kS * puff, rB = 0.135 * s * kG * puff, zs = 0.62 * (0.6 + 0.4 * kG);
  const shE = [0.2155 * s * kS * puff, 0.085 * s, 0.105 * s * puff, lt - 0.062 * s];      // (wide and high enough to meet the top of the arms' shoulder balls)
  P.add('torso', L.topCol, new THREE.CylinderGeometry(rT, rB, T, 12, 1, true).scale(1, 1, zs).translate(0, T / 2, 0), ell(shE[0], shE[1], shE[2], 0, shE[3], 0, 12, 6));
  P.add('torso', skin, new THREE.CylinderGeometry(0.042 * s, 0.045 * s, 0.1 * s, 8, 1, true).translate(0, lt + 0.01 * s, 0.005 * s));
  /** The point of the trunk's surface at angle a round it (0: front, +: towards the person's left) and height y, lifted off it. */
  function onTorso(a, y, lift = 0.008 * s) {
    const r = lerp(rB, rT, clamp(y / T, 0, 1));
    let x = r * Math.sin(a), z = r * zs * Math.cos(a);
    const e = 1 - ((y - shE[3]) / shE[1]) ** 2;             // near the top the shoulders' ellipsoid is outside the tube
    if (e > 0) { const k = Math.sqrt(e) / Math.hypot(x / shE[0], z / shE[2]); if (k > 1) { x *= k; z *= k; } }
    const d = Math.hypot(x, z) || 1;
    return [x * (1 + lift / d), y, z * (1 + lift / d)];
  }
  const band = (a0, a1, y0, y1, n, lift) => sheet(n, 1, (u, v) => onTorso(lerp(a0, a1, u), lerp(y0, y1, v), lift));
  const yNeck = lt - 0.04 * s;
  if (L.collar === 'stand') P.add('torso', L.topCol, new THREE.CylinderGeometry(0.05 * s, 0.054 * s, 0.05 * s, 8, 1, true).translate(0, lt + 0.014 * s, 0.005 * s));
  else if (L.collar) P.add('torso', L.collar, new THREE.CylinderGeometry(0.052 * s, 0.08 * s, 0.045 * s, 8, 1, true).translate(0, lt + 0.0125 * s, 0.004 * s));
  if (L.inner && L.top !== 'tee') {                          // the shirt in the opening of a jacket, coat or vest
    const deep = L.top === 'vest' ? 0.5 : 0.4;
    P.add('torso', L.inner, sheet(2, 3, (u, v) => onTorso((2 * u - 1) * lerp(0.03, 0.36, v), lerp(deep * T, yNeck, v), 0.007 * s)));
  }
  if (L.tie) P.add('torso', L.tie, sheet(1, 3, (u, v) => onTorso((2 * u - 1) * lerp(0.1, 0.055, v), lerp(0.36 * T, yNeck, v), 0.016 * s)));
  if (L.ribbon) P.add('torso', L.ribbon, band(-0.22, 0.22, lt - 0.135 * s, lt - 0.075 * s, 2, 0.018 * s));
  if (L.stripe) P.add('torso', L.stripe, band(-1.25, 1.25, 0.6 * T, 0.6 * T + 0.05 * s, 4, 0.006 * s));
  if (L.bands) {                                             // the reflective bands of a safety vest
    for (const yb of [0.2 * T, 0.5 * T]) P.add('torso', SILVER, sheet(12, 1, (u, v) => onTorso(TAU * u, yb + 0.045 * s * v, 0.006 * s), true));
    for (const a of [-0.5, 0.5, Math.PI - 0.5, Math.PI + 0.5]) P.add('torso', SILVER, sheet(1, 2, (u, v) => onTorso(a + 0.14 * (u - 0.5), lerp(0.5 * T + 0.045 * s, lt - 0.03 * s, v), 0.006 * s)));
  }
  if (L.top === 'parka') {
    P.add('torso', shade(L.topCol, 0.86), band(-0.55, 0.55, 0.05 * T, 0.26 * T, 3, 0.012 * s));                   // the pocket
    if (L.hat !== 'hood') P.add('torso', L.topCol, ell(0.108 * s, 0.062 * s, 0.07 * s, 0, lt - 0.03 * s, -0.1 * s, 8, 5));
    else P.add('torso', L.topCol, new THREE.CylinderGeometry(0.075 * s, 0.1 * s, 0.07 * s, 8, 1, true).translate(0, lt + 0.005 * s, -0.005 * s));
  }
  if (coat || (L.top === 'shirt' && L.type !== 'worker')) P.add('torso', shade(L.topCol, 0.72), sheet(1, 2, (u, v) => onTorso(0.05 * (u - 0.5), lerp(0.02 * s, lt - 0.09 * s, v), 0.006 * s)));
  if (L.flap > 0) {                                          // the top's lower edge, over the hips
    const g = kG * puff, yF = [-(L.flap + 0.04 * gait.run) * s, 0, 0.08 * s];        // (longer on a child that runs, leaning forward)
    const kr = 1 + 0.26 * gait.run;                          // (and wider front to back: leaning forward, it must still go round the hips)
    const kx = 1 + 0.1 * gait.run, rx = [0.18 * s * g * kx, 0.17 * s * g * kx, lerp(rB, rT, 0.08 * s / T) * 1.012], rz = [0.134 * s * g * kr, 0.126 * s * g * kr, lerp(rB, rT, 0.08 * s / T) * zs * 1.02];
    P.add('torso', L.topCol, sheet(12, 2, (u, v) => { const j = Math.round(v * 2), a = TAU * u; return [rx[j] * Math.sin(a), yF[j], rz[j] * Math.cos(a)]; }, true));
  }
  // things worn on the trunk: satchel or rucksack with its straps, shoulder bag with its strap, a parcel in both arms
  const zBack = -(rB + rT) / 2 * zs;
  if (L.back) {
    const rando = L.back === 'randoseru', d = (rando ? 0.15 : 0.13) * s, yc = 0.5 * lt + 0.02 * s;
    P.add('torso', L.backCol, rbox((rando ? 0.24 : 0.25) * s, (rando ? 0.32 : 0.36) * s, d, 0.04 * s, 0, yc, zBack - d / 2 + 0.01 * s));
    if (rando) P.add('torso', shade(L.backCol, 0.82), box(0.245 * s, 0.25 * s, 0.02 * s, 0, yc + 0.03 * s, zBack - d + 0.005 * s));
    else P.add('torso', shade(L.backCol, 0.8), box(0.17 * s, 0.14 * s, 0.03 * s, 0, yc - 0.07 * s, zBack - d));
    for (const e of [-1, 1]) P.add('torso', shade(L.backCol, rando ? 1 : 0.8), sheet(1, 3, (u, v) => onTorso(e * 0.62 + 0.16 * (u - 0.5), lerp(0.3 * T, lt - 0.035 * s, v), 0.011 * s)));
  }
  if (L.sling) {
    const sd = L.sling.side, strap = shade(L.sling.col, 0.8);
    P.add('torso', L.sling.col, rbox(0.085 * s, 0.2 * s, 0.27 * s, 0.03 * s, sd * (rB + 0.06 * s), -0.03 * s, -0.02 * s));
    P.add('torso', strap, sheet(5, 1, (u, v) => onTorso(lerp(-sd * 0.8, sd * 1.45, u), lerp(lt - 0.05 * s, 0.06 * s, u) + (v - 0.5) * 0.045 * s, 0.01 * s)),
      sheet(5, 1, (u, v) => onTorso(Math.PI + lerp(sd * 0.8, -sd * 1.45, u), lerp(lt - 0.05 * s, 0.06 * s, u) + (v - 0.5) * 0.045 * s, 0.01 * s)));
  }
  const boxZ = rB * zs + 0.18 * s, boxY = 0.2 * s;
  if (L.box) P.add('item', CARDBOARD, box(0.36 * s, 0.26 * s, 0.3 * s, 0, boxY, boxZ));
  if (L.box) P.add('item', TAPE, box(0.06 * s, 0.266 * s, 0.306 * s, 0, boxY, boxZ));
  if (L.cord) {                                              // the cord of the headphones, from the neck to a pocket
    const pts = [[0.085 * s, lt + 0.05 * s, 0.01 * s], onTorso(0.5, lt - 0.1 * s, 0.012 * s), onTorso(0.75, 0.3 * T, 0.014 * s), onTorso(1.0, 0.03 * s, 0.014 * s)];
    const a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
    for (let i = 0; i + 1 < pts.length; i++) {
      a.set(...pts[i]); b.set(...pts[i + 1]);
      const len = a.distanceTo(b);
      P.add('torsoFine', WHITE, new THREE.CylinderGeometry(0.004 * s, 0.004 * s, len, 4, 1, true).translate(0, len / 2, 0).applyQuaternion(q.setFromUnitVectors(up, b.clone().sub(a).normalize())).translate(a.x, a.y, a.z));
    }
  }

  // ---- pelvis, and the hem of a skirt or coat (a cone on the pelvis, made wider each frame to hold both legs)
  const hipCol = coat ? L.topCol : L.botCol, wide = skirt || coat;
  P.add('pelvis', hipCol, wide && !(L.flap > 0) ? ell(Math.max(0.157 * kG, 0.182) * s, 0.105 * s, 0.112 * s * kG, 0, 0.01 * s, 0, 12, 6) : ell(0.15 * s * kG, 0.098 * s, 0.102 * s * kG, 0, 0.01 * s, 0, 12, 6));
  const hemLen = coat ? 0.92 * leg : L.bottom === 'skirt' ? (L.type === 'kid' ? 0.62 : L.type === 'student' ? 0.8 : 0.9) * leg : L.bottom === 'longskirt' ? 1.68 * leg : 0;
  const hemTop = -0.02 * s, hemRz = (coat ? 0.145 : L.bottom === 'longskirt' ? 0.19 : 0.15) * s, hemRz0 = 0.092 * s * kG, HEM_MAX = 2.2 * (1 - 0.5 * gait.run), HEM_MID = 0.4;   // (HEM_MAX: how far, in its own depths, a leg may push the hem; a running child's knees come out of it instead)
  /** How deep (half, front to back) the hem is at rest at t = 0 (its upper ring, just inside the hips) .. 1 (its lower edge). */
  const hemDeep = (t) => lerp(hemRz0, hemRz, t);
  if (hemLen) {
    const rxT = Math.max(0.156 * kG, 0.184) * s, rxB = (L.bottom === 'longskirt' && !coat ? 0.25 : coat ? 0.215 : 0.235) * s;
    P.add('hem', coat ? L.topCol : L.botCol, sheet(12, 2, (u, v) => {
      const t = [1, HEM_MID, 0][Math.round(v * 2)], a = TAU * u + Math.PI;        // three rings; the seam at the back
      const sq = (v) => Math.sign(v) * Math.pow(Math.abs(v), 0.55);             // a squarer section than an ellipse: room for both thighs
      return [lerp(rxT, rxB, Math.pow(t, 0.6)) * sq(Math.sin(a)), hemTop - hemLen * t, hemDeep(t) * sq(Math.cos(a))];
    }, true));
  }

  // ---- head: skin, ears, hair or a hat, glasses, a mask, headphones (hs: the head's own scale)
  P.add('head', skin, ell(0.103 * hs, 0.114 * hs, 0.108 * hs, 0, 0.10 * hs, 0.015 * hs));
  for (const e of [-1, 1]) P.add('head', skin, ell(0.016 * hs, 0.03 * hs, 0.02 * hs, e * 0.103 * hs, 0.085 * hs, 0.005 * hs, 6, 4));   // ears
  const hair = L.hairCol;
  if (L.hat === 'hood') P.add('head', L.hatCol, ell(0.126 * hs, 0.128 * hs, 0.13 * hs, 0, 0.112 * hs, -0.022 * hs));
  else {
    P.add('head', hair, ell(0.112 * hs, 0.112 * hs, 0.112 * hs, 0, 0.118 * hs, -0.012 * hs, 12, L.hat ? 6 : 8));
    if (L.hair === 'bob') P.add('head', hair, ell(0.119 * hs, 0.095 * hs, 0.088 * hs, 0, 0.06 * hs, -0.04 * hs, 10, 6));
    else if (L.hair === 'long') P.add('head', hair, ell(0.1 * hs, 0.16 * hs, 0.06 * hs, 0, -0.01 * hs, -0.082 * hs, 10, 6));
    else if (L.hair === 'pony') P.add('head', hair, ell(0.04 * hs, 0.11 * hs, 0.04 * hs, 0, 0, 0, 8, 5).rotateX(0.55).translate(0, 0.07 * hs, -0.16 * hs), ball(0.03 * hs, 0, 6, 4).translate(0, 0.15 * hs, -0.118 * hs));
    else if (L.hair === 'bun') P.add('head', hair, ball(0.05 * hs, 0, 8, 5).translate(0, 0.2 * hs, -0.095 * hs));
    else if (L.hair === 'twin') for (const e of [-1, 1]) P.add('head', hair, ell(0.035 * hs, 0.08 * hs, 0.035 * hs, 0, 0, 0, 6, 4).rotateZ(e * 0.35).translate(e * 0.125 * hs, 0.04 * hs, -0.04 * hs));
  }
  if (L.hat === 'cap') P.add('head', L.hatCol, ell(0.118 * hs, 0.074 * hs, 0.122 * hs, 0, 0.162 * hs, -0.006 * hs, 12, 6), ell(0.082 * hs, 0.011 * hs, 0.08 * hs, 0, 0.146 * hs, 0.15 * hs, 8, 4));
  else if (L.hat === 'brim') P.add('head', L.hatCol, cyl(0.196 * hs, 0.19 * hs, 0.012 * hs, 14, 0, 0.176 * hs, 0), cyl(0.113 * hs, 0.1 * hs, 0.085 * hs, 12, 0, 0.216 * hs, -0.004 * hs));
  else if (L.hat === 'tulip') P.add('head', L.hatCol, ell(0.121 * hs, 0.092 * hs, 0.122 * hs, 0, 0.17 * hs, -0.006 * hs, 12, 6), new THREE.CylinderGeometry(0.122 * hs, 0.18 * hs, 0.05 * hs, 12, 1, true).translate(0, 0.142 * hs, -0.004 * hs));
  else if (L.hat === 'helmet') {
    P.add('head', L.hatCol, ell(0.125 * hs, 0.1 * hs, 0.132 * hs, 0, 0.156 * hs, -0.004 * hs, 12, 6), cyl(0.137 * hs, 0.133 * hs, 0.012 * hs, 12, 0, 0, 0).scale(1, 1, 1.1).translate(0, 0.128 * hs, 0.006 * hs));
    if (L.band) P.add('head', L.band, new THREE.CylinderGeometry(0.1235 * hs, 0.1275 * hs, 0.022 * hs, 12, 1, true).scale(1, 1, 1.056).translate(0, 0.152 * hs, -0.004 * hs));
  }
  if (L.glasses) {
    for (const e of [-1, 1]) P.add('head', DARK, box(0.046 * hs, 0.03 * hs, 0.006 * hs).rotateY(e * 0.3).translate(e * 0.042 * hs, 0.112 * hs, 0.115 * hs),
      box(0.005 * hs, 0.007 * hs, 0.1 * hs).rotateY(-e * 0.4).translate(e * 0.085 * hs, 0.112 * hs, 0.06 * hs));
  }
  if (L.mask) P.add('head', WHITE, ell(0.078 * hs, 0.05 * hs, 0.052 * hs, 0, 0.052 * hs, 0.076 * hs, 8, 5));
  if (L.phones) {
    P.add('head', DARK, new THREE.TorusGeometry(0.122 * hs, 0.009 * hs, 4, 8, Math.PI).translate(0, 0.118 * hs, -0.012 * hs));
    for (const e of [-1, 1]) P.add('head', DARK, new THREE.CylinderGeometry(0.037 * hs, 0.037 * hs, 0.035 * hs, 8).rotateZ(Math.PI / 2).translate(e * 0.117 * hs, 0.088 * hs, 0.002 * hs));
  }

  // ---- arms and legs
  const lo = L.loose, rA = puff > 1 ? 1.05 : 1;
  for (const n of ['L', 'R']) {
    const sd = n === 'L' ? 1 : -1;
    P.add('armU' + n, armCol, ball(0.05 * s * rA));
    if (long) P.add('armU' + n, armCol, tube(upper, 0.05 * s * rA, 0.04 * s * rA));
    else { P.add('armU' + n, skin, tube(upper, 0.045 * s, 0.038 * s)); P.add('armU' + n, armCol, tube(0.55 * upper, 0.053 * s, 0.05 * s)); }
    P.add('armF' + n, long ? armCol : skin, ball(0.04 * s * rA), tube(fore, 0.04 * s * rA, 0.032 * s * rA));
    P.add('armF' + n, skin, ball(0.042 * s, fore + 0.02 * s, 8, 4));                                              // hand
    if (L.phone === sd) P.add('armF' + n, DARK, box(0.07 * s, 0.13 * s, 0.012 * s, 0, fore + 0.07 * s, 0.03 * s));
    if (L.bottom === 'long') {
      P.add('thigh' + n, hipCol, ball(0.075 * s * lo));
      P.add('thigh' + n, L.botCol, tube(leg, 0.075 * s * lo, 0.056 * s * lo));
      P.add('shin' + n, L.botCol, ball(0.054 * s * lo), tube(leg, 0.054 * s * lo, 0.04 * s * lo * lo), ball(0.04 * s * lo * lo, leg, 8, 4));
    } else {
      P.add('thigh' + n, hipCol, ball(0.075 * s));
      P.add('thigh' + n, L.legCol, tube(leg, 0.07 * s, 0.054 * s));
      if (L.bottom === 'shorts') P.add('thigh' + n, L.botCol, tube(0.52 * leg, 0.08 * s, 0.074 * s));
      else if (hemLen) P.add('thigh' + n, L.botCol, tube(0.3 * leg, 0.076 * s, 0.071 * s));      // under a skirt: its colour where a thigh swinging forward meets the hips
      P.add('shin' + n, L.legCol, ball(0.053 * s), tube(leg, 0.053 * s, 0.037 * s), ball(0.037 * s, leg, 8, 4));
      if (L.socks) P.add('shin' + n, L.socks, tube(0.2 * leg, 0.044 * s, 0.041 * s, 0.8 * leg));
    }
    if (L.shoes === 'boots') P.add('shin' + n, L.shoeCol, tube(0.36 * leg, 0.058 * s, 0.05 * s, 0.64 * leg));
    const wS = (L.shoes === 'leather' ? 0.084 : 0.094) * s;
    P.add('shoe' + n, L.shoeCol, rbox(wS, 0.068 * s, heel + toe, 0.028 * s, 0, -ankle + 0.034 * s, (toe - heel) / 2));
    if (L.shoes === 'sneaker') P.add('shoe' + n, hex(L.shoeCol) === 'e8e6e0' ? '#9a9da4' : WHITE, box(wS + 0.004 * s, 0.02 * s, heel + toe + 0.004 * s, 0, -ankle + 0.01 * s, (toe - heel) / 2));
    else if (L.shoes === 'leather') P.add('shoe' + n, shade(L.shoeCol, 0.7), box(wS * 0.9, 0.024 * s, 0.07 * s, 0, -ankle + 0.012 * s, -heel + 0.035 * s));
  }

  // ---- the thing in a hand: it hangs from the grip (the item body's origin) along -Y; a stick stands along +Y
  const H = L.hand, kindH = H && H.kind;
  let caneLen = 0;
  if (kindH === 'tote') {
    P.add('item', H.col, rbox(0.075 * s, 0.27 * s, 0.3 * s, 0.022 * s, 0, -0.275 * s, 0));
    for (const e of [-1, 1]) P.add('item', shade(H.col, 0.8), box(0.016 * s, 0.16 * s, 0.016 * s).rotateX(e * 0.5).translate(0, -0.07 * s, e * 0.04 * s));
  } else if (kindH === 'brief' || kindH === 'school') {
    const sch = kindH === 'school', d = (sch ? 0.105 : 0.075) * s, w = (sch ? 0.35 : 0.38) * s, hh = (sch ? 0.24 : 0.27) * s;
    P.add('item', H.col, rbox(d, hh, w, 0.022 * s, 0, -0.06 * s - hh / 2, 0), box(0.02 * s, 0.07 * s, 0.13 * s, 0, -0.03 * s, 0));
    if (sch) P.add('item', H.accent, box(d + 0.004 * s, 0.03 * s, w + 0.004 * s, 0, -0.06 * s - hh * 0.3, 0));
  } else if (kindH === 'plastic') {
    P.add('item', H.col, ell(0.06 * s, 0.11 * s, 0.105 * s, 0, -0.25 * s, 0, 8, 6), new THREE.CylinderGeometry(0.012 * s, 0.085 * s, 0.17 * s, 8, 1, true).scale(0.62, 1, 1).translate(0, -0.09 * s, 0));
  } else if (kindH === 'umbrella') {
    P.add('item', H.col, new THREE.CylinderGeometry(0.03 * s, 0.013 * s, 0.5 * s, 8, 1, false).translate(0, -0.36 * s, 0));
    P.add('itemFine', DARK, new THREE.CylinderGeometry(0.01 * s, 0.01 * s, 0.14 * s, 6).translate(0, -0.05 * s, 0), new THREE.CylinderGeometry(0.006 * s, 0.006 * s, 0.09 * s, 5).translate(0, -0.65 * s, 0));
  }

  // the 15 rigid bodies: pelvis, torso, head, 2 x (upper arm, forearm with the hand, thigh, shin, shoe), the thing
  // carried (in a hand, or a parcel in both arms) and the hem of a skirt or coat (on the pelvis)
  const g = new THREE.Group(); g.userData.sem = 12;
  const body = (name, parent = g) => { const o = new THREE.Group(); const m = P.build(name); if (m) o.add(m); parent.add(o); return o; };
  const thin = (parent, name) => { const m = P.build(name, { shadow: false }); if (m) { parent.add(m); ctx.noOutline(m); fine.push(m); } };
  const pelvis = body('pelvis'), torso = body('torso'), head = body('head', torso), hem = new THREE.Group();
  pelvis.add(hem);
  let hemBones = null, hemSkin = null;
  if (hemLen) {
    // bones: the upper ring (it turns with the hips); the middle ring's front and back; the lower edge's front and back
    const geo = P.build('hem').geometry, pos = geo.attributes.position, n = pos.count, si = new Uint16Array(4 * n), sw = new Float32Array(4 * n);
    for (let i = 0; i < n; i++) {
      const t = clamp((hemTop - pos.getY(i)) / hemLen, 0, 1), fr = 0.5 + 0.5 * clamp(pos.getZ(i) / hemDeep(t), -1, 1);
      if (t < 0.5 * HEM_MID) { si.set([0, 0, 0, 0], 4 * i); sw.set([1, 0, 0, 0], 4 * i); }
      else { const b = t < 0.5 * (1 + HEM_MID) ? 1 : 3; si.set([b, b + 1, 0, 0], 4 * i); sw.set([fr, 1 - fr, 0, 0], 4 * i); }
    }
    // The skinning is done here and not by a SkinnedMesh: that one costs a bone texture and shaders of its own every
    // frame and pass (measured in a town: 0.2 ms a hem, as much as the rest of the person), and these are few vertices.
    // Same result: each vertex follows its (at most two) bones, positions and normals blended by their weights.
    const nor = geo.attributes.normal, rest = Float32Array.from(pos.array), restN = Float32Array.from(nor.array);
    pos.setUsage(THREE.DynamicDrawUsage); nor.setUsage(THREE.DynamicDrawUsage);
    geo.computeBoundingSphere(); geo.boundingSphere.radius += HEM_MAX * hemRz + 0.1;      // wherever the legs push it
    const skin = new THREE.Mesh(geo, mat);
    skin.name = 'hem'; skin.castShadow = true; skin.receiveShadow = true;
    hemBones = [0, 1, 2, 3, 4].map(() => new THREE.Object3D());
    hemSkin = () => {
      for (const b of hemBones) b.updateMatrix();
      const P = pos.array, N = nor.array;
      for (let i = 0; i < n; i++) {
        const rx = rest[3 * i], ry = rest[3 * i + 1], rz = rest[3 * i + 2], ax = restN[3 * i], ay = restN[3 * i + 1], az = restN[3 * i + 2];
        let x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0;
        for (let k = 0; k < 2; k++) {
          const w = sw[4 * i + k];
          if (!w) continue;
          const e = hemBones[si[4 * i + k]].matrix.elements;
          x += w * (e[0] * rx + e[4] * ry + e[8] * rz + e[12]); y += w * (e[1] * rx + e[5] * ry + e[9] * rz + e[13]); z += w * (e[2] * rx + e[6] * ry + e[10] * rz + e[14]);
          nx += w * (e[0] * ax + e[4] * ay + e[8] * az); ny += w * (e[1] * ax + e[5] * ay + e[9] * az); nz += w * (e[2] * ax + e[6] * ay + e[10] * az);
        }
        const l = Math.hypot(nx, ny, nz) || 1;
        P[3 * i] = x; P[3 * i + 1] = y; P[3 * i + 2] = z; N[3 * i] = nx / l; N[3 * i + 1] = ny / l; N[3 * i + 2] = nz / l;
      }
      pos.needsUpdate = true; nor.needsUpdate = true;
    };
    hem.add(skin);
  }
  head.position.set(0, lt + 0.05 * s, 0.01 * s);
  const side = (n, sgn) => ({ sgn, armU: body('armU' + n), armF: body('armF' + n), thigh: body('thigh' + n), shin: body('shin' + n), shoe: body('shoe' + n),
    shoulder0: new THREE.Vector3(sgn * (fit.shoulderW - 0.008 * s), lt - SHOULDER_Y * s, -0.015 * s),
    // how this arm moves: swing (1: free), how far the elbow and the hand keep out, and what the hand holds
    mode: L.box ? 'box' : L.phone === sgn ? 'phone' : H && H.side === sgn ? (kindH === 'cane' ? 'cane' : 'carry') : L.sling && L.sling.side === sgn ? 'wide' : 'swing' });
  const sides = [side('L', 1), side('R', -1)];
  if (kindH === 'cane') {                                    // as long as the hand is high when standing, and a little more
    const yH = fit.stand + lt - SHOULDER_Y * s - upper * Math.cos(0.05) / Math.hypot(0.12, 1) - (fore + 0.025 * s) * Math.cos(0.5) / Math.hypot(0.05, 1);
    caneLen = 1.03 * yH;
    P.add('itemFine', H.col, new THREE.CylinderGeometry(0.009 * s, 0.012 * s, caneLen, 6).translate(0, caneLen / 2, 0), new THREE.CylinderGeometry(0.014 * s, 0.014 * s, 0.12 * s, 6).rotateX(Math.PI / 2).translate(0, 0.004 * s, 0.015 * s));
  }
  const item = body('item');
  thin(item, 'itemFine'); thin(torso, 'torsoFine');
  hem.matrixAutoUpdate = false;

  const G = gait, Dpi = G.D * Math.PI;
  const hip = new THREE.Vector3(), tgt = new THREE.Vector3(), mid = new THREE.Vector3(), end = new THREE.Vector3(), pole = new THREE.Vector3();
  const sh = new THREE.Vector3(), el = new THREE.Vector3(), hand = new THREE.Vector3(), grip = new THREE.Vector3(), tip = new THREE.Vector3();
  const inv = new THREE.Matrix4(), q1 = new THREE.Vector3(), q2 = new THREE.Vector3(), q3 = new THREE.Vector3(), UPV = new THREE.Vector3(0, 1, 0);
  const foot = { z: 0, y: 0, th: 0 };
  /** The ankle (z forward of the origin, y above the ground) and the foot's pitch th (toe up > 0) at stance progress u:
   *  on the heel first, flat, then over the ball of the foot. The flat foot's ankle is at delta + A (1 - 2u). */
  function stance(u, o) {
    const { ball: bl } = fit, zr = G.delta + G.A * (1 - 2 * u);
    const th = u < U_FLAT ? TOE_UP * (1 - u / U_FLAT) : u > U_HEEL ? -G.heelUp * smooth((u - U_HEEL) / (1 - U_HEEL)) : 0;
    if (th >= 0) { o.z = zr - heel + heel * Math.cos(th) - ankle * Math.sin(th); o.y = heel * Math.sin(th) + ankle * Math.cos(th); }
    else { o.z = zr + bl - bl * Math.cos(th) - ankle * Math.sin(th); o.y = -bl * Math.sin(th) + ankle * Math.cos(th); }
    o.th = th;
  }
  const off = { z: 0, y: 0, th: 0 }, strike = { z: 0, y: 0, th: 0 };
  stance(1, off); stance(0, strike);
  const vSwing = -STEP * TAU * (1 - G.D) * 0.6, lift = (0.03 + 0.05 * G.run) * h * hb.lift, reach = 0.04 * h * hb.reach / 0.0219;
  /** The foot in the air, from toe-off (v = 0) to heel strike (v = 1): it swings through, lifted, the toes coming up.
   *  A long stride (habit reach) shows here: the foot swings out ahead of where it will land and comes back to it. */
  function swing(v, o) {
    const v2 = v * v, v3 = v2 * v, bl = fit.ball, sv = Math.sin(Math.PI * v);
    o.z = (2 * v3 - 3 * v2 + 1) * off.z + (v3 - 2 * v2 + v) * vSwing + (3 * v2 - 2 * v3) * strike.z + (v3 - v2) * vSwing + reach * v2 * v2 * (1 - v) * (1 - v);
    o.th = lerp(-G.heelUp, TOE_UP, smooth((v - 0.05) / 0.8));
    const floor = o.th >= 0 ? heel * Math.sin(o.th) + ankle * Math.cos(o.th) : -bl * Math.sin(o.th) + ankle * Math.cos(o.th);
    o.y = Math.max(lerp(off.y, strike.y, v) + lift * sv, floor);
  }
  // The hem is a skinned cone of three rings: the upper one stays on the hips and turns with them; the middle one and
  // the lower edge each have a bone for their front and one for their back (the sides follow both). Every frame the
  // front bones go as far forward as the legs need (hemHold: points of both legs the hem must hold), the back bones as
  // far back; with nothing to hold they swing a little with the step. So the cloth lies over a thigh that swings
  // forward and hangs straight from there, and stays flat where no leg pushes it.
  const holdT = new Float64Array(12), holdF = new Float64Array(12), holdB = new Float64Array(12), q4 = new THREE.Vector3(), HEM_AT = [0.18, HEM_MID, 0.7, 1];
  let nHold = 0;
  /** A point of a leg the hem must hold (from the pelvis, r: the leg's thickness there). */
  function hemHold(p, r, t) {
    // (1.22: the cone is shallower at the thighs' outer sides than on the middle line, and flat between its 12 edges)
    holdT[nHold] = t; holdF[nHold] = 1.22 * (p.z + r); holdB[nHold] = 1.22 * (p.z - r); nHold++;
  }
  // The walking stick is planted when the other side's foot lands and stays where it stands on the ground (it moves
  // back under the hand as fast as the body moves forward) for CANE_D of the cycle, then swings forward again.
  const caneSd = kindH === 'cane' ? H.side : 0, DcPi = CANE_D * Math.PI, Ac = DcPi * STEP, zc0 = 0.12, vCane = -STEP * (TAU - 2 * DcPi) * 0.5;
  const caneX = caneSd * (fit.shoulderW + 0.045 * s), ct = { z: 0, y: 0 };
  function caneTip(phase, o) {
    const pc = wrap(wrap(phase + (caneSd < 0 ? 0 : Math.PI) - Math.PI) + Dpi - DcPi);     // 0 in the middle of the stick's time on the ground
    if (Math.abs(pc) <= DcPi) { o.z = zc0 - Ac * pc / DcPi; o.y = 0; return; }
    const v = (pc > 0 ? pc - DcPi : pc + TAU - DcPi) / (TAU - 2 * DcPi), v2 = v * v, v3 = v2 * v;
    o.z = (2 * v3 - 3 * v2 + 1) * (zc0 - Ac) + (v3 - 2 * v2 + v) * vCane + (3 * v2 - 2 * v3) * (zc0 + Ac) + (v3 - v2) * vCane;
    o.y = 0.06 * Math.sin(Math.PI * v);
  }
  const carrySd = H && kindH !== 'cane' ? H.side : 0, swingK = ITEM_SWING[kindH] || 0;
  const headDown = L.phone ? 0.3 : 0, headK = L.type === 'elder' ? 0.45 : 0.7;

  let prev, w = 0;                                           // the phase of the call before; 0 walking .. 1 standing
  function pose({ phase = 0, moving, dt, stand } = {}) {
    const still = moving === undefined ? prev !== undefined && phase === prev : !moving;
    if (stand !== undefined) w = stand ? 1 : 0;
    else if (prev === undefined) w = 0;
    else { const t = dt === undefined ? 0.1 : dt; w = clamp(w + (still ? t / STAND_S : -t / WALK_S), 0, 1); }
    prev = phase;
    const ws = smooth(w), go = 1 - ws, sn = Math.sin(phase), cs = Math.cos(phase), run = G.run;

    // pelvis: up over the standing leg, towards it, turned with the leg that swings forward, dropped on that side
    const Hp = lerp(G.H0 + G.bob * Math.cos(2 * (phase - G.lead)), fit.stand, ws);
    const px = -0.012 * h * (1 - 0.6 * run) * cs * go;
    const yawP = -G.tw * sn * go, rollP = -0.04 * cs * go;
    pelvis.position.set(px, Hp, 0); pelvis.rotation.set(0, yawP, rollP);
    // trunk and head: against the pelvis' turn, a little forward (more when running, or by habit: an old person's
    // stoop stays when standing); the shoulder of a hand that carries a load hangs lower; the head keeps looking
    // ahead (down at a phone)
    const lean = (0.035 + hb.lean + 0.14 * run) * go + 0.65 * Math.max(hb.lean, 0) * ws, yawT = 0.07 * Math.min(hb.arm, 1.25) * sn * go;
    const rollT = -0.3 * rollP - carrySd * hb.drop * (1 - 0.4 * ws);
    torso.position.set(px, Hp, 0); torso.rotation.set(lean, yawT, rollT);
    head.rotation.set(-headK * lean + headDown, -0.8 * yawT, -0.8 * rollT);
    pelvis.updateMatrix(); torso.updateMatrix();
    if (hemLen) inv.makeRotationFromQuaternion(pelvis.quaternion).invert();       // the hem hangs upright and faces ahead: it does not turn with the pelvis
    nHold = 0;

    for (let i = 0; i < 2; i++) {
      const sd = sides[i], sgn = sd.sgn;
      // legs: the foot's place from the phase (0 in the middle of this foot's time on the ground), the knee forward
      const p = wrap(phase + (sgn > 0 ? 0 : Math.PI) - Math.PI);
      if (Math.abs(p) <= Dpi) stance((p + Dpi) / (2 * Dpi), foot);
      else swing((p > 0 ? p - Dpi : p + TAU - Dpi) / (TAU - 2 * Dpi), foot);
      tgt.set(sgn * fit.track, lerp(foot.y, ankle, ws), lerp(foot.z, -0.012 * h, ws));
      hip.set(sgn * fit.hipW, 0, 0).applyMatrix4(pelvis.matrix);
      twoBone(hip, tgt, leg, leg, pole.set(sgn * 0.06, 0, 1), mid, end);
      setBone(sd.thigh, hip, mid); setBone(sd.shin, mid, end);
      sd.shoe.position.copy(end); sd.shoe.rotation.set(-foot.th * go, 0, 0);
      if (hemLen) {                                          // this leg from the pelvis, in the person's own directions
        q1.copy(hip).sub(pelvis.position); q2.copy(mid).sub(pelvis.position); q3.copy(end).sub(pelvis.position);
        // the leg at four heights of the hem: on the thigh, or below the knee on the shin
        const rTh = 0.078 * s / Math.max((q1.y - q2.y) / leg, 0.6);            // (a thigh that slants is thicker front to back)
        for (const t of HEM_AT) {
          const y = hemTop - t * hemLen;
          if (y >= q2.y) hemHold(q4.lerpVectors(q1, q2, clamp((y - q1.y) / (q2.y - q1.y), 0, 1)), rTh, t);
          else hemHold(q4.lerpVectors(q2, q3, clamp((y - q2.y) / (q3.y - q2.y), 0, 1)), 0.07 * s, t);
        }
      }
      // arms: from the shoulder, against the leg of the same side; the elbow bends more as the hand comes forward
      sh.copy(sd.shoulder0).applyMatrix4(torso.matrix);
      if (sd.mode === 'box' || sd.mode === 'phone') {        // the hand at a place in front of the trunk
        if (sd.mode === 'box') tgt.set(sgn * 0.2 * s, boxY - 0.09 * s, boxZ + 0.03 * s); else tgt.set(sgn * 0.05 * s, lt - 0.17 * s, rT * zs + 0.19 * s);
        tgt.applyMatrix4(torso.matrix);
        twoBone(sh, tgt, upper, fore, pole.set(sgn * 0.5, -0.6, -1), el, hand);
      } else if (sd.mode === 'cane') {                       // the hand over the stick, the stick on its tip
        caneTip(phase, ct);
        const zt = lerp(ct.z, 0.11, ws), yt = ct.y * go, zg = lerp(zc0 + 0.02 + 0.35 * (ct.z - zc0), 0.08, ws);
        tip.set(caneX + sgn * 0.02 * s, yt, zt);
        const yg = yt + Math.sqrt(Math.max(caneLen * caneLen - (zt - zg) ** 2 - (0.02 * s) ** 2, 0.01));
        tgt.set(caneX, yg + 0.0225 * s, zg - 0.011 * s);
        twoBone(sh, tgt, upper, fore, pole.set(sgn * 0.35, -0.2, -1), el, hand);
        grip.copy(hand).sub(el).normalize().multiplyScalar(0.025 * s).add(hand);
        q4.copy(tip).sub(grip).normalize();
        item.position.copy(tip).addScaledVector(q4, -caneLen); item.quaternion.setFromUnitVectors(UPV, q4);
      } else {
        const free = sd.mode === 'swing', amp = free ? hb.arm : sd.mode === 'wide' ? 0.55 * Math.min(hb.arm, 1) : 0.28;
        const sw = (ph) => -sgn * amp * (0.42 + 0.3 * run) * Math.sin(ph - 0.15);
        const a = lerp(0.04 + sw(phase), 0.02, ws);
        const fwd = 0.5 - 0.5 * sgn * Math.sin(phase - 0.15);
        const bend = free || sd.mode === 'wide' ? lerp(0.26 + 0.35 * fwd + run * (1.0 + 0.2 * fwd), 0.14, ws) : lerp(0.2 + 0.1 * fwd, 0.16, ws);
        const outE = free ? 0.10 : sd.mode === 'wide' ? 0.2 : 0.17, outH = free ? -0.04 : sd.mode === 'wide' ? 0.13 : 0.07;
        el.set(sgn * outE, -Math.cos(a), Math.sin(a)).normalize().multiplyScalar(upper).add(sh);
        hand.set(sgn * outH, -Math.cos(a + bend), Math.sin(a + bend)).normalize().multiplyScalar(fore).add(el);
        if (sd.mode === 'carry') {                           // the thing hangs from the grip and swings after the hand, a little late
          grip.copy(hand).sub(el).normalize().multiplyScalar(0.025 * s).add(hand);
          item.position.copy(grip);
          item.rotation.set((kindH === 'umbrella' ? 0.3 : 0) + swingK * sw(phase - ITEM_LAG) * go, yawT, sgn * (0.04 + 0.025 * swingK * (1 + Math.cos(2 * phase - ITEM_LAG)) * go));
        }
      }
      setBone(sd.armU, sh, el); setBone(sd.armF, el, hand);
    }
    if (L.box) { item.position.copy(torso.position); item.quaternion.copy(torso.quaternion); }
    if (hemLen) {
      // the lower edge, front and back: at least as far as the legs need, and swinging with the step when free;
      // to the side it stays behind the hips' sway
      const zs0 = 0.011 * h * Math.sin(2 * phase - 1.0) * go, dM = hemDeep(HEM_MID), d1 = hemDeep(1);
      let mf = HEM_MID * zs0, mb = mf, df = zs0, db = zs0;
      for (let i = 0; i < nHold; i++) if (holdT[i] <= HEM_MID) {          // the middle ring: between it and the upper ring the cloth is straight
        const u = holdT[i] / HEM_MID;
        mf = Math.max(mf, (holdF[i] - (1 - u) * hemRz0) / u - dM); mb = Math.min(mb, (holdB[i] + (1 - u) * hemRz0) / u + dM);
      }
      mf = Math.min(mf, HEM_MAX * hemRz); mb = Math.max(mb, -HEM_MAX * hemRz);
      for (let i = 0; i < nHold; i++) if (holdT[i] > HEM_MID) {           // the lower edge: straight from the middle ring
        const u = (holdT[i] - HEM_MID) / (1 - HEM_MID);
        df = Math.max(df, (holdF[i] - (1 - u) * (dM + mf)) / u - d1); db = Math.min(db, (holdB[i] + (1 - u) * (dM - mb)) / u + d1);
      }
      df = Math.min(df, HEM_MAX * hemRz + 0.5 * mf); db = Math.max(db, -HEM_MAX * hemRz + 0.5 * mb);
      const sx = 0.6 * 0.012 * h * (1 - 0.6 * run) * (cs - Math.cos(phase - 0.8)) * go;
      hem.matrix.copy(inv);
      hemBones[0].rotation.set(0, yawP, rollP);              // the upper ring turns with the hips; the rest keeps facing ahead
      hemBones[1].position.set(HEM_MID * sx, 0, mf); hemBones[2].position.set(HEM_MID * sx, 0, mb);
      hemBones[3].position.set(sx, 0, df); hemBones[4].position.set(sx, 0, db);
      hemSkin();
    }
  }
  pose({ phase: 0, stand: false });
  prev = undefined;
  return { g, fine, pose, type: L.type, look: L, habit: hb, fit, gait, caneLen, bodies: { pelvis, torso, head, item, hem, sides, hemBones } };
}
