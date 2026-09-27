// ============================================================================
//  KeiView shim — REPLACES Sakuragaoka Station's hand-authored layout.js.
//
//  Upstream, layout.js is the fixed map of one town (roads, lots, spots, timetable).
//  In KeiView the town comes from a KeiSim export instead (web/src/town.js), so the
//  vendored modules only get what they actually read from here: a ground-height
//  function, the sun direction, small math helpers, the walkable bounds and the
//  far-town rectangles used by houses/far.js. configureWorld() fills the mutable
//  parts before any module is built.
//  Units: metres. +X east, -Z north, +Y up (three.js), same as upstream.
// ============================================================================

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

/** Ground height (m). KeiSim towns are flat (y = 0); KeiView raises sidewalks a little, so the
 *  town loader installs its own function through configureWorld({ heightAt }). */
let heightFn = () => 0;
export function heightAt(x, z) { return heightFn(x, z); }

// Late afternoon (~16:00) sun from the west-southwest, as upstream.
export const SUN_DIR = [-0.776, 0.517, 0.362];

export const WORLD = {
  play: { x0: -200, x1: 200, z0: -200, z1: 200 },    // player is clamped inside
  visual: { x0: -900, x1: 900, z0: -900, z1: 900 },  // distant scenery extent
};

/** Rectangles {x0,x1,z0,z1} filled with low-detail houses by houses/far.js. */
export const FAR_TOWN = [];
let farSkipFn = () => false;
/** houses/far.js asks this whether a far-town cell must stay empty (roads, the KeiSim town itself). */
export function farSkip(x, z) { return farSkipFn(x, z); }

export function configureWorld({ play, visual, farTown, farSkip: skip, heightAt: h } = {}) {
  if (h) heightFn = h;
  if (play) Object.assign(WORLD.play, play);
  if (visual) Object.assign(WORLD.visual, visual);
  if (farTown) { FAR_TOWN.length = 0; FAR_TOWN.push(...farTown); }
  if (skip) farSkipFn = skip;
}
