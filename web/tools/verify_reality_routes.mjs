// Run the normal 384×192 live page until the simulator terminates each route.
import assert from 'node:assert/strict';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const endpoint = new URL(process.argv.find(arg=>arg.startsWith('--url='))?.slice(6) || 'http://127.0.0.1:8768/');
assert.equal(endpoint.hostname,'127.0.0.1');
const repair = process.argv.includes('--background-repair');
const mode = repair ? 'repair384' : 'stable384';
const output = fileURLToPath(new URL(repair ? '../../runs/reality/disocclusion384/' : '../../runs/reality/p2_resolution/', import.meta.url));
const started = new Date().toISOString();
const directory = join(output, 'full_routes', started.replace(/[:.]/g, '-'));
const selection = JSON.parse(await readFile(join(output, 'fit', 'selection.json'), 'utf8'));
await mkdir(directory, {recursive: true});
const browser = await puppeteer.launch({executablePath:'/usr/bin/google-chrome', headless:!process.argv.includes('--visible'),
  defaultViewport:{width:1440,height:900}, args:['--window-size=1470,1000']});
const report = {started, browser:await browser.version(), mode, checkpoint:repair ? selection.repair : selection.checkpoint,
  decoder:selection.model, input_sizes:[[320,160],[320,160]], output_size:[384,192], towns:{}};

async function save() {
  const path = join(directory, 'routes.json');
  await writeFile(path+'.tmp', JSON.stringify(report,null,2)+'\n');
  await rename(path+'.tmp',path);
}

async function run(page, town) {
  await page.select('#town', String(town));
  await page.evaluate(() => {
    window.shown = [];
    const panes = document.getElementById('panes');
    const observer = new MutationObserver(() => {
      const tick = Number(panes.dataset.tick);
      if (Number.isInteger(tick)) window.shown.push({tick, at:performance.now()});
    });
    observer.observe(panes,{attributes:true,attributeFilter:['data-tick']});
    window.finishShown = () => { observer.disconnect(); return window.shown; };
  });
  let finalResponse;
  const capture = response => {
    if (response.status() === 410 && new URL(response.url()).pathname === '/frame') finalResponse = response;
  };
  page.on('response', capture);
  const log = setInterval(() => {
    page.$eval('#progress',e=>e.textContent).then(value=>console.log('町'+town, value)).catch(()=>{});
  }, 30000);
  try {
    await page.click('#start');
    await page.waitForFunction(() => window.shown.length || !document.getElementById('start').disabled,
                               {timeout:60000});
    const valid = await page.evaluate(() => ['wide','narrow','generated'].map(id => {
      const c = document.getElementById(id), pixels = c.getContext('2d').getImageData(0,0,c.width,c.height);
      return [c.width,c.height,new Set(pixels.data).size>64];
    }));
    assert.deepEqual(valid, [[320,160,true],[320,160,true],[384,192,true]]);
    console.log('町'+town+' 384×192 ライブ走行中');
    // Each CDP call completes promptly even when the drive lasts longer than Chrome's protocol timeout.
    while (await page.$eval('#start',button=>button.disabled)) await delay(750);
    assert.ok(finalResponse,await page.$eval('#status',e=>e.textContent));
    const result = await finalResponse.json();
    const shown = await page.evaluate(() => window.finishShown());
    const first = shown[0], last = shown.at(-1);
    assert.ok(shown.length>1 && shown.every((v,i)=>!i || v.tick>shown[i-1].tick));
    const {ticks,generated_frames,skipped_frames} = result.timing;
    assert.equal(generated_frames+skipped_frames,ticks);
    assert.ok(last.tick<ticks && shown.length<=generated_frames);
    return {...result, displayed:{frames:shown.length, first_tick:first.tick, last_tick:last.tick,
      fps:(shown.length-1)*1000/(last.at-first.at), generated_but_unshown:generated_frames-shown.length}};
  } finally {
    clearInterval(log);
    page.off('response',capture);
  }
}

try {
  const page = await browser.newPage();
  await page.goto(endpoint.href);
  await page.bringToFront();
  await page.waitForFunction(value => [...document.getElementById('mode').options].some(o=>o.value===value),{},mode);
  await page.select('#mode',mode);
  console.log('Drive report',join(directory,'routes.json'));
  for (const town of [40,41]) {
    const result = await run(page,town);
    report.towns[town] = result;
    await save();
    console.log('町'+town,result.stop_reason,result.distance+' m',result.timing.wall_control_hz.toFixed(2)+' Hz',
                result.displayed.fps.toFixed(2)+' fps');
    if (result.stop_reason === 'stopped') break;
  }
  report.finished = new Date().toISOString();
  await save();
} finally {
  await browser.close();
}
