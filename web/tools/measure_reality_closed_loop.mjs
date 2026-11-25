// Starts and stops runs on the specified local closed-loop viewer and measures actual canvas updates.
import assert from 'node:assert/strict';
import { rename, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const endpoint = new URL(process.argv[2] || 'http://127.0.0.1:8768/');
assert.equal(endpoint.hostname, '127.0.0.1');
const resolution = process.argv.includes('--resolution');
const output = new URL(resolution ? '../../runs/reality/p2_resolution/' : '../../runs/reality/closed_loop/', import.meta.url);
const cases = resolution ? [['40',['stable','stable384']],['41',['stable384','stable']]]
                         : [['40',['pilot','sharp','stable']],['41',['stable','sharp','pilot']]];
const warmup = 16, measured = 64;
const browser = await puppeteer.launch({executablePath:'/usr/bin/google-chrome', headless:true,
  defaultViewport:{width:1440,height:900,deviceScaleFactor:1}, args:['--window-size=1440,900']});
try {
  const page = await browser.newPage();
  await page.bringToFront();
  const errors = [];
  page.on('pageerror', err=>errors.push(err.message));
  await page.goto(endpoint.href);
  if (resolution) await page.waitForFunction(()=>[...document.querySelector('#mode').options].some(v=>v.value==='stable384'));
  await page.evaluate(()=>{
    window.samples = [];
    const panes = document.getElementById('panes');
    new MutationObserver(()=>{
      const tick = Number(panes.dataset.tick);
      if (!Number.isInteger(tick)) return;
      window.samples.push({tick,at:performance.now(),controlHz:Number(document.getElementById('control').textContent)});
    }).observe(panes,{attributes:true,attributeFilter:['data-tick']});
  });
  const results = {};
  let gap, screenshot;
  for (const [town,modes] of cases) {
    results[town] = {};
    for (const mode of modes) {
      await page.select('#town',town);
      await page.select('#mode',mode);
      await page.evaluate(()=>{ window.samples=[]; });
      await page.click('#start');
      try {
        await page.waitForFunction(n=>window.samples.length>=n || !document.getElementById('start').disabled,
                                   {timeout:60000},warmup+measured);
        const samples = await page.evaluate((n,w)=>window.samples.slice(w,w+n),measured,warmup);
        assert.equal(samples.length,measured,await page.$eval('#status',e=>e.textContent));
        assert.ok(samples.every((v,i)=>v.controlHz>0 && (!i||v.tick>samples[i-1].tick)));
        const first=samples[0],last=samples.at(-1);
        const rates = samples.map(r=>r.controlHz).sort((a,b)=>a-b);
        results[town][mode] = {shown:samples.length,display_fps:1000*(samples.length-1)/(last.at-first.at),
          start_tick:first.tick,end_tick:last.tick,unshown_ticks:last.tick-first.tick+1-samples.length,
          median_reported_control_hz:rates[Math.floor(rates.length/2)]};
        const canvases = mode==='pilot' ? ['wide','narrow'] : ['wide','narrow','generated'];
        const valid = await page.evaluate(ids=>ids.every(id=>{
          const canvas=document.getElementById(id),ctx=canvas.getContext('2d');
          const bounds=canvas.getBoundingClientRect(),pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
          return bounds.width>100 && Math.abs(bounds.width-2*bounds.height)<1 && new Set(pixels).size>64;
        }),canvases);
        assert.ok(valid);
        console.log(town,mode,results[town][mode]);
        if (mode!=='pilot') {
          const size = await page.$eval('#generated', c=>[c.width,c.height]);
          assert.deepEqual(size,mode==='stable384' ? [384,192] : [256,128]);
          results[town][mode].native_size = size;
        }
        if (town==='41' && mode===(resolution ? 'stable384' : 'stable')) {
          await page.click('#full');
          await page.waitForFunction(()=>!!document.fullscreenElement);
          await page.click('#full');
          await page.waitForFunction(()=>!document.fullscreenElement);
          await page.$eval('#angle',e=>{e.value=24;e.dispatchEvent(new Event('input'));});
          await page.waitForFunction(()=>document.getElementById('view-label').textContent.includes('+24°'));
          await page.$eval('#angle',e=>{e.value=0;e.dispatchEvent(new Event('input'));});
          await page.waitForFunction(()=>document.getElementById('view-label').textContent.includes('　0°'));
          const path=fileURLToPath(new URL('live-town41.png',output));
          await page.screenshot({path});
          screenshot='live-town41.png';
          // Slow just one viewer request. The control loop must continue without this client.
          let delayed=false;
          const intercept=async request=>{
            if (!delayed && new URL(request.url()).pathname==='/frame') {
              delayed=true;
              await new Promise(resolve=>setTimeout(resolve,850));
            }
            await request.continue();
          };
          const offset=await page.evaluate(()=>window.samples.length);
          page.on('request',intercept);
          await page.setRequestInterception(true);
          await page.waitForFunction(i=>window.samples.slice(i).some((v,k)=>k && v.tick-window.samples[i+k-1].tick>=5),
                                     {timeout:7000},offset);
          gap=await page.evaluate(i=>Math.max(...window.samples.slice(i+1).map((v,k)=>v.tick-window.samples[i+k].tick)),offset);
          await page.setRequestInterception(false);
          page.off('request',intercept);
        }
      } finally {
        if (await page.$eval('#stop',e=>!e.disabled)) await page.click('#stop');
        await page.waitForFunction(()=>!document.getElementById('start').disabled,{timeout:15000});
      }
    }
  }
  await page.click('#bench summary');
  await page.waitForSelector('#benchmark tr:nth-child(7)');
  await page.setViewport({width:390,height:844,deviceScaleFactor:1,isMobile:true});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);
  const report={browser:await browser.version(),warmup_displayed:warmup,measured_displayed:measured,results,
    delayed_client_tick_gap:gap,screenshot};
  const path=fileURLToPath(new URL('browser.json',output));
  await writeFile(path+'.tmp',JSON.stringify(report,null,2)+'\n');
  await rename(path+'.tmp',path);
} finally {
  await browser.close();
}
