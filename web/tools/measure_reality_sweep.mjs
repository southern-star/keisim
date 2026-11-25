// Measure decoded, visible live images in the actual local viewer on a configurable loopback port.
import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';

const [base,town] = process.argv.slice(2);
const endpoint = new URL(base);
assert.equal(endpoint.hostname, '127.0.0.1');
const browser = await puppeteer.launch({executablePath:'/usr/bin/google-chrome', headless:true,
  args:['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required']});
try {
  const page = await browser.newPage();
  await page.setViewport({width:1280,height:900});
  await page.bringToFront();
  const response = [];
  page.on('response',r=>{
    if(new URL(r.url()).pathname!=='/frame' || r.status()!==200) return;
    const h = r.headers();
    response.push({tick:Number(h['x-tick']),angle:Number(h['x-view-angle']),camera:Number(h['x-camera-ms']),model:Number(h['x-model-ms'])});
  });
  await page.evaluateOnNewDocument(()=>{
    window.displaySamples = [];
    document.addEventListener('DOMContentLoaded',()=>{
      const picture = document.getElementById('frame');
      const clock = document.getElementById('clock');
      const input = document.getElementById('orbitAngle');
      new MutationObserver(async ()=>{
        if(picture.hidden || !picture.src.startsWith('blob:')) return;
        const tick = Math.round(parseFloat(clock.textContent)*10);
        const next = (tick+1)%48;
        input.value = -3*(next<=12 ? next : next<=36 ? 24-next : next-48);
        input.dispatchEvent(new Event('input'));
        const url=picture.src;
        try { await picture.decode(); } catch { return; }
        requestAnimationFrame(()=>{
          if(picture.src!==url || picture.naturalWidth!==256 || document.hidden) return;
          if(window.displaySamples.at(-1)?.tick!==tick) window.displaySamples.push({tick,time:performance.now()});
        });
      }).observe(clock,{childList:true});
    });
  });
  await page.goto(base+'?mode=live&town='+town,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.displaySamples.at(-1)?.tick>=71,{timeout:120000});
  await page.click('#play');
  const samples = (await page.evaluate(()=>window.displaySamples)).filter(v=>v.tick>=8 && v.tick<=71);
  const timing = response.filter(v=>v.tick>=8 && v.tick<=71);
  const angles = timing.map(v=>v.angle);
  assert.ok(samples.length>=2 && Math.max(...angles)===36 && Math.min(...angles)===-36);
  const percentile = (values,p)=>[...values].sort((a,b)=>a-b)[Math.floor(p*(values.length-1))];
  const result = {town:Number(town),browser:await browser.version(),warmup:8,displayed:samples.length,
    skipped:samples.at(-1).tick-samples[0].tick+1-samples.length,
    fps:1000*(samples.length-1)/(samples.at(-1).time-samples[0].time),
    model_p50_ms:percentile(timing.map(v=>v.model),.5),model_p95_ms:percentile(timing.map(v=>v.model),.95),
    camera_p50_ms:percentile(timing.map(v=>v.camera),.5),min_angle:Math.min(...angles),max_angle:Math.max(...angles)};
  process.stdout.write(JSON.stringify(result));
} finally {
  await browser.close();
}
