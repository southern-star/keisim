// Minimal static server for KeiView (no dependencies).   usage: node tools/serve.mjs [port]
// When node_modules/three exists (npm install), the page's import map is pointed at it, so the
// viewer also works offline / where cdn.jsdelivr.net is blocked.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml', '.md': 'text/markdown; charset=utf-8',
};
const LOCAL_THREE = fs.existsSync(path.join(root, 'node_modules/three/build/three.module.js'));

export function createServer() {
  return http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p.endsWith('/')) p += 'index.html';
    const file = path.join(root, p);
    if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); return res.end('not found'); }
      if (LOCAL_THREE && p === '/index.html') data = Buffer.from(String(data).replaceAll('https://cdn.jsdelivr.net/npm/three@0.170.0/', '/node_modules/three/'));
      res.writeHead(200, { 'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-store' });
      res.end(data);
    });
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.argv[2] || process.env.PORT || 5174);
  createServer().listen(port, () => console.log(`KeiView: http://localhost:${port}/?town=1000${LOCAL_THREE ? '  (three.js from node_modules)' : ''}`));
}
