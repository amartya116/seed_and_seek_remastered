"""Local web UI: upload a seed CSV and run the whole pipeline with one click.

    python app.py              # then open http://127.0.0.1:8000
    python app.py --port 8080

Steps (each runs as its own script, with this Python interpreter, from the project root):
  profiler.py -> querygroq.py -> equationextractorforembedding.py -> match_groq_queries.py -> select_tables.py
  -> join_heap_nodes.py (joins every heap node's tables from DuckDB, opened read-only)
Shown: output/groq_table_set_max_heap.txt and output/join_report.txt. Needs GROQ_API_KEY in .env and
data/seedandseek.duckdb (store_duckdb.py) for the join step.
Only listens on 127.0.0.1. Uses the standard library only (no web framework).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / 'data' / 'uploads'
OUTPUT = ROOT / 'output'
HEAP = OUTPUT / 'groq_table_set_max_heap.txt'
REPORT = OUTPUT / 'join_report.txt'
PAGE = ROOT / 'web' / 'index.html'
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STEP_TIMEOUT_SECONDS = 30 * 60
LOG_TAIL_CHARS = 6000


def pipeline_steps(csv_path: Path, threshold: float):
    return [
        ('Profile the seed', ['profiler.py', str(csv_path), '--output', 'output/profiler_output.json']),
        ('Queries and relations (Groq)', ['querygroq.py']),
        ('Seed equation', ['equationextractorforembedding.py']),
        ('ChromaDB matching and score matrix', ['match_groq_queries.py']),
        ('Heap of table sets', ['select_tables.py', '--threshold', f'{threshold:g}']),
        ('Join heap nodes (DuckDB, read-only)', ['join_heap_nodes.py', '--seed', str(csv_path)]),
    ]


_lock = threading.Lock()
_state = {'status': 'idle', 'file': None, 'threshold': None, 'steps': [], 'heap': None, 'report': None,
          'error': None}


def snapshot():
    with _lock:
        return json.loads(json.dumps(_state))


def run_pipeline(csv_path: Path, threshold: float):
    steps = pipeline_steps(csv_path, threshold)
    with _lock:
        _state.update(status='running', file=csv_path.name, threshold=threshold, heap=None, report=None, error=None,
                      steps=[{'name': name, 'status': 'pending', 'seconds': None, 'log': ''} for name, _ in steps])
    for old in (HEAP, REPORT):
        if old.exists():
            old.unlink()  # never show results of a previous run
    OUTPUT.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}

    for i, (name, args) in enumerate(steps):
        with _lock:
            _state['steps'][i]['status'] = 'running'
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True,
                                  text=True, encoding='utf-8', errors='replace', timeout=STEP_TIMEOUT_SECONDS)
            log, ok = (proc.stdout or '') + (proc.stderr or ''), proc.returncode == 0
        except subprocess.TimeoutExpired:
            log, ok = f'Timed out after {STEP_TIMEOUT_SECONDS} s', False
        except OSError as exc:
            log, ok = f'Could not start {args[0]}: {exc}', False
        with _lock:
            step = _state['steps'][i]
            step.update(status='done' if ok else 'failed', seconds=round(time.time() - start, 1),
                        log=log[-LOG_TAIL_CHARS:])
            if name.startswith('Heap') and ok and HEAP.exists():
                _state['heap'] = HEAP.read_text(encoding='utf-8')  # show the heap before the join step ends
            if not ok:
                lines = [ln for ln in log.strip().splitlines() if ln.strip()]
                _state['error'] = f'{name} failed: ' + (lines[-1] if lines else 'no output')
                for later in _state['steps'][i + 1:]:
                    later['status'] = 'skipped'
                _state['status'] = 'failed'
                return

    heap = HEAP.read_text(encoding='utf-8') if HEAP.exists() else None
    report = REPORT.read_text(encoding='utf-8') if REPORT.exists() else None
    with _lock:
        _state['heap'] = heap
        _state['report'] = report
        _state['status'] = 'done' if heap else 'failed'
        if not heap:
            _state['error'] = f'Pipeline finished but {HEAP.relative_to(ROOT)} was not written'


def safe_csv_name(name: str) -> str | None:
    base = re.sub(r'[^A-Za-z0-9._()-]+', '_', Path(name or '').name).strip('._')
    return base if base.lower().endswith('.csv') and len(base) > 4 else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        pass

    def _send(self, code, body, content_type='application/json; charset=utf-8', extra=None):
        data = body if isinstance(body, bytes) else (
            json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/status':
            self._send(200, snapshot())
        elif path == '/api/heap.txt':
            if HEAP.exists():
                self._send(200, HEAP.read_bytes(), 'text/plain; charset=utf-8',
                           {'Content-Disposition': f'attachment; filename="{HEAP.name}"'})
            else:
                self._send(404, {'error': 'No heap yet'})
        elif path == '/api/join_report.txt':
            if REPORT.exists():
                self._send(200, REPORT.read_bytes(), 'text/plain; charset=utf-8',
                           {'Content-Disposition': f'attachment; filename="{REPORT.name}"'})
            else:
                self._send(404, {'error': 'No join report yet'})
        else:
            self._send(404, {'error': 'Not found'})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/api/run':
            return self._send(404, {'error': 'Not found'})
        query = parse_qs(url.query)
        name = safe_csv_name(query.get('name', [''])[0])
        if not name:
            return self._send(400, {'error': 'Please upload a .csv file'})
        try:
            threshold = float(query.get('threshold', ['60'])[0])
        except ValueError:
            return self._send(400, {'error': 'Threshold must be a number'})
        if not 0 <= threshold <= 100:
            return self._send(400, {'error': 'Threshold must be between 0 and 100'})
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return self._send(400, {'error': 'The file is empty'})
        if length > MAX_UPLOAD_BYTES:
            return self._send(413, {'error': f'File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB'})
        body = self.rfile.read(length)
        with _lock:
            if _state['status'] == 'running':
                return self._send(409, {'error': 'A run is already in progress'})
            _state['status'] = 'running'  # reserve before the thread starts
        UPLOADS.mkdir(parents=True, exist_ok=True)
        target = UPLOADS / name
        target.write_bytes(body)
        threading.Thread(target=run_pipeline, args=(target.relative_to(ROOT), threshold), daemon=True).start()
        self._send(202, {'started': True, 'file': name})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Seed & Seek running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
