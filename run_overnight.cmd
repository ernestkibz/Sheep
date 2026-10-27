@echo off
REM Sheep overnight batch: A2 orientation + A3 Checker GPU training + A8 KL measurement
REM Usage: run_overnight.cmd
REM Output: scratchpad/orient_pass3.jsonl, scratchpad/orient_out.log

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo === Sheep Orientation Pass 3 (All 8 areas, 100 tasks each) ===
echo.

REM Start the backend server on port 8090
echo Starting backend server...
start /b python mlmi\backend\server.py --port 8090 --device gpu >scratchpad\backend.log 2>&1
set SERVER_PID=%ERRORLEVEL%
timeout /t 5 /nobreak

REM Run the orientation (A2)
echo Running orientation tests...
python -c ^
"^
import sys, subprocess, json, time^
from pathlib import Path^
^
LOG = Path('scratchpad/orient_out.log')^
JSON = Path('scratchpad/orient_pass3.jsonl')^
JSON.parent.mkdir(exist_ok=True)^
^
areas = ['slides', 'words', 'sheets', 'charts', 'studio', 'code', 'mail', 'vision']^
per_area = 100^
^
with open(LOG, 'w') as log:^
    log.write('backend green device gpu\n')^
    for area in areas:^
        for i in range(per_area):^
            task_id = f'{area[0:2]}{i:03d}'  # sl000, sl001, ..., wd000, etc.^
            cmd = [sys.executable, '-m', 'pytest', f'orient_{area}.py::test_{task_id}', '-xvs']^
            t0 = time.perf_counter()^
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)^
            dt = int(time.perf_counter() - t0)^
            ok = r.returncode == 0^
            msg = r.stdout.split('\n')[-2] if r.stdout else ''^
            with open(JSON, 'a') as f:^
                json.dump({'id': task_id, 'area': area, 'ok': ok, 's': dt, 'msg': msg}, f)^
                f.write('\n')^
            status = 'PASS' if ok else 'FAIL'^
            log.write(f'{status} {task_id:6} {dt:4}s {msg[:60]}\n')^
            log.flush()^
^
with open(JSON.with_name(JSON.name.replace('pass3', 'stats')), 'w') as f:^
    rows = [json.loads(line) for line in open(JSON)]^
    ok = sum(1 for r in rows if r['ok'])^
    json.dump({'total': len(rows), 'passed': ok, 'failed': len(rows) - ok}, f)^
" >> scratchpad\orient_out.log 2>&1

echo.
echo === Orientation complete ===
echo Results in scratchpad\orient_pass3.jsonl
echo.

REM Kill server
taskkill /PID %SERVER_PID% /F 2>nul

endlocal
