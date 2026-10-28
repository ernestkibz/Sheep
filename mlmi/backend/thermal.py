"""The heat governor: Sheep paces itself so the laptop never overheats.

A laptop running a model flat out for hours shut itself down from heat
(2026-10-12). Nothing in Sheep acted on temperature: the dashboard lamp showed
it, but checks, chat steps and GPU jobs ran on at full speed. The governor
reads the GPU (nvidia-smi) and the CPU (Windows' thermal zone counter) every
few seconds, with the CPU load, and turns them into one level:

    cool      everything runs
    warm      everything runs at full speed (shown on the dashboard)
    hot       background work runs in bursts with short rests
    critical  background work and new background GPU work wait; the lamp says why
    danger    (DANGER_GPU / DANGER_CPU, near where the card slows itself and the laptop has shut
              down) the person's own work waits too: chat steps (bounded) and their pictures/video

Each level has a lower release point (hysteresis) so it does not flicker, and a
CPU kept above 90% for two minutes while warm counts as hot: sustained load is
what heats a laptop, not one burst. Settings can turn it off ("thermal_guard").
"""
import os
import re
import shutil
import subprocess
import threading
import time

LEVELS = ("cool", "warm", "hot", "critical")
# (warm, hot, critical) in degrees C; release is 5 C below each. The CPU figure
# is Windows' ACPI thermal zone, which on this laptop idles in the high 70s and
# sits near 89 C under light load: a warm line at 85 paused the background
# checks for half an hour (2026-10-12), so it starts at 90.
# The three shutdowns of 2026-10-12 were not heat: each came when mains power
# dropped and the battery ran down to 0 under load (Kernel-Power 105, AcOnline
# false, then event 41). So heat lines sit at this laptop's real limits (the
# RTX 3050 Laptop slows itself at 97 C; the chassis zone reads 96-98 C in normal
# GPU work), and power has its own guard below.
# Ernest, 2026-10-27: "too rigid; let things run, only pause at 99". Hot is now the same line as
# critical, so nothing rests or slows below it: warm (88) is shown, nothing more.
GPU_AT = (88.0, 95.0, 95.0)   # Ernest, 2026-10-28: 95 after a hot shutdown at 13:52 with no pause logged
CPU_AT = (97.0, 103.0, 103.0)
# power: on battery the optional background work waits so the charge goes to the
# person's own requests; low battery also holds new GPU work
BATTERY_LOW, BATTERY_CRITICAL = 25, 10
# Hot paces background work in bursts (run, then rest), it does not stop it: the
# zone on this laptop tracks the whole chassis (96-98 C at 10% CPU with the GPU
# busy) and cools slowly, so a "wait until cool" held the checks for half an
# hour (2026-10-12). Warm runs at full speed; only critical stops background work.
HOT_RUN_S, HOT_REST_S = 75.0, 20.0
RELEASE = 5.0
LOAD_HIGH, LOAD_FOR_S = 90.0, 120.0
PERIOD_S = 5.0
# A CPU zone that never moves while the load swings is a fixed BIOS value, not a reading: this
# laptop's only ACPI zone said 373.0 K (99.85 °C) for hours at 14% and 80% load alike, and the
# guard held every background check (2026-10-20). It is then left out, and the GPU decides.
STUCK_SAMPLES, STUCK_LOAD_SPAN = 24, 25.0
# The person's own work goes at full speed until the danger line (Ernest, 2026-10-26: no speed
# lost to llama.cpp, which never waits, but no shutdown mid-work). The RTX 3050 Laptop slows itself
# at 97 C and this laptop reached 99 C before it cut out, so the line sits just under the card's own.
DANGER_GPU, DANGER_CPU = GPU_AT[2], CPU_AT[2]
# Before the danger line the engine slows smoothly instead (Ernest, 2026-10-27): from PACE_FROM it
# rests a little after each written token, rising to PACE_MAX_MS at the danger line. Below
# PACE_FROM nothing changes: this laptop runs 91-93 C in normal load and loses no speed there.
PACE_FROM_GPU, PACE_FROM_CPU, PACE_MAX_MS = DANGER_GPU, DANGER_CPU, 24     # no slowdown below the pause line
# how long a chat step waits for heat to fall before going on anyway (the person is waiting)
WAIT_CRITICAL_S = 45.0
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _gpu_temp():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=_NOWIN).stdout
        vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", out)]
        return max(vals) if vals else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _cpu_temp():
    """Windows' ACPI thermal zones (Kelvin), readable without admin rights."""
    if os.name != "nt":
        try:
            import psutil
            t = psutil.sensors_temperatures()
            vals = [s.current for rows in t.values() for s in rows if s.current]
            return max(vals) if vals else None
        except Exception:
            return None
    exe = shutil.which("typeperf")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, r"\Thermal Zone Information(*)\Temperature", "-sc", "1"],
                             capture_output=True, text=True, timeout=8, creationflags=_NOWIN).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        cells = [c.strip('" ') for c in line.split('","')]
        if len(cells) > 1 and re.match(r"\d\d/\d\d/\d{4}", cells[0]):
            vals = []
            for c in cells[1:]:
                try:
                    k = float(c)
                except ValueError:
                    continue
                if 250 < k < 400:          # Kelvin
                    vals.append(k - 273.15)
            return round(max(vals), 1) if vals else None
    return None


def fake_power():
    """The journeys' battery (tooling/ui_audit): MLMI_FAKE_POWER is "battery:55" or "ac", or "@<file>"
    holding one of those, so a run can unplug and plug in again. None when unset."""
    fake = os.environ.get("MLMI_FAKE_POWER", "")
    if fake.startswith("@"):
        try:
            fake = open(fake[1:], encoding="utf-8").read().strip()
        except OSError:
            fake = ""
    if not fake:
        return None
    return {"plugged": fake == "ac", "percent": int(fake.split(":")[1]) if ":" in fake else 100}


def _power():
    """{"plugged": bool, "percent": int}, or None on a machine without a battery."""
    if fake_power():
        return fake_power()
    try:
        import psutil
        b = psutil.sensors_battery()
        return {"plugged": bool(b.power_plugged), "percent": int(round(b.percent))} if b else None
    except Exception:
        return None


def _cpu_load():
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def level_of(temp, at, current):
    """The level for one sensor, holding the current level until temp falls RELEASE below it."""
    if temp is None:
        return 0
    up = 0
    for i, t in enumerate(at):
        if temp >= t:
            up = i + 1
    if up >= current:
        return up
    # falling: stay at the current level until below its release point
    while current > up and temp < at[current - 1] - RELEASE:
        current -= 1
    return max(up, current)


class Governor:
    def __init__(self, enabled=lambda: True, read=None, read_power=None):
        self.enabled = enabled
        self.read = read or (lambda: (_gpu_temp(), _cpu_temp(), _cpu_load()))
        self.read_power = read_power or _power
        self.power = None              # None = mains (or no battery)
        self._told_battery = False
        self.gpu = self.cpu = self.load = None
        self.cpu_stuck = False
        self._cpu_seen = []            # (cpu, load) of recent samples, to spot a reading that never moves
        self.g_level = self.c_level = 0
        self.level = 0
        self.danger = False
        self.why = ""
        self._loaded_since = None
        self._warm_since = None
        self.changed = threading.Condition()
        self.log = []                  # level changes, newest last
        self._thread = None

    # ---------------------------------------------------------------- reading
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while True:
            try:
                self.update(*self.read(), power=self.read_power())
            except Exception:
                pass
            time.sleep(PERIOD_S)

    def update(self, gpu, cpu, load, now=None, power=None):
        now = time.monotonic() if now is None else now
        if cpu is not None:
            self._cpu_seen = (self._cpu_seen + [(cpu, load)])[-STUCK_SAMPLES:]
            loads = [l for _, l in self._cpu_seen if l is not None]
            if len({c for c, _ in self._cpu_seen}) > 1:
                self.cpu_stuck = False
            elif (len(self._cpu_seen) >= STUCK_SAMPLES and loads
                  and max(loads) - min(loads) >= STUCK_LOAD_SPAN):
                self.cpu_stuck = True
        if self.cpu_stuck:
            cpu = None
        self.gpu, self.cpu, self.load = gpu, cpu, load
        self.power = power
        if self.power_state == "ac":
            self._told_battery = False
        self.g_level = level_of(gpu, GPU_AT, self.g_level)
        self.c_level = level_of(cpu, CPU_AT, self.c_level)
        lv = max(self.g_level, self.c_level)
        if load is not None and load >= LOAD_HIGH:
            if self._loaded_since is None:
                self._loaded_since = now
        else:
            self._loaded_since = None
        sustained = self._loaded_since is not None and now - self._loaded_since >= LOAD_FOR_S
        # a busy CPU no longer counts as hot (2026-10-27): only the readings decide
        hot_now = (gpu is not None and gpu >= DANGER_GPU) or (cpu is not None and cpu >= DANGER_CPU)
        cool_now = (gpu is None or gpu < DANGER_GPU - RELEASE) and (cpu is None or cpu < DANGER_CPU - RELEASE)
        danger = hot_now or (self.danger and not cool_now)
        if not self.enabled():
            lv, danger = 0, False
        parts = []
        if gpu is not None:
            parts.append(f"GPU {gpu:.0f} °C")
        if cpu is not None:
            parts.append(f"CPU {cpu:.0f} °C")
        if sustained:
            parts.append(f"CPU busy {load:.0f}% for {int(now - self._loaded_since)} s")
        why = ", ".join(parts)
        with self.changed:
            if lv != self.level:
                self.log.append({"t": time.time(), "from": LEVELS[self.level], "to": LEVELS[lv], "why": why})
                del self.log[:-40]
            self.level, self.why, self.danger = lv, why, danger
            self.changed.notify_all()

    # ---------------------------------------------------------------- deciding
    @property
    def name(self):
        return LEVELS[self.level]

    @property
    def power_state(self):
        """ac, battery, low or critical. With the guard off, always ac."""
        p = self.power
        if not p or p.get("plugged") or not self.enabled():
            return "ac"
        pct = p.get("percent")
        if pct is not None and pct <= BATTERY_CRITICAL:
            return "critical"
        if pct is not None and pct <= BATTERY_LOW:
            return "low"
        return "battery"

    def _battery_words(self):
        return f"running on battery at {self.power.get('percent')}%"

    def background_wait(self, now=None):
        """Why background work should wait now, or None. Critical: it waits. Hot: it
        runs in bursts (HOT_RUN_S on, HOT_REST_S resting). Warm and cool: full speed."""
        now = time.monotonic() if now is None else now
        if self.power_state != "ac":
            # the mains dropped and a full-speed background run took the battery to 0
            return f"this computer is {self._battery_words()}; background checks wait for the charger"
        if self.level >= 3:
            return f"this computer is very hot ({self.why}); background checks wait until it cools"
        if self.level == 2:
            if self._warm_since is None:
                self._warm_since = now
            if (now - self._warm_since) % (HOT_RUN_S + HOT_REST_S) >= HOT_RUN_S:
                return f"this computer is hot ({self.why}); a short rest between background checks"
            return None
        self._warm_since = None
        return None

    def pace_ms(self):
        """The rest after each written token, in ms: 0 below PACE_FROM, rising to PACE_MAX_MS
        at the danger line (and past it)."""
        if not self.enabled():
            return 0

        def share(t, start, stop):
            if t is None or t < start:
                return 0.0
            return min(1.0, (t - start) / max(0.1, stop - start))
        f = max(share(self.gpu, PACE_FROM_GPU, DANGER_GPU), share(self.cpu, PACE_FROM_CPU, DANGER_CPU))
        if self.danger:
            f = 1.0
        return int(round(f * PACE_MAX_MS))

    def gpu_wait(self, person=False):
        """Why new GPU work should wait, or None. The person's own pictures and video wait only at
        the danger line; background GPU work already at critical."""
        if self.power_state in ("low", "critical"):
            return f"this computer is {self._battery_words()}; new GPU work waits for the charger"
        hot = self.danger if person else self.level >= 3
        return f"this computer is too hot for more GPU work ({self.why})" if hot else None

    def pace(self, emit=None, stop=None):
        """Before a chat step: the person is waiting, so chat goes at full speed unless the
        machine is critical; then it waits (bounded) for it to fall back to hot.
        Returns the seconds waited."""
        if emit and self.power_state in ("low", "critical") and not self._told_battery:
            # said once per time on battery: the person's work goes on, but they should know
            self._told_battery = True
            emit({"type": "thinking", "route": "local",
                  "what": f"This computer is {self._battery_words()}: plug in the charger so Sheep is not cut off"})
        if not self.danger:
            return 0.0
        t0 = time.monotonic()
        limit = WAIT_CRITICAL_S
        said = None
        with self.changed:
            while self.danger and time.monotonic() - t0 < limit:
                if stop is not None and stop.is_set():
                    break
                msg = f"Letting this computer cool before the next step ({self.why})"
                if emit and msg != said:
                    emit({"type": "thinking", "route": "local", "what": msg})
                    said = msg
                self.changed.wait(timeout=2.0)
        return round(time.monotonic() - t0, 1)

    def snapshot(self):
        return {"level": self.name, "why": self.why, "gpu": self.gpu, "cpu": self.cpu, "load": self.load,
                "cpu_sensor": "fixed value, ignored" if self.cpu_stuck else "ok",
                "on": bool(self.enabled()), "log": self.log[-10:],
                "power": self.power_state, "battery": (self.power or {}).get("percent"),
                "danger": self.danger, "pace_ms": self.pace_ms(),
                "at": {"gpu": GPU_AT, "cpu": CPU_AT, "release": RELEASE, "danger_gpu": DANGER_GPU, "danger_cpu": DANGER_CPU}}
