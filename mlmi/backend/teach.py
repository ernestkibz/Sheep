"""Sheep trains a Checker model for its job (Ernest, 2026-10-06: "the orchestrator
should be able to train the checker model when it's plugged in").

The Checker's judge does two things (checker.Judge): it answers a question blind,
in a few words, and it says whether a given answer agrees with that reference
(YES or NO). The lessons teach exactly those two prompts, from:
  - exact arithmetic (the answers are computed, never guessed);
  - a bank of plain facts (capitals, counts, science, translations);
  - open requests, with on-topic and off-topic sentences;
  - the faults the rules know (loops, nonsense, role markers, "I don't know").
Every exam question (checker.EXAM, checker.EXAM_MORE) is kept out of the lessons,
so the exam measures what was learned, not what was memorised.

    lessons(seed) -> [{"messages": [...]}]       (the last message is the answer to learn)
    Trainer(...)  -> runs: configure, lessons, train (LoRA), merge, quantize, exam, swap
"""
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

TRAINING_TRIAL_STEPS = 8
TRAINING_TRIAL_EXAMPLES = 64  # one effective batch per optimizer step, spread across lesson types


def stratified_trial(rows, limit=TRAINING_TRIAL_EXAMPLES, seed=7):
    """Pick a small deterministic, even sample without dropping lesson types."""
    groups = {}
    rnd = random.Random(seed)
    for row in rows:
        groups.setdefault(str(row.get("kind") or "other"), []).append(row)
    for values in groups.values():
        rnd.shuffle(values)
    selected = []
    kinds = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for kind in kinds:
            if groups[kind] and len(selected) < limit:
                selected.append(groups[kind].pop())
    return selected


def _model_revision(path):
    """Exact content identity for the installed .ssf held fixed across a trial."""
    from .checker import model_revision
    return model_revision(path)


def reusable_baseline(saved_grade, path, revision, exam_revision, exam_size):
    """Reuse only a pass recorded for these exact weight bytes and exam revision."""
    if not isinstance(saved_grade, dict):
        return None
    try:
        same_path = str(Path(saved_grade.get("path", "")).resolve()).casefold() == str(Path(path).resolve()).casefold()
        right, of = int(saved_grade.get("right")), int(saved_grade.get("of"))
    except (TypeError, ValueError, OSError):
        return None
    if (not same_path or saved_grade.get("model_revision") != revision or
            saved_grade.get("exam_revision") != exam_revision or saved_grade.get("exam_ok") is not True or
            of != exam_size or not 0 <= right <= of):
        return None
    return {"right": right, "of": of, "frame": saved_grade.get("frame", "own"), "exam_ok": True,
            "accuracy": float(saved_grade.get("accuracy") or right / max(1, of)),
            "model_revision": revision, "cached": True}


from .paths import HOME, REPO

CAPITALS = [("Kenya", "Nairobi"), ("Uganda", "Kampala"), ("Tanzania", "Dodoma"), ("Rwanda", "Kigali"),
            ("Ethiopia", "Addis Ababa"), ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Egypt", "Cairo"),
            ("South Africa", "Pretoria"), ("Morocco", "Rabat"), ("Japan", "Tokyo"), ("China", "Beijing"),
            ("India", "New Delhi"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Portugal", "Lisbon"), ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"),
            ("Mexico", "Mexico City"), ("Australia", "Canberra"), ("Russia", "Moscow"), ("the United Kingdom", "London"),
            ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
            ("Poland", "Warsaw"), ("Greece", "Athens"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
            ("Saudi Arabia", "Riyadh"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Thailand", "Bangkok"),
            ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("the Philippines", "Manila"), ("South Korea", "Seoul"),
            ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Cuba", "Havana"),
            ("the Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Austria", "Vienna"), ("Denmark", "Copenhagen"),
            ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Senegal", "Dakar"), ("Sudan", "Khartoum"),
            ("Somalia", "Mogadishu"), ("Malawi", "Lilongwe"), ("Botswana", "Gaborone")]
FACTS = [("How many legs does an insect have?", "Six", ["Eight", "Four"]),
         ("How many legs does a dog have?", "Four", ["Six", "Two"]),
         ("How many days are in a week?", "Seven", ["Five", "Ten"]),
         ("How many minutes are in an hour?", "Sixty", ["One hundred", "Thirty"]),
         ("How many hours are in a day?", "Twenty-four", ["Twelve", "Thirty"]),
         ("What is the largest planet in the solar system?", "Jupiter", ["Saturn", "Mars"]),
         ("Which planet is called the red planet?", "Mars", ["Venus", "Neptune"]),
         ("What is the freezing point of water in Celsius?", "0 degrees", ["100 degrees", "32 degrees"]),
         ("Who wrote Romeo and Juliet?", "William Shakespeare", ["Charles Dickens", "Mark Twain"]),
         ("What is the chemical formula of water?", "H2O", ["CO2", "NaCl"]),
         ("What gas do people breathe out?", "Carbon dioxide", ["Oxygen", "Helium"]),
         ("What is the longest river in Africa?", "The Nile", ["The Congo", "The Niger"]),
         ("Which ocean is the largest?", "The Pacific Ocean", ["The Atlantic Ocean", "The Indian Ocean"]),
         ("How many continents are there?", "Seven", ["Five", "Nine"]),
         ("What colour is the sky on a clear day?", "Blue", ["Green", "Red"]),
         ("What do bees make?", "Honey", ["Milk", "Silk"]),
         ("Which is the tallest mountain in Africa?", "Mount Kilimanjaro", ["Mount Kenya", "Mount Elgon"]),
         ("What is the square root of 64?", "8", ["6", "16"]),
         ("Which animal is the largest on Earth?", "The blue whale", ["The elephant", "The giraffe"]),
         ("What is the hardest natural substance?", "Diamond", ["Gold", "Iron"]),
         ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Pablo Picasso", "Vincent van Gogh"]),
         ("What is the main language of Brazil?", "Portuguese", ["Spanish", "French"]),
         ("How many sides does a triangle have?", "Three", ["Four", "Five"]),
         ("What is the currency of Japan?", "The yen", ["The won", "The dollar"]),
         ("Which organ pumps blood around the body?", "The heart", ["The liver", "The lungs"]),
         ("What do plants need from the sun to make food?", "Light", ["Rain", "Salt"]),
         ("In which continent is Egypt?", "Africa", ["Asia", "Europe"]),
         ("What is the smallest prime number?", "2", ["1", "3"]),
         ("How many months have 31 days?", "Seven", ["Six", "Five"]),
         ("Which metal is liquid at room temperature?", "Mercury (the metal)", ["Iron", "Copper"]),
         ("What is the capital letter after B?", "C", ["D", "A"]),
         ("Which bird is a symbol of peace?", "The dove", ["The crow", "The eagle"]),
         ("What is H2O commonly called?", "Water", ["Salt", "Air"]),
         ("How many players are on a football team on the pitch?", "Eleven", ["Nine", "Fifteen"]),
         ("Which country is home to the kangaroo?", "Australia", ["Brazil", "Kenya"])]
WORDS = [("hello", "Spanish", "Hola", "Bonjour"), ("good morning", "French", "Bonjour", "Buenos dias"),
         ("thank you", "Spanish", "Gracias", "Danke"), ("yes", "German", "Ja", "Si"), ("water", "Swahili", "Maji", "Agua"),
         ("friend", "Swahili", "Rafiki", "Amigo"), ("goodbye", "Spanish", "Adios", "Au revoir"),
         ("cat", "French", "Chat", "Gato"), ("book", "Spanish", "Libro", "Livre"), ("house", "German", "Haus", "Casa"),
         ("please", "French", "S'il vous plait", "Por favor"), ("dog", "Spanish", "Perro", "Chien")]
OPEN = [("the moon", "The moon orbits the Earth about once a month."),
        ("coffee", "Coffee is brewed from roasted coffee beans."),
        ("football", "Football is played by two teams of eleven players."),
        ("rain", "Rain falls when water droplets in clouds grow heavy."),
        ("the desert", "A desert gets very little rain each year."),
        ("trees", "Trees take in carbon dioxide and give out oxygen."),
        ("the sun", "The sun is the star at the centre of our solar system."),
        ("elephants", "Elephants are the largest land animals."),
        ("music", "Music can lift people's mood."),
        ("computers", "Computers follow instructions written as programs."),
        ("bread", "Bread is baked from flour, water and yeast."),
        ("rivers", "Rivers carry fresh water to lakes and seas."),
        ("mountains", "Mountains rise high above the land around them."),
        ("books", "Books carry stories and knowledge from one person to another."),
        ("bicycles", "A bicycle has two wheels and is moved by pedals."),
        ("winter", "Winter is the coldest season of the year."),
        ("cats", "Cats are small animals often kept as pets."),
        ("phones", "Phones let people talk across long distances."),
        ("farming", "Farming grows the food that people eat."),
        ("schools", "Schools are places where children learn.")]
# Faults only: an honest "I don't know." is not nonsense and is never taught as one.
NONSENSE = ["Banana window the seven quickly purple.", "the the the the the the the the the the",
            "<|assistant|> <|user|> answer answer", "Answer: the of and to in is.", "Yes yes yes yes yes yes yes yes.",
            "Question: Question: Question: Question:", ""]
FORMS_SHORT = ["{a}.", "{a}", "It is {a}.", "The answer is {a}.", "{a}, of course."]
JUDGE_SYSTEM = None        # set from checker.JUDGE_COMPARE (one source of truth)


def _blind(q, a):
    return {"messages": [{"role": "user", "content": f"{q}\nAnswer in a few words."}, {"role": "assistant", "content": a}]}


def _compare(q, ref, given, yes):
    return {"messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"Question: {q}\nReference answer: {ref}\nGiven answer: {given}\n"
                                                     "Do they agree? YES or NO."},
                         {"role": "assistant", "content": "YES" if yes else "NO"}]}


def lessons(seed=7, arith=300):
    """The lessons, shuffled, with every exam question left out."""
    from . import checker
    global JUDGE_SYSTEM
    JUDGE_SYSTEM = checker.JUDGE_COMPARE
    held = {q.lower() for q, _, _ in checker.EXAM + checker.EXAM_MORE}
    rnd = random.Random(seed)
    out = []

    def add(kind, q, a, wrongs, sentence=None):
        if q.lower() in held:
            return
        out.append({"kind": kind, **_blind(q, a + "." if not a.endswith(".") else a)})
        for _ in range(2):
            form = rnd.choice(FORMS_SHORT)
            out.append({"kind": kind, **_compare(q, a, sentence if sentence and rnd.random() < 0.5 else form.format(a=a), True)})
        for w in wrongs[:2]:
            form = rnd.choice(FORMS_SHORT)
            out.append({"kind": kind, **_compare(q, a, form.format(a=w), False)})
        out.append({"kind": "faults", **_compare(q, a, rnd.choice(NONSENSE[:6]), False)})

    for c, cap in CAPITALS:
        others = rnd.sample([x for _, x in CAPITALS if x != cap], 2)
        add("facts", f"What is the capital of {c}?", cap, others, f"The capital of {c} is {cap}.")
    for q, a, wrong in FACTS:
        add("facts", q, a, wrong)
    for word, lang, right, wrong in WORDS:
        add("translation", f"Translate '{word}' into {lang}.", right, [wrong])
    n = 0
    while n < arith:
        a, b = rnd.randint(2, 99), rnd.randint(2, 30)
        op = rnd.choice(["plus", "minus", "times"])
        val = a + b if op == "plus" else a - b if op == "minus" else a * b
        q = f"What is {a} {op} {b}?"
        if q.lower() in held:
            continue
        near = val + rnd.choice([-10, -2, -1, 1, 2, 10])
        swap = int(str(abs(val))[::-1]) * (1 if val >= 0 else -1) if abs(val) > 9 and str(abs(val))[::-1] != str(abs(val)) else near + 3
        out.append({"kind": "arithmetic", **_blind(q, str(val))})
        out.append({"kind": "arithmetic", **_compare(q, str(val), rnd.choice(FORMS_SHORT).format(a=val), True)})
        out.append({"kind": "faults", **_compare(q, str(val), rnd.choice(FORMS_SHORT).format(a=rnd.choice([near, swap])), False)})
        n += 1
    for topic, sent in OPEN:
        q = f"Write one sentence about {topic}."
        if q.lower() in held:
            continue
        out.append({"kind": "open_requests", **_blind(q, sent)})
        other = rnd.choice([s for t, s in OPEN if t != topic])
        out.append({"kind": "open_requests", **_compare(q, sent, sent, True)})
        out.append({"kind": "open_requests", **_compare(q, sent, other, False)})
        out.append({"kind": "faults", **_compare(q, sent, rnd.choice(NONSENSE[:4]), False)})
    rnd.shuffle(out)
    return out


# ------------------------------------------------------------------ the whole training run
class Trainer:
    """configure -> lessons -> train (LoRA on the GPU lent by the chat model) ->
    merge -> quantize to Sheep -> exam both -> keep the better one that passes."""

    def __init__(self, checker, orch, path, source, done, training_mode="cpu"):
        self.checker, self.orch, self.path, self.source, self.done = checker, orch, path, Path(source), done
        self.training_mode = training_mode if training_mode in ("cpu", "gpu_bounded") else "cpu"
        self.state = {"state": "running", "stage": "starting", "path": path, "t": time.time(), "log": [],
                      "training_mode": self.training_mode, "training_steps": TRAINING_TRIAL_STEPS,
                      "stage_times": {}}
        self.proc, self.work = None, None
        self.thread_lock = threading.Lock()
        self.stop_requested = threading.Event()

    def say(self, stage, line=""):
        self.state["stage"] = stage
        if line:
            self.state["log"] = (self.state["log"] + [line])[-60:]

    def _publish_trial(self):
        if self.state.get("state") in ("running", "stopping"):
            self.checker.trial = {"state": "training", "path": self.path, "t": self.state.get("t", time.time())}
        else:
            self.checker.trial = dict(self.state)

    def start(self):
        self._launch(resume=False)
        return self.state

    def _launch(self, resume):
        with self.thread_lock:
            if self.state.get("state") in ("running", "starting") and self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("Checker training is already running")
            self.state.update(state="running", stage="resuming training" if resume else "starting",
                              training_mode=self.training_mode)
            threading.Thread(target=self._run, args=(resume,), daemon=True).start()

    def resume(self, training_mode=None):
        if self.state.get("state") not in ("paused", "stopped"):
            raise RuntimeError("Checker training is not paused")
        if self.state.get("before"):
            from . import checker as C
            if (self.state.get("baseline_revision") != _model_revision(self.path) or
                    self.state.get("exam_revision") != C.EXAM_REVISION):
                raise RuntimeError("the Checker model or held-out exam changed; start a fresh trial to compare safely")
        checkpoint = Path(self.state.get("checkpoint_dir") or "")
        has_checkpoint = (checkpoint / "latest.txt").is_file()
        # A run can be deferred before its first optimizer step when the GPU is
        # already warm. In that case, let the user resume on CPU or retry GPU;
        # there is no state to restore yet, so start that run from the beginning.
        waiting_for_cool = str(self.state.get("stage", "")).startswith("GPU is warm") and not self.state.get("step")
        if not has_checkpoint and not waiting_for_cool:
            raise RuntimeError("there is no saved Checker training checkpoint to resume")
        if training_mode in ("cpu", "gpu_bounded"):
            self.training_mode = training_mode
        if self.state.get("stop_file"):
            Path(self.state["stop_file"]).unlink(missing_ok=True)
        self._launch(resume=has_checkpoint)
        return self.state

    def stop(self):
        self.stop_requested.set()
        if self.proc and self.proc.poll() is None:
            self.state.update(state="stopping", stage="saving a safe checkpoint before stopping")
            stop_file = self.state.get("stop_file")
            if stop_file:
                Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
                Path(stop_file).touch()
        elif self.state.get("state") in ("paused", "running"):
            self.state.update(state="stopping", stage="stopping after the current safe operation")

    def _stop_at_boundary(self):
        if self.stop_requested.is_set():
            self.state.update(state="stopped", stage="stopped at the last safe checkpoint")
            self._publish_trial()
            return True
        return False

    def _examine(self, path, gpu_already_lent=False):
        from . import checker as C
        borrowed = False
        if self.orch and not gpu_already_lent:
            borrowed = self.orch.lend_gpu("examining a Checker", 2.0, wait=True, priority=2)
        try:
            return C.configure(self.checker, path)
        finally:
            if borrowed:
                self.orch.give_back_gpu()

    def _prepare_work(self):
        if self.work is None:
            work_root = HOME / "checker-training" / Path(self.path).stem
            work_root.mkdir(parents=True, exist_ok=True)
            work = work_root / f"run-{time.time_ns()}"
            work.mkdir(parents=False, exist_ok=False)
            self.work = work
            self.state["work"] = str(work)
        else:
            work = self.work
        checkpoint_dir = work / "checkpoint"
        stop_file = work / "stop.requested"
        lessons_file = work / "lessons.jsonl"
        merged = work / "merged"
        self.state.update(checkpoint_dir=str(checkpoint_dir), stop_file=str(stop_file))
        return checkpoint_dir, stop_file, lessons_file, merged

    def _baseline_exam(self, resume, lessons_file, stage_times):
        from . import checker as C
        before = self.state.get("before")
        if resume:
            if not lessons_file.is_file():
                raise RuntimeError("the saved training lessons are missing; the candidate was not changed")
            return before
        baseline_revision = _model_revision(self.path)
        self.state["baseline_revision"] = baseline_revision
        self.state["exam_revision"] = C.EXAM_REVISION
        saved_grade = self.checker.cfg.get("checker") or {}
        before = reusable_baseline(saved_grade, self.path, baseline_revision, C.EXAM_REVISION,
                                   len(C.EXAM) + len(C.EXAM_MORE))
        if before:
            self.state["baseline_exam_reused"] = True
            self.say("reusing the matching held-out exam", "the model file and exam revision both match")
        else:
            self.say("examining the model as it is")
            exam_started = time.monotonic()
            before = self._examine(self.path)
            stage_times["baseline_exam_seconds"] = round(time.monotonic() - exam_started, 1)
            before["model_revision"] = baseline_revision
        self.say("examining the model as it is", f"before training: {before['right']}/{before['of']} "
                 f"({before['frame']} frame)")
        self.state["before"] = before
        if self._stop_at_boundary():
            return None
        if not before.get("exam_ok"):
            self.state.update(state="rolled back", why="candidate failed its initial held-out Checker exam; training was not started")
            self._publish_trial()
            return None
        selected_lessons = stratified_trial(lessons(), TRAINING_TRIAL_EXAMPLES)
        with open(lessons_file, "w", encoding="utf-8") as f:
            for lesson in selected_lessons:
                f.write(json.dumps(lesson) + "\n")
        self.state["lesson_examples"] = len(selected_lessons)
        self.say("training", f"short trial: {len(selected_lessons)} balanced examples across every lesson type; "
                 f"at most {TRAINING_TRIAL_STEPS} steps")
        return before

    def _choose_training_device(self, gpu_lease, checker_training_advice, nvidia_thermal):
        from .orchestrator import probe
        device = "cpu"
        thermal = nvidia_thermal()
        pause_at = thermal.get("pause_at_c") if thermal.get("available") else None
        live_device = probe(0)
        available_ram_gb = (live_device.get("ram", {}).get("available") or 0) / 2**30
        source_bytes = sum(p.stat().st_size for pattern in ("*.safetensors", "*.bin", "*.pt")
                           for p in self.source.glob(pattern) if p.is_file()) or None
        if self.training_mode != "gpu_bounded":
            cpu_fit = checker_training_advice(cuda_ready=False, free_vram_gb=None,
                                               available_ram_gb=available_ram_gb,
                                               source_bytes=source_bytes, thermal=thermal)
            if not cpu_fit["cpu_available"]:
                raise RuntimeError(f"CPU training needs about {cpu_fit['cpu_need_gb']:.1f} GB of available RAM; "
                                   f"this computer currently has {available_ram_gb:.1f} GB free. "
                                   "Choose GPU training if its fit check passes.")
            return device, thermal, pause_at
        try:
            import torch
            cuda_ready = torch.cuda.is_available()
        except Exception:
            cuda_ready = False
        if not cuda_ready:
            raise RuntimeError("bounded GPU training requires Sheep's CUDA training runtime")
        projected_device = self.orch.freed(live_device) if self.orch else live_device
        projected_gpu = projected_device.get("gpu") or {}
        projected_free_gb = max(0, (projected_gpu.get("vram_total", 0) -
                                     projected_gpu.get("vram_used", 0)) / 2**30) if projected_gpu else None
        projected_fit = checker_training_advice(cuda_ready=True, free_vram_gb=projected_free_gb,
                                                 available_ram_gb=available_ram_gb,
                                                 source_bytes=source_bytes, thermal=thermal)
        self.state.update(preflight={"projected_free_vram_gb": projected_free_gb,
                                     "gpu_need_gb": projected_fit["gpu_need_gb"],
                                     "basis": "after Chat yields"})
        if not projected_fit["gpu_available"]:
            raise RuntimeError(projected_fit["reason"])
        if pause_at is not None and thermal.get("current_c", 0) >= pause_at:
            self.state.update(state="paused", stage="GPU is warm; choose CPU or resume after it cools",
                              why=f"GPU is warm at {thermal['current_c']} °C; training pauses at {pause_at} °C")
            self._publish_trial()
            return None
        if self.orch:
            gpu_lease["lent"] = self.orch.lend_gpu("training the Checker", projected_fit["gpu_need_gb"],
                                                   wait=True, priority=1, cancel_event=self.stop_requested)
            if not gpu_lease["lent"]:
                if self._stop_at_boundary():
                    return None
                raise RuntimeError("the GPU could not be reserved for Checker training")
        # Check the live driver after Chat yields; this decides whether loading may begin.
        live_device = probe(0)
        actual_gpu = live_device.get("gpu") or {}
        actual_free_gb = max(0, (actual_gpu.get("vram_total", 0) -
                                 actual_gpu.get("vram_used", 0)) / 2**30) if actual_gpu else None
        try:
            actual_free_bytes, _total_bytes = torch.cuda.mem_get_info()
            actual_free_gb = actual_free_bytes / 2**30
        except Exception:
            pass
        actual_fit = checker_training_advice(cuda_ready=True, free_vram_gb=actual_free_gb,
                                              available_ram_gb=(live_device.get("ram", {}).get("available") or 0) / 2**30,
                                              source_bytes=source_bytes, thermal=thermal)
        self.state["preflight"].update(actual_free_vram_gb=actual_free_gb,
                                        checked_after_chat_yield=bool(self.orch))
        if not actual_fit["gpu_available"]:
            actual_free_gb = actual_free_gb if actual_free_gb is not None else 0.0
            raise RuntimeError(f"After Chat yielded the GPU, only {actual_free_gb:.1f} GB is free; "
                               f"this trial needs about {actual_fit['gpu_need_gb']:.1f} GB. "
                               "Chat will return to its previous device; free GPU memory and retry.")
        return "cuda", thermal, pause_at

    def _run(self, resume=False):
        from . import checker as C
        from .quantizer import recipe
        from .training_policy import checker_training_advice, nvidia_thermal
        gpu_lease = {"lent": False}
        # Keep the original held-out exam across an in-process resume. Without
        # it, a resumed candidate cannot be compared with its incumbent.
        stage_times = self.state.setdefault("stage_times", {})

        try:
            checkpoint_dir, stop_file, lf, merged = self._prepare_work()

            before = self._baseline_exam(resume, lf, stage_times)
            if before is None:
                return

            device_details = self._choose_training_device(gpu_lease, checker_training_advice, nvidia_thermal)
            if device_details is None:
                return
            device, thermal, pause_at = device_details
            cmd = [sys.executable, str(REPO / "mlmi" / "backend" / "teach_worker.py"), "--src", str(self.source),
                   "--lessons", str(lf), "--out", str(merged), "--device", device,
                   "--checkpoint-dir", str(checkpoint_dir), "--stop-file", str(stop_file),
                   "--epochs", "1", "--max-steps", str(TRAINING_TRIAL_STEPS), "--checkpoint-every", "4"]
            if resume:
                cmd.append("--resume")
            if device == "cuda":
                if pause_at is not None:
                    cmd += ["--pause-at-c", str(pause_at)]
                else:
                    cmd += ["--max-chunk-seconds", "300"]
            self.state.update(device=device, thermal=thermal, queue="running")
            ok = False
            paused = stopped = False
            worker_started = time.monotonic()
            self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace",
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self.proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                ev = json.loads(line)
                if "step" in ev:
                    progress = round(ev["step"] / max(1, ev["of"]), 3)
                    elapsed = int(ev.get("s", 0) or 0)
                    chunk_steps = max(1, int(ev.get("chunk_step", ev["step"])))
                    update = {"progress": progress, "step": ev["step"], "steps": ev["of"]}
                    if elapsed > 0:
                        rate = elapsed / chunk_steps
                        update["eta_seconds"] = max(0, int(rate * (ev["of"] - ev["step"])))
                    self.state.update(**update)
                    self.say("training", f"step {ev['step']}/{ev['of']}, loss {ev.get('loss', 0):.3f}")
                if ev.get("stage"):
                    self.say(ev["stage"])
                if ev.get("timings"):
                    stage_times.update(ev["timings"])
                if ev.get("paused"):
                    paused = True
                    self.state.update(state="paused", why=ev.get("why", "training paused at a safe checkpoint"),
                                      temperature_c=ev.get("temperature_c"), checkpoint_dir=str(checkpoint_dir))
                if ev.get("stopped"):
                    stopped = True
                    self.state.update(state="stopped", why=ev.get("why", "stopped at a safe checkpoint"))
                if ev.get("ok"):
                    ok = True
                if ev.get("error"):
                    self.say("training", "error: " + ev["error"])
            self.proc.wait()
            stage_times.setdefault("worker_wall_seconds", round(time.monotonic() - worker_started, 1))
            if paused or stopped:
                self._publish_trial()
                return
            if not ok:
                raise RuntimeError("training did not finish: " + " | ".join(self.state["log"][-2:]))
            if self._stop_at_boundary():
                return
            # 5. quantize back to a Sheep file (Q6: a Checker is small, closeness matters more than bytes)
            self.say("quantizing the trained Checker")
            quantize_started = time.monotonic()
            cfg = json.load(open(merged / "config.json"))
            # its own folder: in ~/.mlmi/models it would be offered as a chat model
            dst = HOME / "checkers" / (Path(self.path).stem + "-checker.ssf")
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")          # renamed when whole, never left half-written
            q = subprocess.Popen(recipe(merged, part, cfg.get("num_hidden_layers", 24), level="high"), cwd=str(REPO),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            quantize_lines = []
            for line in q.stdout:
                line = line.strip()
                if not line:
                    continue
                quantize_lines.append(line)
                layer = re.search(r"layer\s+(\d+)/(\d+)", line, re.I)
                if layer:
                    current, total = int(layer.group(1)), max(1, int(layer.group(2)))
                    self.state["quantize_progress"] = round(current / total, 3)
                self.say("quantizing the trained Checker", line)
            q_rc = q.wait()
            if q_rc or not part.exists():
                stage_times["quantize_seconds"] = round(time.monotonic() - quantize_started, 1)
                raise RuntimeError("quantizing failed: " + " | ".join(quantize_lines[-3:])[-300:])
            self.state["quantize_progress"] = 1.0
            import os
            os.replace(part, dst)
            stage_times["quantize_seconds"] = round(time.monotonic() - quantize_started, 1)
            # 6. exam the trained one; keep the better that passes
            self.say("examining the trained Checker")
            exam_started = time.monotonic()
            after = self._examine(str(dst), gpu_already_lent=gpu_lease["lent"])
            stage_times["candidate_exam_seconds"] = round(time.monotonic() - exam_started, 1)
            after["model_revision"] = _model_revision(dst)
            if self._stop_at_boundary():
                return
            self.state["after"] = after
            self.say("examining the trained Checker", f"after training: {after['right']}/{after['of']}")
            best = max([(before, self.path), (after, str(dst))], key=lambda x: (x[0]["exam_ok"], x[0]["right"]))
            self.done(best[1], best[0])
            self.state.update(state="installed" if best[0]["exam_ok"] else "rolled back", chosen=best[1],
                              trained_better=best[1] == str(dst))
            self.say("done", f"kept {'the trained' if best[1] == str(dst) else 'the original'} Checker: "
                     f"{best[0]['right']}/{best[0]['of']}")
            self._publish_trial()
        except Exception as e:
            self.state.update(state="failed", why=str(e)[:300])
            self.say("failed", str(e)[:300])
            self._publish_trial()
        finally:
            if gpu_lease["lent"] and self.orch:
                self.orch.give_back_gpu()
