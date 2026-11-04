#!/usr/bin/env python3
"""Soak test for obs-spectrogram.

Starts OBS in a throwaway profile (fresh XDG_CONFIG_HOME) with the plugin
from the build directory, then drives it through obs-websocket for a while:
creating and removing spectrograms, switching audio sources, renaming and
deleting the source being captured, toggling visibility, studio mode and
scene collections.  Along the way it checks that the spectrogram actually
animates, and afterwards that OBS shut down cleanly with zero leaked
allocations and no runaway RSS growth.

Linux only.  Needs OBS >= 29.1 with obs-websocket 5, Xvfb (unless
--no-xvfb), ffmpeg, and the Python packages websockets and Pillow.

    tests/soak/soak.py --build-dir build --minutes 10
"""

import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from PIL import Image
from websockets.sync.client import connect

NOT_READY = 207  # obs-websocket RequestStatus::NotReady
PLUGIN = "obs-spectrogram"
STAGED_MODULE = PLUGIN + "-under-test"
KIND = "spectrogram_source"
OUTPUT_MIX_ID = "__output_mix__"
SCENES = ["Soak A", "Soak B"]


def log(msg):
	print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SoakFailure(Exception):
	pass


# ── obs-websocket v5 client ─────────────────────────────────────────────────

class ObsClient:
	def __init__(self, port):
		self.ws = connect(f"ws://127.0.0.1:{port}", max_size=64 * 1024 * 1024)
		hello = json.loads(self.ws.recv())
		assert hello["op"] == 0, hello
		self.ws.send(json.dumps({"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 0}}))
		identified = json.loads(self.ws.recv())
		assert identified["op"] == 2, identified

	def call(self, request_type, data=None, check=True):
		# the server comes up before OBS has finished loading
		deadline = time.time() + 30
		while True:
			status, response = self.send_request(request_type, data)
			if status.get("code") != NOT_READY or time.time() > deadline:
				break
			time.sleep(0.5)
		if check and not status["result"]:
			raise SoakFailure(f"{request_type} {data} failed: {status}")
		return response

	def send_request(self, request_type, data):
		request_id = str(uuid.uuid4())
		self.ws.send(json.dumps({"op": 6, "d": {
			"requestType": request_type,
			"requestId": request_id,
			"requestData": data or {},
		}}))
		while True:
			msg = json.loads(self.ws.recv(timeout=30))
			if msg["op"] == 7 and msg["d"]["requestId"] == request_id:
				return msg["d"]["requestStatus"], msg["d"].get("responseData", {})

	def close(self):
		self.ws.close()


def connect_when_ready(proc, port, timeout):
	deadline = time.time() + timeout
	while time.time() < deadline:
		if proc.poll() is not None:
			raise SoakFailure(f"OBS exited during startup (code {proc.returncode})")
		try:
			return ObsClient(port)
		except (OSError, AssertionError):
			time.sleep(0.5)
	raise SoakFailure("timed out waiting for obs-websocket")


# ── environment setup ───────────────────────────────────────────────────────

def write_file(path, content):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		f.write(content)


def find_free_port():
	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[1]


def create_profile(root, port):
	"""Seed a fresh OBS config: no first-run wizard, websocket enabled."""
	cfg = os.path.join(root, "config", "obs-studio")
	# OBS >= 31 reads these from user.ini, older versions from global.ini.
	# No tray icon: it would show up on the real desktop's D-Bus session.
	for ini in ("user.ini", "global.ini"):
		write_file(os.path.join(cfg, ini),
		           "[General]\nFirstRun=true\n\n[BasicWindow]\nSysTrayEnabled=false\n")
	write_file(os.path.join(cfg, "plugin_config", "obs-websocket", "config.json"), json.dumps({
		"first_load": False,
		"server_enabled": True,
		"server_port": port,
		"alerts_enabled": False,
		"auth_required": False,
		"server_password": "",
	}))
	return os.path.join(root, "config")


def stage_plugin(root, build_dir, repo_dir):
	"""Lay the build out the way OBS_PLUGINS_PATH / _DATA_PATH expect.

	It's staged under a different module name so that a copy installed
	system-wide can be disabled (see disable_installed_plugin) instead of
	shadowing the build under test."""
	bin_dir = os.path.join(root, "plugins", "bin")
	data_dir = os.path.join(root, "plugins", "data", STAGED_MODULE)
	os.makedirs(bin_dir)
	plugin_path = os.path.join(bin_dir, f"{STAGED_MODULE}.so")
	shutil.copy(os.path.join(build_dir, f"{PLUGIN}.so"), plugin_path)
	shutil.copytree(os.path.join(repo_dir, "data"), data_dir)
	return plugin_path, bin_dir, os.path.dirname(data_dir)


def disable_installed_plugin(config_home):
	"""Disable any installed obs-spectrogram through the plugin manager
	(OBS >= 32).  On older OBS an installed copy wins, and the soak
	stops with an explanation instead of testing the wrong binary."""
	write_file(os.path.join(config_home, "obs-studio", "plugin_manager", "modules.json"),
	           json.dumps([{
		           "display_name": "Spectrogram", "module_name": PLUGIN, "id": "",
		           "version": "", "enabled": False,
		           "sources": [KIND], "outputs": [], "encoders": [], "services": [],
	           }]))


def create_test_audio(root):
	"""A sweep plus a chord, so both ends of the spectrum light up."""
	path = os.path.join(root, "test-signal.wav")
	subprocess.run([
		"ffmpeg", "-loglevel", "error", "-y",
		"-f", "lavfi", "-i", "sine=f=220:d=4",
		"-f", "lavfi", "-i", "sine=f=1760:d=4",
		"-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(50+t*1200)*t):d=4",
		"-filter_complex", "amix=inputs=3,volume=2",
		"-ar", "48000", "-ac", "2", path,
	], check=True)
	return path


def find_asan_runtime():
	out = subprocess.run(["gcc", "-print-file-name=libasan.so"],
	                     capture_output=True, text=True).stdout.strip()
	path = os.path.realpath(out)
	if not os.path.isfile(path):
		raise SoakFailure("--asan: gcc's libasan.so not found")
	return path


def start_xvfb():
	"""Start a private X server; returns (process, DISPLAY).  Retries,
	because Xvfb instances started at the same moment can be handed the
	same display number, and then only one of them works."""
	for _ in range(5):
		r, w = os.pipe()
		xvfb = subprocess.Popen(["Xvfb", "-displayfd", str(w), "-screen", "0", "1920x1080x24",
		                         "-nolisten", "tcp"], pass_fds=(w,),
		                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		os.close(w)
		number = os.read(r, 16).decode().strip()
		os.close(r)
		display = f":{number}"
		time.sleep(0.5)
		probe = subprocess.run(["xdotool", "getdisplaygeometry"], capture_output=True,
		                       env=dict(os.environ, DISPLAY=display))
		if number and xvfb.poll() is None and probe.returncode == 0:
			return xvfb, display
		xvfb.kill()
		xvfb.wait()
		time.sleep(random.random())
	raise SoakFailure("Xvfb failed to start")


def launch_obs(args, display, config_home, bin_dir, data_dir, out_path):
	env = dict(os.environ)
	env["DISPLAY"] = display
	env["XDG_CONFIG_HOME"] = config_home
	env["OBS_PLUGINS_PATH"] = bin_dir
	env["OBS_PLUGINS_DATA_PATH"] = data_dir
	if args.asan:
		env["LD_PRELOAD"] = find_asan_runtime()
		# leaks are counted by libobs itself; ASan is here for memory errors
		env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:log_path=" + \
			os.path.join(os.path.dirname(out_path), "asan")

	out = open(out_path, "w")
	proc = subprocess.Popen([args.obs, "--multi", "--disable-shutdown-check",
	                         "--disable-missing-files-check"],
	                        env=env, stdout=out, stderr=subprocess.STDOUT)
	return proc, out


def quit_obs(proc, display):
	"""Quit via Ctrl+Q, like a user would.  SIGTERM takes a different
	shutdown path, which in OBS 32 crashes in the frontend's own teardown
	before the leak counter is printed."""
	env = dict(os.environ, DISPLAY=display)
	windows = subprocess.run(["xdotool", "search", "--pid", str(proc.pid), "--name", "^OBS "],
	                         env=env, capture_output=True, text=True).stdout.split()
	if windows:
		subprocess.run(["xdotool", "key", "--window", windows[0], "ctrl+q"], env=env)
		try:
			proc.wait(timeout=60)
			return None
		except subprocess.TimeoutExpired:
			pass
	proc.terminate()
	try:
		proc.wait(timeout=60)
		return "Ctrl+Q did not quit OBS; fell back to SIGTERM"
	except subprocess.TimeoutExpired:
		proc.kill()
		return "OBS did not exit within 60 s of SIGTERM"


def read_rss_mb(pid):
	with open(f"/proc/{pid}/status") as f:
		for line in f:
			if line.startswith("VmRSS:"):
				return int(line.split()[1]) / 1024
	return 0.0


def find_undestroyed_spectrograms(stdout_text):
	"""Every spectrogram created must also be destroyed.  libobs logs both
	at debug level, which only goes to stdout."""
	created = re.findall(rf"^debug: source '(.+?)' \({KIND}\) created", stdout_text, re.M)
	destroyed = set(re.findall(r"^debug: source '(.+?)' destroyed", stdout_text, re.M))
	return [name for name in created if name not in destroyed]


def read_obs_log(config_home):
	logs = os.path.join(config_home, "obs-studio", "logs")
	files = sorted(os.listdir(logs)) if os.path.isdir(logs) else []
	if not files:
		return ""
	with open(os.path.join(logs, files[-1]), errors="replace") as f:
		return f.read()


# ── checks ──────────────────────────────────────────────────────────────────

def measure_lit_fraction(obs, name):
	"""Fraction of pixels brighter than the default palette's black floor."""
	shot = obs.call("GetSourceScreenshot", {"sourceName": name, "imageFormat": "png"})
	png = base64.b64decode(shot["imageData"].split(",", 1)[1])
	gray = Image.open(io.BytesIO(png)).convert("L")
	hist = gray.histogram()
	return sum(hist[32:]) / max(1, sum(hist))


def expect_animating(obs, name, what, wait=2.0):
	time.sleep(wait)
	lit = measure_lit_fraction(obs, name)
	if lit < 0.02:
		raise SoakFailure(f"{what}: spectrogram '{name}' shows no signal ({lit:.1%} lit)")
	log(f"  ok: {what} ({lit:.0%} lit)")


def get_audio_ref(obs, name):
	return obs.call("GetInputSettings", {"inputName": name})["inputSettings"].get("audio_source")


def list_audio_choices(obs, name):
	items = obs.call("GetInputPropertiesListPropertyItems",
	                 {"inputName": name, "propertyName": "audio_source"})["propertyItems"]
	return {i["itemValue"]: i["itemName"] for i in items}


# ── scenario ────────────────────────────────────────────────────────────────

# random operations and their relative weights
OPS = {
	"create": 6, "remove": 4, "settings": 10, "resize": 3, "audio": 8, "toggle": 6,
	"props": 4, "scene": 4, "studio": 2, "mute": 2, "recreate_media": 1,
}


class Soak:
	def __init__(self, obs, audio_file, rng):
		self.obs = obs
		self.audio_file = audio_file
		self.rng = rng
		self.spectrograms = {}  # name -> scene
		self.media_name = "Test Signal"
		self.media_uuid = None
		self.counter = 0

	def create_media(self, scene):
		r = self.obs.call("CreateInput", {
			"sceneName": scene,
			"inputName": self.media_name,
			"inputKind": "ffmpeg_source",
			"inputSettings": {"local_file": self.audio_file, "looping": True,
			                  "close_when_inactive": False},
			"sceneItemEnabled": True,
		})
		self.media_uuid = r["inputUuid"]

	def create_spectrogram(self, scene, settings):
		self.counter += 1
		name = f"Spectrogram {self.counter}"
		self.obs.call("CreateInput", {
			"sceneName": scene, "inputName": name, "inputKind": KIND,
			"inputSettings": settings, "sceneItemEnabled": True,
		})
		self.spectrograms[name] = scene
		return name

	def randomize_settings(self):
		rng = self.rng
		s = {
			"analysis_mode": rng.choice([0, 1]),
			"fft_size": rng.choice([256, 512, 1024, 2048, 4096, 8192]),
			"overlap": rng.choice([0.0, 0.5, 0.75, 0.95]),
			"window_type": rng.randrange(4),
			"resonate_bins": rng.choice([2, 60, 120, 512, 1024]),
			"resonate_hop": rng.choice([64, 128, 512, 2048]),
			"resonate_smooth": rng.random() < 0.5,
			"log_scale": rng.random() < 0.7,
			"min_freq": rng.choice([1.0, 20.0, 200.0]),
			"max_freq": rng.choice([100.0, 8000.0, 20000.0, 24000.0]),
			"gain_db": rng.uniform(-40, 40),
		}
		return s

	def randomize_size(self):
		return {"width": self.rng.choice([64, 320, 800, 1920, 4096]),
		        "height": self.rng.choice([64, 300, 1080, 4096])}

	def pick_audio_ref(self):
		return self.rng.choice([
			self.media_uuid, self.media_uuid, self.media_uuid,
			self.media_name,            # legacy name-based setting
			OUTPUT_MIX_ID, "",
			str(uuid.uuid4()),          # missing source
		])

	# ── scripted checks, run once up front ─────────────────────────────

	def run_functional_checks(self):
		obs = self.obs
		log("functional checks")
		for scene in SCENES:
			obs.call("CreateScene", {"sceneName": scene})
		obs.call("SetCurrentProgramScene", {"sceneName": SCENES[0]})
		self.create_media(SCENES[0])

		probe = self.create_spectrogram(SCENES[0], {"audio_source": self.media_uuid})
		choices = list_audio_choices(obs, probe)
		if self.media_uuid not in choices or OUTPUT_MIX_ID not in choices or "" not in choices:
			raise SoakFailure(f"audio source list is missing entries: {choices}")
		if any(KIND in v or v.startswith("Spectrogram") for v in choices.values()):
			raise SoakFailure(f"audio source list offers a spectrogram: {choices}")
		log(f"  ok: audio source list = {sorted(choices.values())}")
		expect_animating(obs, probe, "capturing media source by UUID")

		for mode in (1, 0):
			obs.call("SetInputSettings", {"inputName": probe,
			                              "inputSettings": {"analysis_mode": mode}})
			expect_animating(obs, probe, f"analysis mode {'Resonate' if mode else 'FFT'}")

		obs.call("SetInputSettings", {"inputName": probe,
		                              "inputSettings": {"audio_source": OUTPUT_MIX_ID}})
		expect_animating(obs, probe, "capturing the output mix")

		obs.call("SetInputSettings", {"inputName": probe,
		                              "inputSettings": {"audio_source": self.media_name}})
		time.sleep(1.5)
		ref = get_audio_ref(obs, probe)
		if ref != self.media_uuid:
			raise SoakFailure(f"legacy name setting was not migrated to UUID (got {ref!r})")
		log("  ok: legacy name setting migrated to UUID")

		self.media_name = "Test Signal (renamed)"
		obs.call("SetInputName", {"inputUuid": self.media_uuid, "newInputName": self.media_name})
		expect_animating(obs, probe, "audio source renamed")

		switch_scene_collections(obs)
		expect_animating(obs, probe, "scene collection reloaded", wait=4)

		obs.call("RemoveInput", {"inputName": probe})
		self.spectrograms.pop(probe)

	# ── random churn ───────────────────────────────────────────────────

	def run_random_step(self, ops):
		obs, rng = self.obs, self.rng
		op = rng.choices(list(ops), weights=list(ops.values()))[0]

		needs_spectrogram = op in ("remove", "settings", "resize", "audio", "toggle", "props")
		if op == "create" or (needs_spectrogram and not self.spectrograms):
			if len(self.spectrograms) < 8:
				settings = self.randomize_settings()
				if "resize" in ops and rng.random() < 0.3:
					settings.update(self.randomize_size())
				settings["audio_source"] = self.pick_audio_ref()
				self.create_spectrogram(rng.choice(SCENES), settings)
		elif op == "remove":
			name = rng.choice(list(self.spectrograms))
			obs.call("RemoveInput", {"inputName": name})
			self.spectrograms.pop(name)
		elif op == "settings":
			obs.call("SetInputSettings", {"inputName": rng.choice(list(self.spectrograms)),
			                              "inputSettings": self.randomize_settings()})
		elif op == "resize":
			obs.call("SetInputSettings", {"inputName": rng.choice(list(self.spectrograms)),
			                              "inputSettings": self.randomize_size()})
		elif op == "audio":
			obs.call("SetInputSettings", {"inputName": rng.choice(list(self.spectrograms)),
			                              "inputSettings": {"audio_source": self.pick_audio_ref()}})
		elif op == "toggle":
			name = rng.choice(list(self.spectrograms))
			scene = self.spectrograms[name]
			item = obs.call("GetSceneItemId", {"sceneName": scene, "sourceName": name})
			obs.call("SetSceneItemEnabled", {"sceneName": scene,
			                                 "sceneItemId": item["sceneItemId"],
			                                 "sceneItemEnabled": rng.random() < 0.6})
		elif op == "scene":
			obs.call("SetCurrentProgramScene", {"sceneName": rng.choice(SCENES)})
		elif op == "studio":
			obs.call("SetStudioModeEnabled", {"studioModeEnabled": rng.random() < 0.5})
		elif op == "mute":
			obs.call("SetInputMute", {"inputUuid": self.media_uuid,
			                          "inputMuted": rng.random() < 0.3})
		elif op == "props":
			list_audio_choices(obs, rng.choice(list(self.spectrograms)))
		elif op == "recreate_media":
			# a new name each time: the frontend can hold on to a freshly
			# removed source for a while, keeping its name taken
			obs.call("RemoveInput", {"inputUuid": self.media_uuid})
			self.counter += 1
			self.media_name = f"Test Signal {self.counter}"
			self.create_media(SCENES[0])
		return op

	def remove_all_spectrograms(self, out_path):
		"""Remove every spectrogram and wait until OBS has destroyed them.
		OBS only processes a removal from a scene that isn't live when that
		scene is next shown, so each scene is shown once."""
		for name in list(self.spectrograms):
			self.obs.call("RemoveInput", {"inputName": name})
			self.spectrograms.pop(name)
		current = self.obs.call("GetCurrentProgramScene")["currentProgramSceneName"]
		for scene in SCENES:
			self.obs.call("SetCurrentProgramScene", {"sceneName": scene})
			time.sleep(0.2)
		self.obs.call("SetCurrentProgramScene", {"sceneName": current})

		deadline = time.time() + 10
		while True:
			with open(out_path, errors="replace") as f:
				pending = find_undestroyed_spectrograms(f.read())
			if not pending:
				return
			if time.time() > deadline:
				raise SoakFailure(f"removed spectrograms were not destroyed: {pending}")
			time.sleep(0.2)

	def reset_to_known_good(self):
		"""Put one spectrogram on the program scene, capturing the media."""
		obs = self.obs
		obs.call("SetStudioModeEnabled", {"studioModeEnabled": False})
		obs.call("SetCurrentProgramScene", {"sceneName": SCENES[0]})
		obs.call("SetInputMute", {"inputUuid": self.media_uuid, "inputMuted": False})
		name = self.create_spectrogram(SCENES[0], {"audio_source": self.media_uuid,
		                                           "width": 400, "height": 200})
		expect_animating(obs, name, "still works after churn")
		obs.call("RemoveInput", {"inputName": name})
		self.spectrograms.pop(name)


class Session:
	"""One OBS instance in a throwaway profile, with the build staged."""

	def __init__(self, args, repo_dir, name):
		self.args = args
		self.root = tempfile.mkdtemp(prefix=f"obs-spectrogram-{name}-")
		self.port = find_free_port()
		self.config_home = create_profile(self.root, self.port)
		disable_installed_plugin(self.config_home)
		self.plugin_path, bin_dir, data_dir = stage_plugin(
			self.root, os.path.abspath(args.build_dir), repo_dir)
		self.audio_file = create_test_audio(self.root)
		self.out_path = os.path.join(self.root, "obs-stdout.txt")
		self.xvfb, self.display = (None, os.environ.get("DISPLAY", "")) \
			if args.no_xvfb else start_xvfb()
		self.proc, self.out = launch_obs(args, self.display, self.config_home,
		                                 bin_dir, data_dir, self.out_path)

	def connect(self):
		obs = connect_when_ready(self.proc, self.port, timeout=60)
		version = check_plugin_loaded(self.config_home, self.plugin_path)
		return obs, version

	def shut_down(self):
		"""Returns a problem description, or None."""
		problem = None
		if self.proc.poll() is None:
			problem = quit_obs(self.proc, self.display)
		self.out.close()
		if self.xvfb:
			self.xvfb.terminate()
		return problem

	def read_leaks(self):
		m = re.search(r"Number of memory leaks: (\d+)", read_obs_log(self.config_home))
		return int(m.group(1)) if m else None


def switch_scene_collections(obs):
	obs.call("CreateSceneCollection", {"sceneCollectionName": "Soak Other"})
	time.sleep(2)
	obs.call("SetCurrentSceneCollection", {"sceneCollectionName": "Soak"})


def measure_baseline_leaks(args, repo_dir):
	"""OBS itself leaks a little when scene collections are switched (1
	allocation per switch in OBS 32.1).  Measure that with no spectrogram
	involved, so only what the plugin adds counts against it."""
	log("measuring OBS's own leak baseline")
	session = Session(args, repo_dir, "baseline")
	try:
		obs, _ = session.connect()
		obs.call("CreateSceneCollection", {"sceneCollectionName": "Soak"})
		switch_scene_collections(obs)
		obs.close()
	finally:
		problem = session.shut_down()
	leaks = session.read_leaks()
	if problem or leaks is None:
		raise SoakFailure(f"baseline run did not shut down cleanly: {problem}")
	shutil.rmtree(session.root, ignore_errors=True)
	return leaks


def summarize_rss(checkpoints):
	"""RSS at checkpoints taken with no spectrograms alive.  Churning
	textures of up to 4096x4096 raises the allocator's and the GL driver's
	high-water mark before it settles, so only growth over the second half
	of the run counts as a leak."""
	if len(checkpoints) < 4:
		return None
	return checkpoints[len(checkpoints) // 2], checkpoints[-1]


def check_plugin_loaded(config_home, plugin_path):
	obs_log = read_obs_log(config_home)
	m = re.search(r"\[obs-spectrogram\] version (\S+) loaded from (\S+)", obs_log)
	if not m:
		raise SoakFailure("plugin did not load (no version line in the OBS log)")
	if os.path.realpath(m.group(2)) != os.path.realpath(plugin_path) or \
			f"Source '{KIND}' already exists" in obs_log:
		raise SoakFailure("an installed copy of the plugin shadows the build under "
		                  "test; remove it, or use OBS >= 32 which lets the soak disable it")
	return m.group(1)


def run_soak(args, session, stats):
	obs, version = session.connect()
	proc = session.proc
	log(f"OBS pid {proc.pid}, plugin {version} from the build under test")

	obs.call("CreateSceneCollection", {"sceneCollectionName": "Soak"})
	soak = Soak(obs, session.audio_file, random.Random(args.seed))
	soak.run_functional_checks()

	log(f"random churn for {args.minutes:g} min (seed {args.seed})")
	end = time.time() + args.minutes * 60
	next_checkpoint = time.time()
	steps = 0
	while True:
		now = time.time()
		if now >= next_checkpoint or now >= end:
			soak.remove_all_spectrograms(session.out_path)
			time.sleep(1)
			stats["rss"].append(read_rss_mb(proc.pid))
			log(f"  checkpoint: {steps} steps, RSS {stats['rss'][-1]:.0f} MB")
			next_checkpoint = now + args.checkpoint_interval
		if now >= end:
			break
		if proc.poll() is not None:
			raise SoakFailure(f"OBS exited mid-soak (code {proc.returncode})")
		op = soak.run_random_step(args.ops)
		stats["ops"][op] = stats["ops"].get(op, 0) + 1
		steps += 1
		time.sleep(args.step_delay)

	soak.reset_to_known_good()
	obs.close()


def run(args):
	repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	try:
		baseline_leaks = measure_baseline_leaks(args, repo_dir)
	except SoakFailure as e:
		print(f"FAIL: {e}")
		return 1

	session = Session(args, repo_dir, "soak")
	log(f"working directory: {session.root}")
	failure = None
	stats = {"rss": [], "ops": {}}
	try:
		run_soak(args, session, stats)
	except SoakFailure as e:
		failure = str(e)
	except Exception as e:  # websocket drop mid-call usually means a crash
		failure = f"{type(e).__name__}: {e}"
	finally:
		log("shutting down OBS")
		failure = failure or session.shut_down()

	root, config_home, out_path, proc = \
		session.root, session.config_home, session.out_path, session.proc
	obs_log = read_obs_log(config_home)
	leaks = session.read_leaks()
	asan_logs = [f for f in os.listdir(root) if f.startswith("asan")]
	with open(out_path, errors="replace") as f:
		undestroyed = find_undestroyed_spectrograms(f.read())

	print()
	print("── results " + "─" * 60)
	print(f"operations:   {dict(sorted(stats['ops'].items()))}")
	rss_summary = summarize_rss(stats["rss"])
	if rss_summary:
		first, last = rss_summary
		print(f"RSS:          {first:.0f} MB -> {last:.0f} MB over the second half, at idle "
		      f"(all: {', '.join(f'{r:.0f}' for r in stats['rss'])})")
		if "resize" in args.ops:
			print("              (not judged: resizing between 64x64 and 4096x4096 moves the "
			      "allocator's high-water mark; use --ops without resize for a memory verdict)")
		elif last - first > args.rss_growth_mb:
			failure = failure or (f"idle RSS still growing: +{last - first:.0f} MB over the second "
			                      f"half (limit {args.rss_growth_mb:g} MB)")
	if leaks is not None:
		print(f"leaks:        {leaks} (libobs allocation counter at exit; "
		      f"OBS baseline {baseline_leaks})")
		if leaks > baseline_leaks:
			failure = failure or f"{leaks - baseline_leaks} allocations leaked beyond OBS's baseline"
	else:
		failure = failure or "no leak report in the OBS log: OBS did not shut down cleanly"
	print(f"spectrograms: {len(undestroyed)} never destroyed")
	if undestroyed:
		failure = failure or f"spectrograms never destroyed (leaked reference?): {undestroyed}"
	if proc.returncode != 0:
		failure = failure or f"OBS exited with code {proc.returncode}"
	if asan_logs:
		failure = failure or f"AddressSanitizer reported errors: {asan_logs}"
	print(f"OBS log:      {os.path.join(config_home, 'obs-studio', 'logs')}")
	print(f"OBS stdout:   {out_path}")

	if failure:
		print(f"\nFAIL: {failure}")
		print("\nlast lines of the OBS log:")
		print("\n".join(obs_log.splitlines()[-25:]))
		return 1
	print("\nPASS")
	if not args.keep:
		shutil.rmtree(root, ignore_errors=True)
	return 0


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--build-dir", default="build", help="directory containing obs-spectrogram.so")
	p.add_argument("--obs", default="obs", help="OBS executable")
	p.add_argument("--minutes", type=float, default=10, help="length of the random churn phase")
	p.add_argument("--seed", type=int, default=int(time.time()), help="RNG seed, to replay a run")
	p.add_argument("--step-delay", type=float, default=0.25, help="seconds between random operations")
	p.add_argument("--checkpoint-interval", type=float, default=60,
	               help="seconds between idle RSS checkpoints")
	p.add_argument("--rss-growth-mb", type=float, default=64,
	               help="fail if idle RSS grows more than this over the second half")
	p.add_argument("--ops", default=",".join(OPS),
	               help="comma-separated random operations to use (default: all of them)")
	p.add_argument("--asan", action="store_true", help="preload AddressSanitizer (build with -fsanitize=address)")
	p.add_argument("--no-xvfb", action="store_true", help="show OBS on the current display")
	p.add_argument("--keep", action="store_true", help="keep the working directory after a pass")
	args = p.parse_args()
	unknown = set(args.ops.split(",")) - set(OPS)
	if unknown:
		p.error(f"unknown --ops {sorted(unknown)}; choose from {', '.join(OPS)}")
	args.ops = {op: OPS[op] for op in args.ops.split(",")}
	sys.exit(run(args))


if __name__ == "__main__":
	main()
