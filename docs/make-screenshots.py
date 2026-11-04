#!/usr/bin/env python3
"""Regenerate the images in docs/images/ from a headless OBS.

Uses the same throwaway-profile setup as the soak test, plays a
synthesized music loop, and captures:

  hero.png        the showcase scene (FFT above, Resonate below)
  scrolling.gif   a few seconds of the same, animated
  properties.png  the source's properties dialog

    docs/make-screenshots.py --build-dir build

Needs what the soak test needs, plus ImageMagick's `import` and numpy.
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "soak"))
import soak  # noqa: E402

RATE = 48000
CANVAS = (1280, 720)
BPM = 112
PALETTE_DEFAULT = (0xFF000000, 0xFF00FFFF, 0xFFFFFFFF)  # black, yellow, white
PALETTE_INFERNO = (0xFF1A0500, 0xFF7936B6, 0xFFBFFDFC)  # OBS colors are ABGR


def midi_to_hz(note):
	return 440.0 * 2 ** ((note - 69) / 12)


def synth_pluck(freq, dur):
	t = np.arange(int(dur * RATE)) / RATE
	out = np.zeros_like(t)
	for k in range(1, 12):
		if freq * k > RATE / 2:
			break
		out += np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (3 + 1.5 * k)) / k
	return out * np.minimum(1, t * 400)


def synth_kick(dur=0.35):
	t = np.arange(int(dur * RATE)) / RATE
	f = 45 + 110 * np.exp(-t * 30)
	return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 9)


def synth_hat(dur=0.06, rng=np.random.default_rng(1)):
	n = rng.standard_normal(int(dur * RATE))
	n = np.diff(n, prepend=0)  # crude high-pass
	return n * np.exp(-np.arange(len(n)) / RATE * 70) * 0.25


def mix_in(buf, sig, at, gain=1.0):
	i = int(at * RATE)
	j = min(len(buf), i + len(sig))
	buf[i:j] += sig[:j - i] * gain


def create_music(path):
	"""Eight bars: arpeggiated i-VI-III-VII in A minor over kick and hats,
	with a rising sweep over the last two bars."""
	beat = 60 / BPM
	bars = 8
	buf = np.zeros(int(bars * 4 * beat * RATE) + RATE)
	chords = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]]
	for bar in range(bars):
		chord = chords[bar % 4]
		t0 = bar * 4 * beat
		mix_in(buf, synth_pluck(midi_to_hz(chord[0] - 12), 4 * beat), t0, 0.5)
		for step in range(8):
			note = chord[[0, 1, 2, 3, 2, 1, 3, 2][step]] + (12 if step >= 6 else 0)
			mix_in(buf, synth_pluck(midi_to_hz(note), beat * 1.5), t0 + step * beat / 2, 0.35)
		for b in range(4):
			if b % 2 == 0:
				mix_in(buf, synth_kick(), t0 + b * beat, 0.8)
			mix_in(buf, synth_hat(), t0 + b * beat + beat / 2)
	sweep_start = 6 * 4 * beat
	t = np.arange(int(2 * 4 * beat * RATE)) / RATE
	f = 200 * (6000 / 200) ** (t / t[-1])
	mix_in(buf, np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.minimum(1, t * 2) * 0.12, sweep_start)

	buf /= np.max(np.abs(buf)) * 1.1
	pcm = (np.repeat(buf[:, None], 2, axis=1) * 32767).astype(np.int16)
	with wave.open(path, "wb") as w:
		w.setnchannels(2)
		w.setsampwidth(2)
		w.setframerate(RATE)
		w.writeframes(pcm.tobytes())


def add_text(obs, scene, name, text, x, y, size=28):
	r = obs.call("CreateInput", {
		"sceneName": scene, "inputName": name, "inputKind": "text_ft2_source_v2",
		"inputSettings": {"text": text, "font": {"face": "Sans", "size": size, "style": "Bold"},
		                  "color1": 0xFFFFFFFF, "color2": 0xFFFFFFFF},
		"sceneItemEnabled": True})
	obs.call("SetSceneItemTransform", {"sceneName": scene, "sceneItemId": r["sceneItemId"],
	                                   "sceneItemTransform": {"positionX": x, "positionY": y}})


def add_spectrogram(obs, scene, name, settings, x, y):
	r = obs.call("CreateInput", {"sceneName": scene, "inputName": name,
	                             "inputKind": soak.KIND, "inputSettings": settings,
	                             "sceneItemEnabled": True})
	obs.call("SetSceneItemTransform", {"sceneName": scene, "sceneItemId": r["sceneItemId"],
	                                   "sceneItemTransform": {"positionX": x, "positionY": y}})


def build_scene(obs, music_file):
	scene = "Showcase"
	obs.call("CreateScene", {"sceneName": scene})
	obs.call("SetCurrentProgramScene", {"sceneName": scene})
	media_uuid = obs.call("CreateInput", {"sceneName": scene, "inputName": "Music",
	                                      "inputKind": "ffmpeg_source",
	                                      "inputSettings": {"local_file": music_file,
	                                                        "looping": True},
	                                      "sceneItemEnabled": True})["inputUuid"]
	obs.call("CreateInput", {"sceneName": scene, "inputName": "Background",
	                         "inputKind": "color_source_v3",
	                         "inputSettings": {"color": 0xFF201812, "width": CANVAS[0],
	                                           "height": CANVAS[1]},
	                         "sceneItemEnabled": True})
	w, h = 1200, 300
	common = {"audio_source": media_uuid, "width": w, "height": h,
	          "min_freq": 30.0, "max_freq": 16000.0, "db_min": -80.0, "log_scale": True}
	# both hop 512 samples per column, so the two panels scroll in step
	add_spectrogram(obs, scene, "FFT", dict(common, analysis_mode=0, fft_size=2048, overlap=0.75,
	                                        gain_db=-10.0,
	                                        color_0=PALETTE_DEFAULT[0], color_1=PALETTE_DEFAULT[1],
	                                        color_2=PALETTE_DEFAULT[2]), 40, 40)
	add_spectrogram(obs, scene, "Resonate", dict(common, analysis_mode=1, resonate_bins=300,
	                                             resonate_hop=512, resonate_smooth=True,
	                                             color_0=PALETTE_INFERNO[0], color_1=PALETTE_INFERNO[1],
	                                             color_2=PALETTE_INFERNO[2]), 40, 380)
	add_text(obs, scene, "FFT label", "FFT", 56, 48)
	add_text(obs, scene, "Resonate label", "Resonate", 56, 388)
	return scene


def grab_program(obs, scene):
	shot = obs.call("GetSourceScreenshot", {"sourceName": scene, "imageFormat": "png"})
	return Image.open(io.BytesIO(base64.b64decode(shot["imageData"].split(",", 1)[1]))).convert("RGB")


def grab_window(display, title_regex, out_path):
	env = dict(os.environ, DISPLAY=display)
	win = subprocess.run(["xdotool", "search", "--name", title_regex], env=env,
	                     capture_output=True, text=True).stdout.split()
	if not win:
		raise soak.SoakFailure(f"no window matching {title_regex!r}")
	subprocess.run(["import", "-window", win[-1], out_path], env=env, check=True)
	return win[-1]


def open_audio_source_list(display, window):
	"""Click the Audio Source combo box: first row of the form, right of
	the preview.  Moves the dialog to the top-left corner first so the
	popup can be cropped from the root window."""
	env = dict(os.environ, DISPLAY=display)
	subprocess.run(["xdotool", "windowmove", "--sync", window, "0", "0"], env=env)
	time.sleep(0.5)
	subprocess.run(["xdotool", "mousemove", "--window", window, "425", "297", "click", "1"],
	               env=env, check=True)


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--build-dir", default="build")
	p.add_argument("--obs", default="obs")
	p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
	args = p.parse_args()
	args.asan = False
	args.no_xvfb = False
	os.makedirs(args.out_dir, exist_ok=True)

	repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	session = soak.Session(args, repo_dir, "screenshots")
	create_music(session.audio_file)  # replace the soak test's test tones
	try:
		obs, _ = session.connect()
		obs.call("SetVideoSettings", {"baseWidth": CANVAS[0], "baseHeight": CANVAS[1],
		                              "outputWidth": CANVAS[0], "outputHeight": CANVAS[1]})
		obs.call("CreateSceneCollection", {"sceneCollectionName": "Screenshots"})
		scene = build_scene(obs, session.audio_file)

		time.sleep(14)  # fill the history
		grab_program(obs, scene).save(os.path.join(args.out_dir, "hero.png"))

		frames, stamps = [], []
		while len(frames) < 50:
			stamps.append(time.time())
			frames.append(grab_program(obs, scene).resize((480, 270), Image.LANCZOS)
			              .quantize(64, dither=Image.Dither.NONE))
			time.sleep(0.05)
		# play back at the speed it was captured
		durations = [int((b - a) * 1000) for a, b in zip(stamps, stamps[1:])]
		durations.append(durations[-1])
		frames[0].save(os.path.join(args.out_dir, "scrolling.gif"), save_all=True,
		               append_images=frames[1:], duration=durations, loop=0, optimize=True)

		obs.call("OpenInputPropertiesDialog", {"inputName": "Resonate"})
		time.sleep(2)
		props = grab_window(session.display, "Properties for 'Resonate'",
		                    os.path.join(args.out_dir, "properties.png"))
		open_audio_source_list(session.display, props)
		time.sleep(1)
		subprocess.run(["import", "-window", "root", "-crop", "720x580+0+0",
		                os.path.join(args.out_dir, "audio-sources.png")],
		               env=dict(os.environ, DISPLAY=session.display), check=True)
		obs.close()
	finally:
		session.shut_down()
	print(f"wrote images to {args.out_dir}")


if __name__ == "__main__":
	main()
