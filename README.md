# obs-spectrogram

A rolling spectrogram source plugin for [OBS Studio](https://obsproject.com/). Captures audio from any OBS audio source and displays a live, scrolling spectrogram using GPU-accelerated rendering.

## Features

- **Two analysis modes** selectable per source:
  - **FFT** — short-time Fourier transform with configurable size (256–8192), overlap, and window function (Rectangular, Hann, Hamming, Blackman)
  - **Resonate** — sample-by-sample resonator bank based on *"Resonate: Efficient Low Latency Spectral Analysis of Audio Signals"* by Alexandre R. J. François (ICMC 2025 Best Paper Award); no buffering required, minimal latency
- **Efficient rolling display** — one texture column is written per analysis frame; two quads with `AddressU = Wrap` render the circular buffer without any CPU-side pixel shuffling
- **Logarithmic or linear frequency axis** with configurable min/max frequency
- **Configurable dB floor/ceiling and gain**
- **3-stop color palette** with full alpha support (Low / Mid / High power)
- **Any OBS audio source** selectable as input

## Settings

| Setting | Description |
|---|---|
| Audio Source | OBS source to capture audio from |
| Width / Height | Texture and display dimensions (height up to 8192) |
| Analysis Mode | FFT or Resonate |
| **FFT mode** | |
| FFT Size | DFT frame length (256 – 8192 samples) |
| Overlap | Fraction of frame reused between columns (0 – 0.95) |
| Window Function | Rectangular, Hann, Hamming, Blackman |
| **Resonate mode** | |
| Resonators | Number of geometrically-spaced resonator bands (2 – 1024) |
| Column Hop | Samples consumed per display column (64 – 2048) |
| Double Smoothing | Apply a second EWMA pass (Eq. 8 of the paper) for a smoother readout |
| **Common** | |
| Min / Max Frequency | Frequency range mapped to the display (Hz) |
| Floor / Ceiling dB | dB range mapped to the color palette |
| Gain (dB) | Additive gain applied before color mapping |
| Logarithmic Frequency Scale | Log-spaced frequency axis (perceptually uniform) |
| Color: Low / Mid / High Power | Three color stops interpolated across the dB range |

## Building

Requires **OBS Studio** development headers and **CMake ≥ 3.16**.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Installing (Linux)

```bash
sudo cmake --install build
```

This installs the plugin to `/usr/lib/obs-plugins/` and the shader to `/usr/share/obs/obs-plugins/obs-spectrogram/`.

### Installing manually (Linux)

```bash
mkdir -p ~/.config/obs-studio/plugins/obs-spectrogram/bin/64bit
mkdir -p ~/.config/obs-studio/plugins/obs-spectrogram/data/shaders

cp build/obs-spectrogram.so \
   ~/.config/obs-studio/plugins/obs-spectrogram/bin/64bit/

cp data/shaders/spectrogram.effect \
   ~/.config/obs-studio/plugins/obs-spectrogram/data/shaders/
```

## Usage

1. In OBS, add a new source → **Spectrogram**
2. In the source properties, select the **Audio Source** to visualize
3. Choose **FFT** or **Resonate** analysis mode and tune the parameters
4. Resize and position the source like any other OBS source

## How the rendering works

The texture is a circular buffer `width × height` pixels. Each analysis frame writes one column at the current write head and advances it by one. The display is drawn as a single full-screen quad with UV range `[write_col/W, write_col/W + 1.0]` and `AddressU = Wrap`, so the GPU sampler handles the wraparound with no CPU work.

## Resonate mode

The Resonate algorithm maintains a bank of *N* independent complex resonators, each updated once per input sample:

```
P ← P · e^{−iω_k·Δt}          (rotate phasor)
R ← (1 − α_k)·R + α_k · x · P  (EWMA of demodulated signal)
```

where `α_k = 1 − e^{−Δt · f_k / log(1 + f_k)}` gives each resonator a time constant proportional to its period. Resonators are geometrically spaced between the configured frequency bounds. The magnitude `|R|` is sampled every *hop* input samples to produce one display column.

Unlike FFT, Resonate requires no buffering and can produce a new column after as few as 1 sample, making it suitable for extremely low-latency visualization.

## License

MIT — see [LICENSE](LICENSE).

Resonate algorithm © 2025 Alexandre R. J. François, used for reference implementation purposes under the terms of the [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/) license under which the paper is published.
