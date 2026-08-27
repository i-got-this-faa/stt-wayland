# stt-wayland

Local, privacy-preserving **Wayland speech-to-text daemon** for Linux desktops (Hyprland, Sway, etc.).

| Component | Role |
|-----------|------|
| **sttd** | Session daemon: PipeWire capture → webrtc-vad → audio.cpp → overlay + tray + virtual-keyboard inject |
| **sttctl** | CLI control client (`start` / `stop` / `toggle` / …) |
| **stt-settings** | GTK4/libadwaita profile editor for models and daemon configuration |
| **proto** | Shared newline-delimited JSON IPC types |

**No cloud STT.** Models run on-device through [audio.cpp](https://github.com/0xShug0/audio.cpp) — modern GGUF ASR families (Qwen3 ASR, Parakeet TDT, Nemotron ASR, Citrinet). Native Wayland only (no X11 injection helpers, no Wine).

## Architecture (v1)

```
  mic (PipeWire) ──► webrtc-vad ──► mpsc<Vec<f32>> ──► audio.cpp
                         │                                  │
                         │                          committed text
                         ▼                                  │
                   gtk4 overlay ◄── preview (~200ms)        │
                                                            ▼
                                              virtual-keyboard-unstable-v1
```

- **Audio**: `pipewire-rs` direct stream, mono **16 kHz f32**. Open on Start, close on Stop. Pause stops feeding VAD without tearing the stream.
- **VAD**: `webrtc-vad`; silence gap ~**500 ms** closes a chunk. Bounded channel with backpressure to the STT thread.
- **IPC**: Unix socket `$XDG_RUNTIME_DIR/sttd.sock`, NDJSON via the `proto` crate.
- **Modes**: `Live` (commit+inject per VAD chunk) | `Edit` (buffer → EditReview pane).
- **Overlay**: `gtk4` + `gtk4-layer-shell`, `Layer::Overlay`. Keyboard interactivity `None` while recording; `OnDemand` only in EditReview.
- **Inject**: `virtual-keyboard-unstable-v1` via `wayland-client` (not wtype/ydotool). Missing protocol → clear status error.
- **Tray**: StatusNotifierItem over the session D-Bus. The bundled icon reflects Idle, Recording, Paused, and Error; click toggles capture and the menu also exposes Quit.

### State machine

```
Idle --Start(mode)--> Recording(Live|Edit)
Recording --Pause--> Paused --Resume--> Recording
Recording(Live) --Stop--> Idle          (flush, destroy overlay, close audio)
Recording(Edit) --Stop--> EditReview    (transcribe full buffer)
EditReview --Copy--> inject then Idle
EditReview --Discard--> Idle
```

`Start` while already `Recording` is a no-op `Ok`. Toggle: Idle/Paused/EditReview → Start; Recording → Stop.

## Build dependencies (system)

### Arch Linux

```bash
sudo pacman -S --needed base-devel rust cmake clang pkgconf \
  cblas \
  pipewire pipewire-pulse libpipewire \
  gtk4 gtk4-layer-shell libadwaita wayland wayland-protocols
```

### Debian / Ubuntu

```bash
sudo apt install build-essential pkg-config cmake clang \
  libopenblas-dev \
  libpipewire-0.3-dev libspa-0.2-dev \
  libgtk-4-dev libgtk4-layer-shell-dev libadwaita-1-dev \
  libwayland-dev wayland-protocols
```

### Fedora

```bash
sudo dnf install gcc gcc-c++ cmake clang pkgconf-pkg-config \
  openblas-devel \
  pipewire-devel gtk4-devel gtk4-layer-shell-devel libadwaita-devel \
  wayland-devel wayland-protocols-devel
```

### Rust crates of note

| Crate | Purpose | Pinned line |
|-------|---------|-------------|
| `pipewire` | Audio capture | **0.9** (0.8 breaks on PipeWire ≥1.4) |
| `webrtc-vad` | Voice activity detection | 0.4 |
| `audio-cpp` | audio.cpp bindings (needs **CMake** + C++ toolchain) | **0.2** |
| `gtk4`, `gtk4-layer-shell`, `libadwaita` | Overlay + settings UI | 0.9 / 0.4 / 0.9 |
| `wayland-client`, `wayland-protocols-misc` | virtual-keyboard inject | 0.31 / 0.3 |

`audio-cpp` compiles audio.cpp (ggml) from source on first build. The default build is CPU-only and self-contained.

### Feature flags (`sttd`)

| Feature | Default | Notes |
|---------|---------|-------|
| `audio` | yes | PipeWire |
| `stt-engine` | yes | audio-cpp + webrtc-vad |
| `overlay` | yes | gtk4 + layer-shell |
| `inject` | yes | wayland virtual-keyboard |
| `tray` | yes | StatusNotifierItem over D-Bus |
| `settings` | yes | GTK4 settings application |
| `post-process` | yes | Local text normalization (candle, quantized Qwen3) |

Check without system GUI/audio libs:

```bash
cargo check -p proto -p sttctl
cargo check -p sttd --no-default-features
```

Full daemon (default):

```bash
cargo build --release
```

Binaries: `target/release/sttd`, `target/release/sttctl`, `target/release/stt-settings`.

## Model setup

Launch the model manager:

```bash
stt-settings
```

The **Speech Model** dropdown can download, select, cancel, and delete verified
models. The catalog includes Nemotron 3.5 ASR Streaming in Q5 and Q8 variants,
Parakeet TDT v3, Whisper Large v3 Turbo, Qwen3-ASR, Moonshine Streaming, and
Whisper Base English. Downloads use pinned model revisions, verify the exact
size and SHA-256 hash, and are installed atomically under:

```
~/.local/share/sttd/models/
```

Existing custom GGUF and legacy whisper.cpp `.bin` paths remain readable from
profiles, but catalog download and deletion are intentionally confined to the
managed models directory.

## Post-processing (text normalization)

Final transcripts can pass through a local **text-normalization model** after
recognition — punctuation and capitalization, spoken numbers/dates/emails
written out, fillers ("um", "uh") removed. The first catalog model is
**[S1-mini by Superwhisper](https://huggingface.co/superwhisper/s1-mini)**, a
0.6B Qwen3 fine-tune (Q4_K_M GGUF, ~462 MiB) that runs on the CPU. Inference
uses [candle](https://github.com/huggingface/candle) in-process — no extra
services — on the STT worker thread, so committed text ordering and
backpressure are unchanged.

Enable it in `stt-settings` → **Post-Processing**: download the model, flip the
toggle, and pick the control-line values (Styling / Structure / Context) that
steer the model. Or edit the profile:

```toml
[post_process]
enabled = true
model = "s1-mini"          # catalog id
styling = "semi-formal"    # casual | semi-casual | semi-formal | formal
structure = "prose"        # prose | lists
context = "general"        # general | email
```

Notes:

- The model loads lazily on the first final transcript; restart `sttd` after
  changing these settings.
- Previews (the live overlay) stay raw — only committed text is normalized.
- Failures never break dictation: if the model is missing or errors, the raw
  transcript is committed and the problem is logged.
- Empty output is valid (filler-only input normalizes to nothing).
- Model files live under `~/.local/share/sttd/models/post-process/<id>/`.

## Configuration

Path: **`~/.config/sttd/config.toml`**

See [`contrib/config.example.toml`](contrib/config.example.toml). Sections:

- `[audio]` — sample rate, silence gap, channel capacity, `max_edit_seconds` (~5 min)
- `[model]` — path/name, directory, `residency = "preload" | "lazy"`
- `[post_process]` — text-normalization phase (`enabled`, catalog `model`, control line)
- `[overlay]` — preview crossfade (~200 ms), edit buffer cap
- `[injection]` — enable virtual-keyboard inject
- `[tray]` — enable the StatusNotifierItem icon (defaults to on)

### Profiles and settings GUI

Launch `stt-settings` to manage named profiles under
`~/.config/sttd/profiles/`. Its model dropdown handles verified downloads,
selection, cancellation, and safe deletion, while the rest of the app edits
audio/VAD, overlay, injection, and tray settings. **Save** updates only the
selected profile; **Apply** also mirrors it to
`~/.config/sttd/config.toml` for the existing systemd unit. Restart `sttd` after
applying a profile. A model cannot be deleted while a saved profile references
it.

The daemon can also use a profile directly without changing the active config:

```bash
sttd --profile Work
```

Optional first-run:

```bash
mkdir -p ~/.config/sttd
cp contrib/config.example.toml ~/.config/sttd/config.toml
```

## Running

Quick start (from the repo after `cargo build --release`):

```bash
# terminal 1 — daemon (foreground)
RUST_LOG=info ./target/release/sttd

# or background
./target/release/sttd &
sttctl toggle
```

Control client (`sttctl` uses a **500 ms** status timeout and up to **120 s** for
commands that may load a model or flush a final transcript):

```bash
sttctl start --mode live   # or: sttctl start --mode edit
sttctl pause
sttctl resume
sttctl stop
sttctl toggle --mode live
sttctl mode edit
sttctl status
```

Typical session:

```bash
sttd &
sttctl toggle          # start Live capture
# speak…
sttctl toggle          # stop → inject committed text (Live)
sttctl status
```

Socket: `$XDG_RUNTIME_DIR/sttd.sock`.

Idle behaviour: no PipeWire stream and no mapped overlay surface (GTK runtime stays warm on its thread).

Settings GUI: `stt-settings` is a separate GTK4/libadwaita application and does
not need the daemon to be running. Profile changes are persisted atomically;
switching profiles affects the next daemon start/restart.

Tray behaviour: when enabled, `sttd` registers a StatusNotifierItem on the
session D-Bus. The icon shows idle, recording, paused, or error state; a
left-click toggles capture, and the menu contains the current state/model,
Start/Stop, Settings, and Quit. A missing session bus or tray host is a silent no-op, so
the daemon remains usable on headless sessions.

### systemd user unit

```bash
mkdir -p ~/.config/systemd/user
cp contrib/sttd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sttd.service
```

Template: [`contrib/sttd.service`](contrib/sttd.service).

### Hyprland keybind

```conf
# ~/.config/hypr/hyprland.conf
bind = SUPER, Space, exec, sttctl toggle --mode live
bind = SUPER SHIFT, Space, exec, sttctl toggle --mode edit
bind = SUPER, Escape, exec, sttctl stop
```

### Sway keybind

```conf
# ~/.config/sway/config
bindsym $mod+space exec sttctl toggle --mode live
bindsym $mod+Shift+space exec sttctl toggle --mode edit
bindsym $mod+Escape exec sttctl stop
```

Ensure `sttctl` is on `PATH` (e.g. `~/.local/bin` or `export PATH="$PWD/target/release:$PATH"`).

## IPC protocol (summary)

Newline-delimited JSON. Requests (`type` tag):

| type | fields |
|------|--------|
| `start` | `mode`: `live` \| `edit` |
| `stop` | |
| `toggle` | `mode` |
| `set_mode` | `mode` |
| `pause` | |
| `resume` | |
| `status` | |

Responses: `ok`, `error { message }`, `status { state, mode, error?, model?, inject_available }`.

**Edge behaviour**

- Missing virtual-keyboard protocol → status `error` string; inject disabled; capture still works.
- `Start` while already `Recording` → no-op `Ok`.
- Edit buffer capped (~5 minutes / `audio.max_edit_seconds`).
- Model missing on preload → warning + lazy retry on first Start.

## Project layout

```
stt-wayland/
  Cargo.toml          # workspace
  proto/              # IPC types
  sttctl/             # CLI
  sttd/               # daemon
    src/
      main.rs         # socket server + core event loop + pipeline wiring
      settings.rs     # GTK4 profile/model settings application
      audio.rs        # PipeWire stream
      vad.rs          # webrtc-vad chunker
      stt.rs          # audio.cpp worker thread
      postprocess.rs  # text-normalization phase (S1-mini, candle)
      postprocess/    # post-processing tests
      model_catalog.rs # pinned models + verified download/delete helpers
      state.rs        # state machine
      inject.rs       # virtual-keyboard
      config.rs
      overlay/        # gtk4 live card + edit pane
  contrib/
    config.example.toml
    sttd.service
```

## Status

Integrate phase complete for v1 wiring:

- Unix socket NDJSON server (`$XDG_RUNTIME_DIR/sttd.sock`)
- State machine (Idle / Recording / Paused / EditReview × Live|Edit)
- Audio → VAD → STT pipeline with backpressure
- Live commit+inject; Edit full-buffer → EditReview (Copy/Discard)
- GTK overlay on a dedicated thread (layer-shell Overlay)

## License

MIT
