# stt-wayland

Local, privacy-preserving **Wayland speech-to-text daemon** for Linux desktops (Hyprland, Sway, etc.).

| Component | Role |
|-----------|------|
| **sttd** | Session daemon: PipeWire capture → webrtc-vad → whisper.cpp → overlay + tray + virtual-keyboard inject |
| **sttctl** | CLI control client (`start` / `stop` / `toggle` / …) |
| **stt-settings** | GTK4 profile editor for models and daemon configuration |
| **proto** | Shared newline-delimited JSON IPC types |

**No cloud STT.** Models are on-device (`base.en` / `small.en` via [whisper-rs](https://crates.io/crates/whisper-rs) / whisper.cpp). Native Wayland only (no X11 injection helpers, no Wine).

## Architecture (v1)

```
  mic (PipeWire) ──► webrtc-vad ──► mpsc<Vec<f32>> ──► whisper-rs
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
  pipewire pipewire-pulse libpipewire \
  gtk4 gtk4-layer-shell wayland wayland-protocols
```

### Debian / Ubuntu

```bash
sudo apt install build-essential pkg-config cmake clang \
  libpipewire-0.3-dev libspa-0.2-dev \
  libgtk-4-dev libgtk4-layer-shell-dev \
  libwayland-dev wayland-protocols
```

### Fedora

```bash
sudo dnf install gcc gcc-c++ cmake clang pkgconf-pkg-config \
  pipewire-devel gtk4-devel gtk4-layer-shell-devel \
  wayland-devel wayland-protocols-devel
```

### Rust crates of note

| Crate | Purpose | Pinned line |
|-------|---------|-------------|
| `pipewire` | Audio capture | **0.9** (0.8 breaks on PipeWire ≥1.4) |
| `webrtc-vad` | Voice activity detection | 0.4 |
| `whisper-rs` | whisper.cpp bindings (needs **cmake** + **clang/libclang**) | **0.16** |
| `gtk4`, `gtk4-layer-shell` | Overlay + settings UI | 0.9 / 0.4 |
| `wayland-client`, `wayland-protocols-misc` | virtual-keyboard inject | 0.31 / 0.3 |

`whisper-rs` compiles whisper.cpp from source on first build; expect several minutes and a C/C++ toolchain.

### Feature flags (`sttd`)

| Feature | Default | Notes |
|---------|---------|-------|
| `audio` | yes | PipeWire |
| `stt-engine` | yes | whisper-rs + webrtc-vad |
| `overlay` | yes | gtk4 + layer-shell |
| `inject` | yes | wayland virtual-keyboard |
| `tray` | yes | StatusNotifierItem over D-Bus |
| `settings` | yes | GTK4 settings application |

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

Place GGML Whisper models under:

```
~/.local/share/sttd/models/
```

Example (base.en):

```bash
mkdir -p ~/.local/share/sttd/models
# Official ggml models, e.g. from Hugging Face ggml-org/whisper.cpp
curl -L -o ~/.local/share/sttd/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

v1 targets **base.en** or **small.en**. Configure name/residency in `config.toml`.

## Configuration

Path: **`~/.config/sttd/config.toml`**

See [`contrib/config.example.toml`](contrib/config.example.toml). Sections:

- `[audio]` — sample rate, silence gap, channel capacity, `max_edit_seconds` (~5 min)
- `[model]` — path/name, directory, `residency = "preload" | "lazy"`
- `[overlay]` — preview crossfade (~200 ms), edit buffer cap
- `[injection]` — enable virtual-keyboard inject
- `[tray]` — enable the StatusNotifierItem icon (defaults to on)

### Profiles and settings GUI

Launch `stt-settings` to manage named profiles under
`~/.config/sttd/profiles/`. It discovers local `.bin` Whisper models and edits
the model, audio/VAD, overlay, injection, and tray settings. **Save** updates
only the selected profile; **Apply profile** also mirrors it to
`~/.config/sttd/config.toml` for the existing systemd unit. Restart `sttd` after
applying a profile. Use **Refresh models** after adding a model while the editor
is open.

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
commands that may load Whisper or flush a final transcript):

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

Settings GUI: `stt-settings` is a separate GTK4 application and does not need
the daemon to be running. Profile changes are persisted atomically; switching
profiles affects the next daemon start/restart.

Tray behaviour: when enabled, `sttd` registers a StatusNotifierItem on the
session D-Bus. The icon shows idle, recording, paused, or error state; a
left-click toggles capture, and the menu contains the current state/model,
Start/Stop, and Quit. A missing session bus or tray host is a silent no-op, so
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
      stt.rs          # whisper-rs worker thread
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
