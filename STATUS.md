# stt-wayland — STATUS

Last verified: 2026-07-24 (workspace on Linux, Wayland session present).

## What works (build / unit / IPC)

| Item | Status |
|------|--------|
| `cargo check --workspace` | **OK** (default features) |
| `cargo build --workspace` | **OK** → `target/debug/sttd`, `target/debug/sttctl` |
| `cargo check -p sttd --no-default-features` | **OK** (stub backends; no system GUI/audio required) |
| `cargo test -p proto` | **OK** (25 tests + 1 doctest) |
| `cargo test --workspace` | **OK** (proto + sttd unit tests: state, config, vad, stt helpers, inject probe) |
| `sttctl` against missing socket | **OK** — exits non-zero with connect error (`No such file or directory` when sock absent; `daemon not responding` on I/O timeout) |
| Daemon IPC smoke (custom `--socket`) | **OK** — bind listen, `status` / `start` / `stop` NDJSON round-trips |
| Inject probe on this host | **OK** — `inject_available: true` (virtual-keyboard-unstable-v1 bound) |
| Overlay GTK warm-up | **OK** — GTK thread starts; CSS load clean after removing invalid `spacing` rules |

## What needs system deps / assets (not a compile failure here)

This environment already has **libpipewire**, **gtk4**, **cmake/clang** (whisper-rs built), and a **Wayland** display. Still required for *end-to-end speech*:

### Whisper model (runtime)

Default path: `~/.local/share/sttd/models/ggml-base.en.bin`

```bash
mkdir -p ~/.local/share/sttd/models
curl -L -o ~/.local/share/sttd/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
# optional: ggml-small.en.bin
```

Without the model file, the daemon starts, surfaces a clear status `error`, and **Start** returns `error` (preload falls back to lazy; first Start still fails until the file exists).

### Distro packages (if building on a clean machine)

| Distro | Packages |
|--------|----------|
| **Arch** | `base-devel rust cmake clang pkgconf pipewire libpipewire gtk4 gtk4-layer-shell wayland wayland-protocols` |
| **Debian/Ubuntu** | `build-essential pkg-config cmake clang libpipewire-0.3-dev libspa-0.2-dev libgtk-4-dev libgtk4-layer-shell-dev libwayland-dev wayland-protocols` |
| **Fedora** | `gcc gcc-c++ cmake clang pkgconf-pkg-config pipewire-devel gtk4-devel gtk4-layer-shell-devel wayland-devel wayland-protocols-devel` |

See also [README.md](README.md) feature flags:

```bash
cargo check -p proto -p sttctl
cargo check -p sttd --no-default-features   # no PW/GTK/whisper link
cargo build --release                       # full daemon
```

### Compositor capabilities

- **gtk4-layer-shell** (wlr-layer-shell) for overlay
- **virtual-keyboard-unstable-v1** for injection (missing → status error string; capture still allowed)

## Runtime tests intentionally **not** claimed here

The following were **not** fully validated on this pass (no model file; no long speech session):

- Live mic → VAD close → whisper commit → keyboard inject into a focused client
- Edit mode full-buffer STT → EditReview TextView edit → Copy inject
- Overlay visual crossfade / pause UI during real capture
- PipeWire non-default node selection under load

Daemon *did* start under Wayland and reported inject ready; that is **not** the same as proving text appeared in an app.

## Architecture vs remaining TODOs

Implemented wiring matches v1 design:

- Unix socket `$XDG_RUNTIME_DIR/sttd.sock`, NDJSON `proto` (`Start/Stop/Toggle/SetMode/Pause/Resume/Status`)
- States: Idle / Recording / Paused / EditReview × Live|Edit
- PipeWire mono 16 kHz f32; open on Start, close on Stop; Pause = stop feeding VAD
- webrtc-vad ~500 ms silence gap; bounded STT command channel + backpressure drops
- whisper-rs worker thread; preload|lazy residency
- Live: commit+inject per chunk; preview overlay-only (throttled)
- Edit: accumulate → full STT → EditReview; Copy/Discard
- Overlay: Layer::Overlay; keyboard None recording; OnDemand EditReview only
- Inject: virtual-keyboard-unstable-v1 via wayland-client (not wtype/ydotool)

Possible follow-ups (not blockers for compile):

1. Install model + run a real Live inject demo in a focused terminal.
2. Align `contrib/config.example.toml` keys fully with section-11 names (`vad_silence_ms`, `crossfade_ms`, `path`) — aliases already work.
3. Optional: map overlay `anchor` from config into `OverlayLayout` (layout defaults are hard-coded bottom-stretch today).
4. Live preview STT competes with LiveChunk on the same worker queue — under heavy load, previews may starve; acceptable for v1.
5. First-run helper: copy example config / print model download URL when file missing.

## How to bind a key

### Hyprland

```conf
# ~/.config/hypr/hyprland.conf
bind = SUPER, Space, exec, sttctl toggle --mode live
bind = SUPER SHIFT, Space, exec, sttctl toggle --mode edit
bind = SUPER, Escape, exec, sttctl stop
```

### Sway

```conf
# ~/.config/sway/config
bindsym $mod+space exec sttctl toggle --mode live
bindsym $mod+Shift+space exec sttctl toggle --mode edit
bindsym $mod+Escape exec sttctl stop
```

Put `sttctl` on `PATH` (e.g. `~/.local/bin` or `target/release`).

### systemd user unit

```bash
mkdir -p ~/.config/systemd/user
cp contrib/sttd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sttd.service
```

## Quick verify commands

```bash
cd /path/to/stt-wayland
cargo build --workspace
cargo test -p proto
./target/debug/sttctl status          # expect connect error if daemon down
RUST_LOG=info ./target/debug/sttd     # then, other terminal:
sttctl status
sttctl toggle --mode live
```
