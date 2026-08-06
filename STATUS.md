# stt-wayland — STATUS

Last verified: 2026-08-01 (workspace on Linux, Wayland session present).

## What works (build / unit / IPC)

| Item | Status |
|------|--------|
| `cargo check --workspace` | **OK** (default features) |
| `cargo build --workspace` | **OK** → `target/debug/sttd`, `target/debug/sttctl` |
| `cargo build --workspace --release` | **OK** → release binaries build successfully |
| `cargo check -p sttd --no-default-features` | **OK** (stub backends; no system GUI/audio required) |
| `cargo test -p proto` | **OK** (25 tests + 1 doctest) |
| `cargo test --workspace` | **OK** (proto + sttd unit tests: state, config, vad, stt helpers, inject probe) |
| `stt-settings` GTK/libadwaita application | **OK** — profile/model/settings editor builds and uses the shared TOML schema (runtime launch pending verification) |
| `sttctl` against missing socket | **OK** — exits non-zero with connect error (`No such file or directory` when sock absent; `daemon not responding` on I/O timeout) |
| Daemon IPC smoke (custom `--socket`) | **OK** — bind listen, `status` / `start` / `stop` NDJSON round-trips |
| Inject probe on this host | **OK** — `inject_available: true` (virtual-keyboard-unstable-v1 bound) |
| Overlay GTK warm-up | **OK** — GTK thread starts; CSS load clean after removing invalid `spacing` rules |
| Release speech pipeline | **OK** — sample speech passed through PipeWire/VAD/Whisper and appeared in a focused Wayland text field |

## What needs system deps / assets (not a compile failure here)

This environment already has **libpipewire**, **gtk4**, **cmake/clang** (whisper-rs built), a **Wayland** display, and the model below. On another machine, the following runtime dependencies are still required for *end-to-end speech*:

### Whisper model (runtime)

Default path: `~/.local/share/sttd/models/ggml-base.en.bin`

```bash
mkdir -p ~/.local/share/sttd/models
curl -L -o ~/.local/share/sttd/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
# optional: ggml-small.en.bin
```

Without the model file, the daemon starts, surfaces a clear status `error`, and **Start** returns `error` (preload falls back to lazy; first Start still fails until the file exists). It is installed here at `/home/radhey/.local/share/sttd/models/ggml-base.en.bin`.

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

## Runtime validation

The bundled release daemon was validated on this Wayland session with the
official `ggml-base.en.bin` model installed at the default path:

- Live audio from an explicit PipeWire source was chunked by VAD, transcribed
  by Whisper, and injected into a focused GTK text field.
- `start`, `stop`, `toggle`, `pause`, `resume`, `status`, and edit-review IPC
  paths were exercised.
- Overlay positioning was checked visually; the default bottom card is
  centered.
- A second daemon correctly refuses an already-active socket.

Manual microphone quality, edit-review text editing/copy, and non-default
device selection still depend on the user’s preferred application and audio
hardware.

## Architecture vs remaining TODOs

Implemented wiring matches v1 design:

- Unix socket `$XDG_RUNTIME_DIR/sttd.sock`, NDJSON `proto` (`Start/Stop/Toggle/SetMode/Pause/Resume/Status`)
- States: Idle / Recording / Paused / EditReview × Live|Edit
- PipeWire mono 16 kHz f32; open on Start, close on Stop; Pause = stop feeding VAD
- webrtc-vad ~500 ms silence gap; bounded final STT channel with coalesced,
  lossy preview work
- whisper-rs worker thread; preload|lazy residency
- Live: commit+inject per chunk; preview overlay-only (throttled)
- Edit: accumulate → full STT → EditReview; Copy/Discard
- Overlay: Layer::Overlay; keyboard None recording; OnDemand EditReview only
- Inject: virtual-keyboard-unstable-v1 via wayland-client (not wtype/ydotool)

Possible follow-ups (not blockers for normal use):

1. First-run helper: copy example config / print model download URL when the file is missing.

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
