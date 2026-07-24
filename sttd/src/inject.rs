//! Text injection via `virtual-keyboard-unstable-v1` (wayland-client).
//!
//! Native Wayland only — no wtype/ydotool/Wine.
//! Injects committed text in per-chunk batches.
//! Missing protocol → clear status error (do not crash).
//!
//! Approach (same idea as wtype): build a one-shot XKB keymap that maps each
//! needed Unicode scalar to its own keycode, upload it through
//! `zwp_virtual_keyboard_v1.keymap`, then emit press/release pairs for the
//! whole committed chunk. No window/surface is created.

use anyhow::{bail, Result};
use tracing::{debug, warn};

use crate::config::InjectionConfig;

#[cfg(feature = "inject")]
use anyhow::{anyhow, Context};
#[cfg(feature = "inject")]
use tracing::info;

/// Virtual-keyboard text injector.
///
/// Pure Wayland client — never creates a window/surface.
pub struct Injector {
    enabled: bool,
    available: bool,
    last_error: Option<String>,
    /// When true (default), inject a committed chunk as fast as the protocol allows.
    batch_mode: bool,
    char_delay_ms: u32,
    #[cfg(feature = "inject")]
    backend: Option<VkBackend>,
}

impl Injector {
    pub fn new(cfg: &InjectionConfig) -> Result<Self> {
        #[cfg(feature = "inject")]
        {
            match VkBackend::connect() {
                Ok(backend) => {
                    debug!(
                        enabled = cfg.enabled,
                        batch_mode = cfg.batch_mode,
                        "inject: virtual-keyboard-unstable-v1 ready"
                    );
                    Ok(Self {
                        enabled: cfg.enabled,
                        available: true,
                        last_error: None,
                        batch_mode: cfg.batch_mode,
                        char_delay_ms: cfg.char_delay_ms,
                        backend: Some(backend),
                    })
                }
                Err(err) => {
                    let msg = format!("{err:#}");
                    if cfg.enabled {
                        warn!(
                            error = %msg,
                            "inject: virtual-keyboard unavailable; text injection disabled"
                        );
                    } else {
                        debug!(error = %msg, "inject: probe failed (injection disabled in config)");
                    }
                    Ok(Self {
                        enabled: cfg.enabled,
                        available: false,
                        last_error: Some(msg),
                        batch_mode: cfg.batch_mode,
                        char_delay_ms: cfg.char_delay_ms,
                        backend: None,
                    })
                }
            }
        }

        #[cfg(not(feature = "inject"))]
        {
            let msg = "inject feature not compiled in (rebuild with --features inject)".to_string();
            if cfg.enabled {
                warn!("{msg}");
            }
            Ok(Self {
                enabled: cfg.enabled,
                available: false,
                last_error: Some(msg),
                batch_mode: cfg.batch_mode,
                char_delay_ms: cfg.char_delay_ms,
            })
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.available
    }

    /// Last init / runtime error for Status reporting.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Human-readable error when injection was requested but unavailable.
    pub fn unavailable_message() -> &'static str {
        "virtual-keyboard-unstable-v1 protocol is not available on this compositor"
    }

    /// Inject committed text (batch for the whole chunk). No-op if disabled/unavailable.
    pub fn inject_text(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        if !self.is_enabled() {
            if self.enabled && !self.available {
                bail!(
                    "{}",
                    self.last_error
                        .as_deref()
                        .unwrap_or_else(|| Self::unavailable_message())
                );
            }
            return Ok(());
        }

        // Whole committed chunk is always one inject_text call (batch from the
        // outside). `batch_mode` only controls inter-character delay.
        let delay = if self.batch_mode {
            0
        } else {
            self.char_delay_ms
        };

        debug!(
            n_chars = text.chars().count(),
            delay_ms = delay,
            batch_mode = self.batch_mode,
            "inject: batch"
        );

        #[cfg(feature = "inject")]
        {
            let backend = self
                .backend
                .as_mut()
                .ok_or_else(|| anyhow!(Self::unavailable_message()))?;
            match backend.inject_text(text, delay) {
                Ok(()) => {
                    self.last_error = None;
                    Ok(())
                }
                Err(err) => {
                    let msg = format!("{err:#}");
                    self.last_error = Some(msg.clone());
                    Err(anyhow!(msg))
                }
            }
        }

        #[cfg(not(feature = "inject"))]
        {
            let _ = text;
            bail!("{}", Self::unavailable_message());
        }
    }
}

// ─── Wayland backend (feature = "inject") ───────────────────────────────────

#[cfg(feature = "inject")]
mod wl {
    use super::*;
    use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
    use std::time::{Duration, Instant};

    use wayland_client::{
        globals::{registry_queue_init, GlobalListContents},
        protocol::{wl_registry, wl_seat},
        Connection, Dispatch, EventQueue, QueueHandle,
    };
    use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
        zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
        zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
    };

    /// Linux/evdev key state values used by the virtual-keyboard protocol.
    const KEY_RELEASED: u32 = 0;
    const KEY_PRESSED: u32 = 1;

    /// XKB keymap format (`wl_keyboard.keymap_format.xkb_v1`).
    const KEYMAP_FORMAT_XKB_V1: u32 = 1;

    /// Minimal press→release settle so focused clients see a real stroke.
    const KEY_SETTLE: Duration = Duration::from_millis(2);

    pub(super) struct VkBackend {
        conn: Connection,
        queue: EventQueue<VkState>,
        state: VkState,
        /// Kept alive so the compositor keeps the seat association valid.
        _seat: wl_seat::WlSeat,
        _manager: ZwpVirtualKeyboardManagerV1,
        keyboard: ZwpVirtualKeyboardV1,
        /// Monotonic-ish millisecond clock for key timestamps.
        t0: Instant,
    }

    /// Dispatch state (no protocol events need handling for injection).
    pub(super) struct VkState;

    impl VkBackend {
        pub(super) fn connect() -> Result<Self> {
            let conn = Connection::connect_to_env().context(
                "failed to connect to Wayland display (is WAYLAND_DISPLAY set?)",
            )?;

            let (globals, mut queue) = registry_queue_init::<VkState>(&conn)
                .context("failed to initialize Wayland registry")?;
            let qh = queue.handle();

            // Require virtual-keyboard-unstable-v1.
            let manager: ZwpVirtualKeyboardManagerV1 = match globals.bind(&qh, 1..=1, ()) {
                Ok(m) => m,
                Err(_) => {
                    bail!("{}", super::Injector::unavailable_message());
                }
            };

            // First seat is fine for a pure input client.
            let seat: wl_seat::WlSeat = globals
                .bind(&qh, 1..=9, ())
                .context("no wl_seat advertised by compositor")?;

            // Drain any seat capability events.
            let mut state = VkState;
            queue
                .roundtrip(&mut state)
                .context("Wayland roundtrip after binding seat/manager failed")?;

            let keyboard = manager.create_virtual_keyboard(&seat, &qh, ());

            queue
                .roundtrip(&mut state)
                .context("Wayland roundtrip after create_virtual_keyboard failed")?;

            info!("inject: bound zwp_virtual_keyboard_manager_v1 + seat");

            Ok(Self {
                conn,
                queue,
                state,
                _seat: seat,
                _manager: manager,
                keyboard,
                t0: Instant::now(),
            })
        }

        fn now_ms(&self) -> u32 {
            self.t0.elapsed().as_millis() as u32
        }

        fn flush(&mut self) -> Result<()> {
            self.queue
                .roundtrip(&mut self.state)
                .context("Wayland roundtrip failed during injection")?;
            let _ = self.conn.flush();
            Ok(())
        }

        pub(super) fn inject_text(&mut self, text: &str, char_delay_ms: u32) -> Result<()> {
            let (keymap_entries, key_seq) = build_keymap(text);
            if key_seq.is_empty() {
                return Ok(());
            }

            let keymap = render_xkb_keymap(&keymap_entries);
            self.upload_keymap(&keymap)?;

            for (i, &keycode) in key_seq.iter().enumerate() {
                self.type_keycode(keycode)?;
                if char_delay_ms > 0 && i + 1 < key_seq.len() {
                    std::thread::sleep(Duration::from_millis(u64::from(char_delay_ms)));
                }
            }

            // Leave modifiers clear after a batch.
            self.keyboard.modifiers(0, 0, 0, 0);
            self.flush()?;
            Ok(())
        }

        fn upload_keymap(&mut self, keymap: &str) -> Result<()> {
            let (fd, size) = memfd_with_bytes(keymap.as_bytes())
                .context("failed to create memfd for XKB keymap")?;

            // Protocol: compositor mmap's the fd; size includes trailing NUL
            // (wtype-compatible).
            self.keyboard
                .keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), size);
            self.flush()
                .context("failed to upload virtual-keyboard keymap")?;
            // fd must stay open until after the roundtrip.
            drop(fd);
            Ok(())
        }

        fn type_keycode(&mut self, keycode: u32) -> Result<()> {
            let t = self.now_ms();
            self.keyboard.key(t, keycode, KEY_PRESSED);
            self.flush()?;
            std::thread::sleep(KEY_SETTLE);

            let t = self.now_ms();
            self.keyboard.key(t, keycode, KEY_RELEASED);
            self.flush()?;
            std::thread::sleep(KEY_SETTLE);
            Ok(())
        }
    }

    // ── Dispatch impls (noop — we only send requests) ───────────────────────

    impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for VkState {
        fn event(
            _state: &mut Self,
            _proxy: &wl_registry::WlRegistry,
            _event: wl_registry::Event,
            _data: &GlobalListContents,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            // Dynamic global add/remove ignored; injection uses the initial set.
        }
    }

    impl Dispatch<wl_seat::WlSeat, ()> for VkState {
        fn event(
            _state: &mut Self,
            _proxy: &wl_seat::WlSeat,
            _event: wl_seat::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            // Capabilities / name not needed for virtual keyboard.
        }
    }

    impl Dispatch<ZwpVirtualKeyboardManagerV1, ()> for VkState {
        fn event(
            _state: &mut Self,
            _proxy: &ZwpVirtualKeyboardManagerV1,
            _event: wayland_protocols_misc::zwp_virtual_keyboard_v1::client::zwp_virtual_keyboard_manager_v1::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            // No events defined on the manager.
        }
    }

    impl Dispatch<ZwpVirtualKeyboardV1, ()> for VkState {
        fn event(
            _state: &mut Self,
            _proxy: &ZwpVirtualKeyboardV1,
            _event: wayland_protocols_misc::zwp_virtual_keyboard_v1::client::zwp_virtual_keyboard_v1::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            // No events defined on the keyboard object.
        }
    }

    // ── Keymap construction ─────────────────────────────────────────────────

    /// One entry in the temporary XKB map (1-based index = linux keycode).
    struct KeymapEntry {
        /// XKB keysym name, e.g. "a", "Return", "U2014".
        keysym: String,
    }

    /// Build unique keymap entries and the sequence of 1-based keycodes for `text`.
    fn build_keymap(text: &str) -> (Vec<KeymapEntry>, Vec<u32>) {
        let mut entries: Vec<KeymapEntry> = Vec::new();
        // char → 1-based keycode
        let mut index: Vec<(char, u32)> = Vec::new();
        let mut seq: Vec<u32> = Vec::with_capacity(text.chars().count());

        for ch in text.chars() {
            let keycode = if let Some(&(_, kc)) = index.iter().find(|(c, _)| *c == ch) {
                kc
            } else {
                let keysym = char_to_keysym_name(ch);
                entries.push(KeymapEntry { keysym });
                let kc = entries.len() as u32; // 1-based
                index.push((ch, kc));
                kc
            };
            seq.push(keycode);
        }

        (entries, seq)
    }

    /// Map a Unicode scalar to an XKB keysym name.
    ///
    /// Specials get well-known names; everything else uses the `UXXXX` form so
    /// we do not need libxkbcommon at runtime for reverse lookup. Each character
    /// is a level-1 key of its own, so Shift is never required.
    fn char_to_keysym_name(ch: char) -> String {
        match ch {
            '\n' | '\r' => "Return".to_string(),
            '\t' => "Tab".to_string(),
            '\u{1b}' => "Escape".to_string(),
            ' ' => "space".to_string(),
            c if c.is_ascii_graphic() => match c {
                '!' => "exclam".into(),
                '"' => "quotedbl".into(),
                '#' => "numbersign".into(),
                '$' => "dollar".into(),
                '%' => "percent".into(),
                '&' => "ampersand".into(),
                '\'' => "apostrophe".into(),
                '(' => "parenleft".into(),
                ')' => "parenright".into(),
                '*' => "asterisk".into(),
                '+' => "plus".into(),
                ',' => "comma".into(),
                '-' => "minus".into(),
                '.' => "period".into(),
                '/' => "slash".into(),
                ':' => "colon".into(),
                ';' => "semicolon".into(),
                '<' => "less".into(),
                '=' => "equal".into(),
                '>' => "greater".into(),
                '?' => "question".into(),
                '@' => "at".into(),
                '[' => "bracketleft".into(),
                '\\' => "backslash".into(),
                ']' => "bracketright".into(),
                '^' => "asciicircum".into(),
                '_' => "underscore".into(),
                '`' => "grave".into(),
                '{' => "braceleft".into(),
                '|' => "bar".into(),
                '}' => "braceright".into(),
                '~' => "asciitilde".into(),
                // a-z A-Z 0-9: single-char XKB names.
                c => c.to_string(),
            },
            // Unicode keysym: U+XXXX → "UXXXX" (xkbcommon accepts this form).
            c => format!("U{:04X}", c as u32),
        }
    }

    /// Render a complete XKB keymap text blob (NUL-terminated for wtype parity).
    fn render_xkb_keymap(entries: &[KeymapEntry]) -> String {
        // XKB keycodes used: 9 .. 8+len  → linux/evdev keycodes 1 .. len
        let max_keycode = entries.len() + 8 + 1;
        let mut out = String::with_capacity(256 + entries.len() * 48);
        out.push_str("xkb_keymap {\n");
        out.push_str("xkb_keycodes \"(unnamed)\" {\n");
        out.push_str("minimum = 8;\n");
        out.push_str(&format!("maximum = {max_keycode};\n"));
        for i in 0..entries.len() {
            // <K1> = 9, linux keycode 1
            let name_idx = i + 1;
            let xkb_code = i + 8 + 1;
            out.push_str(&format!("<K{name_idx}> = {xkb_code};\n"));
        }
        out.push_str("};\n");
        out.push_str("xkb_types \"(unnamed)\" { include \"complete\" };\n");
        out.push_str("xkb_compatibility \"(unnamed)\" { include \"complete\" };\n");
        out.push_str("xkb_symbols \"(unnamed)\" {\n");
        for (i, entry) in entries.iter().enumerate() {
            let name_idx = i + 1;
            out.push_str(&format!("key <K{name_idx}> {{[{}]}};\n", entry.keysym));
        }
        out.push_str("};\n");
        out.push_str("};\n");
        // Trailing NUL is part of the blob size (matches wtype).
        out.push('\0');
        out
    }

    /// Create a memfd (or unlinked temp file), write `bytes`, rewind, return (fd, size).
    fn memfd_with_bytes(bytes: &[u8]) -> Result<(OwnedFd, u32)> {
        let fd = unsafe {
            let name = c"sttd-keymap";
            let raw = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC);
            if raw >= 0 {
                OwnedFd::from_raw_fd(raw)
            } else {
                // Fallback: mkstemp + unlink so the fd is the only reference.
                let mut path = std::ffi::CString::new("/tmp/sttd-keymap-XXXXXX")
                    .map_err(|e| anyhow!("path: {e}"))?
                    .into_bytes_with_nul();
                let raw = libc::mkstemp(path.as_mut_ptr().cast::<libc::c_char>());
                if raw < 0 {
                    return Err(std::io::Error::last_os_error())
                        .context("memfd_create and mkstemp both failed");
                }
                let _ = libc::unlink(path.as_ptr().cast::<libc::c_char>());
                OwnedFd::from_raw_fd(raw)
            }
        };

        let mut written = 0usize;
        while written < bytes.len() {
            let n = unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    bytes[written..].as_ptr().cast::<libc::c_void>(),
                    bytes.len() - written,
                )
            };
            if n < 0 {
                return Err(std::io::Error::last_os_error()).context("write keymap memfd");
            }
            written += n as usize;
        }

        let off = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_SET) };
        if off < 0 {
            return Err(std::io::Error::last_os_error()).context("lseek keymap memfd");
        }

        let size = u32::try_from(written).context("keymap too large")?;
        Ok((fd, size))
    }
}

#[cfg(feature = "inject")]
use wl::VkBackend;

#[cfg(all(test, feature = "inject"))]
mod tests {
    use super::*;

    #[test]
    fn connect_reports_availability() {
        // Does not require a live compositor for the test to pass — only checks
        // that new() never panics and surfaces a clear error when unavailable.
        let cfg = InjectionConfig::default();
        let inj = Injector::new(&cfg).expect("Injector::new must not fail hard");
        if inj.is_available() {
            assert!(inj.last_error().is_none());
            assert!(inj.is_enabled());
        } else {
            let err = inj.last_error().unwrap_or("");
            assert!(
                !err.is_empty(),
                "when unavailable, last_error() must be set"
            );
        }
    }

    #[test]
    fn inject_empty_is_ok() {
        let cfg = InjectionConfig::default();
        let mut inj = Injector::new(&cfg).unwrap();
        inj.inject_text("").unwrap();
    }
}
