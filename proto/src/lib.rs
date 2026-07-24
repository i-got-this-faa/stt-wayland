//! Shared newline-delimited JSON IPC protocol for `sttd` / `sttctl`.
//!
//! # Wire format
//!
//! One JSON object per line over a Unix domain socket at
//! `$XDG_RUNTIME_DIR/sttd.sock` ([`SOCKET_NAME`]).
//!
//! Clients send a single [`Request`] line and read a single [`Response`] line.
//! Recommended connect/read/write timeout: [`CLIENT_TIMEOUT_MS`] (500 ms).
//!
//! # Example
//!
//! ```
//! use proto::{Request, Response, Mode, parse_request_line, to_response_line, encode_line, decode_line};
//!
//! let req = Request::Start { mode: Mode::Live };
//! let wire = encode_line(&req).unwrap();
//! assert!(wire.ends_with('\n'));
//!
//! let parsed = parse_request_line(&wire).unwrap();
//! assert_eq!(parsed, req);
//!
//! let resp = Response::ok();
//! let out = to_response_line(&resp).unwrap();
//! let back: Response = decode_line(&out).unwrap();
//! assert_eq!(back, resp);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// Default socket file name under `$XDG_RUNTIME_DIR`.
pub const SOCKET_NAME: &str = "sttd.sock";

/// Recommended client connect / read / write timeout (milliseconds).
pub const CLIENT_TIMEOUT_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

/// Operating mode for capture / injection behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// VAD-closed chunks are committed and injected immediately.
    /// In-progress preview is overlay-only (~200 ms crossfade).
    #[default]
    Live,
    /// Accumulate transcription into an editable buffer ([`DaemonState::EditReview`]).
    /// Injection happens only on explicit Copy.
    Edit,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Live => write!(f, "live"),
            Mode::Edit => write!(f, "edit"),
        }
    }
}

impl std::str::FromStr for Mode {
    type Err = ParseModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "live" => Ok(Mode::Live),
            "edit" => Ok(Mode::Edit),
            other => Err(ParseModeError(other.to_string())),
        }
    }
}

/// Error returned when parsing a [`Mode`] from a string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown mode '{0}' (expected live|edit)")]
pub struct ParseModeError(pub String);

// ---------------------------------------------------------------------------
// DaemonState
// ---------------------------------------------------------------------------

/// High-level daemon lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    /// Not capturing; overlay hidden (or idle).
    #[default]
    Idle,
    /// Actively capturing audio and running VAD/STT.
    Recording,
    /// Stream may stay open; VAD is not fed (no new chunks).
    Paused,
    /// Edit-mode review: buffer editable, keyboard interactivity OnDemand.
    EditReview,
}

impl fmt::Display for DaemonState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaemonState::Idle => write!(f, "idle"),
            DaemonState::Recording => write!(f, "recording"),
            DaemonState::Paused => write!(f, "paused"),
            DaemonState::EditReview => write!(f, "edit_review"),
        }
    }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Client → daemon requests (one JSON object per line).
///
/// Serialized as externally tagged objects with a `"type"` field using
/// `snake_case` variant names, e.g. `{"type":"start","mode":"live"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Begin capture in the given mode.
    ///
    /// Start while already [`DaemonState::Recording`] is a no-op `Ok`.
    Start { mode: Mode },
    /// Stop capture; tear down stream; return to Idle (or EditReview if Edit).
    Stop,
    /// Toggle: Idle/Paused → Start(mode); Recording → Stop.
    Toggle { mode: Mode },
    /// Change mode without necessarily starting capture.
    SetMode { mode: Mode },
    /// Stop feeding VAD without tearing the PipeWire stream.
    Pause,
    /// Resume feeding VAD after Pause.
    Resume,
    /// Query current status (daemon replies with [`Response::Status`]).
    Status,
}

impl Request {
    /// Encode this request as a single newline-terminated JSON line.
    pub fn to_line(&self) -> Result<String, serde_json::Error> {
        encode_line(self)
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Daemon → client responses (one JSON object per line).
///
/// Serialized as externally tagged objects with a `"type"` field using
/// `snake_case` variant names, e.g. `{"type":"ok"}` or
/// `{"type":"error","message":"..."}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// Request succeeded.
    Ok {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    /// Request failed with a clear, user-facing error.
    ///
    /// Example: missing `virtual-keyboard-unstable-v1` protocol.
    Error { message: String },
    /// Full status snapshot (response to [`Request::Status`]).
    Status {
        state: DaemonState,
        mode: Mode,
        /// Sticky / last operational error (e.g. inject protocol missing).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Loaded model identifier, if known (e.g. `"base.en"`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Whether `virtual-keyboard-unstable-v1` is available for injection.
        #[serde(default)]
        inject_available: bool,
    },
}

impl Response {
    /// Successful response with no message.
    pub fn ok() -> Self {
        Response::Ok { message: None }
    }

    /// Successful response with an optional informational message.
    pub fn ok_msg(msg: impl Into<String>) -> Self {
        Response::Ok {
            message: Some(msg.into()),
        }
    }

    /// Error response with a user-facing message.
    pub fn error(msg: impl Into<String>) -> Self {
        Response::Error {
            message: msg.into(),
        }
    }

    /// Encode this response as a single newline-terminated JSON line.
    pub fn to_line(&self) -> Result<String, serde_json::Error> {
        to_response_line(self)
    }
}

// ---------------------------------------------------------------------------
// Line helpers (newline-delimited JSON)
// ---------------------------------------------------------------------------

/// Encode a value as a single JSON line (trailing `\n`).
pub fn encode_line<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut s = serde_json::to_string(value)?;
    s.push('\n');
    Ok(s)
}

/// Decode a single JSON line (trailing whitespace / newline ignored).
pub fn decode_line<T: for<'de> Deserialize<'de>>(line: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(line.trim())
}

/// Parse a client request from one newline-delimited JSON line.
///
/// Accepts a line with or without a trailing `\n`.
pub fn parse_request_line(line: &str) -> Result<Request, serde_json::Error> {
    decode_line(line)
}

/// Encode a daemon response as one newline-terminated JSON line.
pub fn to_response_line(response: &Response) -> Result<String, serde_json::Error> {
    encode_line(response)
}

/// Encode a client request as one newline-terminated JSON line.
pub fn to_request_line(request: &Request) -> Result<String, serde_json::Error> {
    encode_line(request)
}

/// Parse a daemon response from one newline-delimited JSON line.
pub fn parse_response_line(line: &str) -> Result<Response, serde_json::Error> {
    decode_line(line)
}

// ---------------------------------------------------------------------------
// Socket path
// ---------------------------------------------------------------------------

/// Resolve the default sttd Unix socket path: `$XDG_RUNTIME_DIR/sttd.sock`.
///
/// Falls back to `/tmp/sttd.sock` if `XDG_RUNTIME_DIR` is unset (not ideal for
/// production; the daemon should refuse to start without `XDG_RUNTIME_DIR`).
pub fn default_socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(dir).join(SOCKET_NAME)
    } else {
        PathBuf::from("/tmp").join(SOCKET_NAME)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Request round-trips (every variant) --------------------------------

    fn assert_request_roundtrip(req: Request) {
        let line = to_request_line(&req).expect("encode request");
        assert!(
            line.ends_with('\n'),
            "request line must end with newline: {line:?}"
        );
        assert_eq!(
            line.matches('\n').count(),
            1,
            "exactly one trailing newline expected"
        );

        let via_parse = parse_request_line(&line).expect("parse_request_line");
        assert_eq!(via_parse, req);

        // Without trailing newline still works.
        let trimmed = line.trim_end_matches('\n');
        let via_trim = parse_request_line(trimmed).expect("parse without newline");
        assert_eq!(via_trim, req);

        // encode_line / decode_line are equivalent.
        let via_encode = encode_line(&req).unwrap();
        assert_eq!(via_encode, line);
        let via_decode: Request = decode_line(&via_encode).unwrap();
        assert_eq!(via_decode, req);

        // Request::to_line matches.
        assert_eq!(req.to_line().unwrap(), line);
    }

    #[test]
    fn roundtrip_request_start_live() {
        assert_request_roundtrip(Request::Start { mode: Mode::Live });
    }

    #[test]
    fn roundtrip_request_start_edit() {
        assert_request_roundtrip(Request::Start { mode: Mode::Edit });
    }

    #[test]
    fn roundtrip_request_stop() {
        assert_request_roundtrip(Request::Stop);
    }

    #[test]
    fn roundtrip_request_toggle_live() {
        assert_request_roundtrip(Request::Toggle { mode: Mode::Live });
    }

    #[test]
    fn roundtrip_request_toggle_edit() {
        assert_request_roundtrip(Request::Toggle { mode: Mode::Edit });
    }

    #[test]
    fn roundtrip_request_set_mode_live() {
        assert_request_roundtrip(Request::SetMode { mode: Mode::Live });
    }

    #[test]
    fn roundtrip_request_set_mode_edit() {
        assert_request_roundtrip(Request::SetMode { mode: Mode::Edit });
    }

    #[test]
    fn roundtrip_request_pause() {
        assert_request_roundtrip(Request::Pause);
    }

    #[test]
    fn roundtrip_request_resume() {
        assert_request_roundtrip(Request::Resume);
    }

    #[test]
    fn roundtrip_request_status() {
        assert_request_roundtrip(Request::Status);
    }

    /// Catch missing Request variants if new ones are added without tests.
    #[test]
    fn roundtrip_all_request_variants_exhaustive() {
        let variants = [
            Request::Start { mode: Mode::Live },
            Request::Start { mode: Mode::Edit },
            Request::Stop,
            Request::Toggle { mode: Mode::Live },
            Request::Toggle { mode: Mode::Edit },
            Request::SetMode { mode: Mode::Live },
            Request::SetMode { mode: Mode::Edit },
            Request::Pause,
            Request::Resume,
            Request::Status,
        ];
        for req in variants {
            assert_request_roundtrip(req);
        }
    }

    // --- Stable JSON shapes -------------------------------------------------

    #[test]
    fn request_json_wire_shapes() {
        assert_eq!(
            serde_json::to_string(&Request::Start { mode: Mode::Live }).unwrap(),
            r#"{"type":"start","mode":"live"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Stop).unwrap(),
            r#"{"type":"stop"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Toggle { mode: Mode::Edit }).unwrap(),
            r#"{"type":"toggle","mode":"edit"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::SetMode { mode: Mode::Live }).unwrap(),
            r#"{"type":"set_mode","mode":"live"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Pause).unwrap(),
            r#"{"type":"pause"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Resume).unwrap(),
            r#"{"type":"resume"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Status).unwrap(),
            r#"{"type":"status"}"#
        );
    }

    // --- Response round-trips -----------------------------------------------

    fn assert_response_roundtrip(resp: Response) {
        let line = to_response_line(&resp).expect("to_response_line");
        assert!(line.ends_with('\n'));
        let back = parse_response_line(&line).expect("parse_response_line");
        assert_eq!(back, resp);
        assert_eq!(resp.to_line().unwrap(), line);
    }

    #[test]
    fn roundtrip_response_ok() {
        assert_response_roundtrip(Response::ok());
        assert_response_roundtrip(Response::ok_msg("started"));
    }

    #[test]
    fn roundtrip_response_error() {
        assert_response_roundtrip(Response::error(
            "virtual-keyboard-unstable-v1 not available",
        ));
    }

    #[test]
    fn roundtrip_response_status() {
        assert_response_roundtrip(Response::Status {
            state: DaemonState::Recording,
            mode: Mode::Edit,
            error: None,
            model: Some("base.en".into()),
            inject_available: true,
        });
        assert_response_roundtrip(Response::Status {
            state: DaemonState::Idle,
            mode: Mode::Live,
            error: Some("inject unavailable".into()),
            model: None,
            inject_available: false,
        });
        assert_response_roundtrip(Response::Status {
            state: DaemonState::Paused,
            mode: Mode::Live,
            error: None,
            model: Some("small.en".into()),
            inject_available: true,
        });
        assert_response_roundtrip(Response::Status {
            state: DaemonState::EditReview,
            mode: Mode::Edit,
            error: None,
            model: Some("base.en".into()),
            inject_available: true,
        });
    }

    #[test]
    fn response_ok_skips_null_message() {
        let json = serde_json::to_string(&Response::ok()).unwrap();
        assert_eq!(json, r#"{"type":"ok"}"#);
        // Deserializing without message field still works.
        let back: Response = serde_json::from_str(r#"{"type":"ok"}"#).unwrap();
        assert_eq!(back, Response::ok());
    }

    #[test]
    fn response_status_defaults() {
        // Minimal status JSON (optional fields omitted).
        let back: Response = serde_json::from_str(
            r#"{"type":"status","state":"idle","mode":"live"}"#,
        )
        .unwrap();
        assert_eq!(
            back,
            Response::Status {
                state: DaemonState::Idle,
                mode: Mode::Live,
                error: None,
                model: None,
                inject_available: false,
            }
        );
    }

    #[test]
    fn response_json_wire_shapes() {
        assert_eq!(
            serde_json::to_string(&Response::error("boom")).unwrap(),
            r#"{"type":"error","message":"boom"}"#
        );
        assert_eq!(
            serde_json::to_string(&Response::ok_msg("hi")).unwrap(),
            r#"{"type":"ok","message":"hi"}"#
        );
    }

    // --- Mode / DaemonState -------------------------------------------------

    #[test]
    fn mode_from_str_and_display() {
        assert_eq!("live".parse::<Mode>().unwrap(), Mode::Live);
        assert_eq!("EDIT".parse::<Mode>().unwrap(), Mode::Edit);
        assert_eq!(Mode::Live.to_string(), "live");
        assert_eq!(Mode::Edit.to_string(), "edit");
        let err = "nope".parse::<Mode>().unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn mode_serde_snake_case() {
        assert_eq!(serde_json::to_string(&Mode::Live).unwrap(), r#""live""#);
        assert_eq!(serde_json::to_string(&Mode::Edit).unwrap(), r#""edit""#);
    }

    #[test]
    fn daemon_state_display_and_serde() {
        assert_eq!(DaemonState::Idle.to_string(), "idle");
        assert_eq!(DaemonState::Recording.to_string(), "recording");
        assert_eq!(DaemonState::Paused.to_string(), "paused");
        assert_eq!(DaemonState::EditReview.to_string(), "edit_review");

        assert_eq!(
            serde_json::to_string(&DaemonState::EditReview).unwrap(),
            r#""edit_review""#
        );
        let s: DaemonState = serde_json::from_str(r#""edit_review""#).unwrap();
        assert_eq!(s, DaemonState::EditReview);
    }

    // --- Socket path / constants --------------------------------------------

    #[test]
    fn socket_name_and_timeout_constants() {
        assert_eq!(SOCKET_NAME, "sttd.sock");
        assert_eq!(CLIENT_TIMEOUT_MS, 500);
    }

    #[test]
    fn default_socket_path_uses_xdg_or_tmp() {
        let path = default_socket_path();
        assert!(path.ends_with(SOCKET_NAME));
        // Either $XDG_RUNTIME_DIR/sttd.sock or /tmp/sttd.sock
        let s = path.to_string_lossy();
        assert!(
            s.contains("sttd.sock"),
            "path should contain socket name: {s}"
        );
    }

    // --- Malformed input ----------------------------------------------------

    #[test]
    fn parse_request_rejects_garbage() {
        assert!(parse_request_line("not json").is_err());
        assert!(parse_request_line(r#"{"type":"nope"}"#).is_err());
        assert!(parse_request_line("").is_err());
    }

    #[test]
    fn parse_response_rejects_garbage() {
        assert!(parse_response_line("nope").is_err());
        assert!(parse_response_line(r#"{"type":"ok","message":1}"#).is_err());
    }
}
