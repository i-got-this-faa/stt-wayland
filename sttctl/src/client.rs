//! Unix-socket client for talking to `sttd`.
//!
//! Wire: newline-delimited JSON ([`proto::Request`] / [`proto::Response`]).
//! Status requests use a short timeout; mutating commands use the longer
//! [`proto::OPERATION_TIMEOUT_MS`] budget because Stop may wait for Whisper to
//! finish the final utterance.
//!
//! Socket path resolution (highest priority first):
//! 1. Explicit path passed to [`Client::from_override`]
//! 2. `$STTD_SOCK` environment variable
//! 3. [`proto::default_socket_path`] (`$XDG_RUNTIME_DIR/sttd.sock`)

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use proto::{
    decode_line, default_socket_path, encode_line, Request, Response, CLIENT_TIMEOUT_MS,
    OPERATION_TIMEOUT_MS,
};

/// Message printed (and used as the error) when the daemon times out.
pub const DAEMON_NOT_RESPONDING: &str = "daemon not responding";

/// Thin synchronous client used by the CLI.
#[derive(Debug, Clone)]
pub struct Client {
    socket_path: PathBuf,
}

impl Client {
    /// Resolve socket from CLI override, else `$STTD_SOCK`, else default path.
    pub fn from_override(socket: Option<PathBuf>) -> Self {
        Self {
            socket_path: resolve_socket_path(socket),
        }
    }

    /// Send one JSON request line, read one response line.
    ///
    /// On connect/read/write timeout, returns an error whose message is
    /// [`DAEMON_NOT_RESPONDING`]. Status uses the short timeout; operations
    /// use the model/inference timeout.
    pub fn call(&self, request: &Request) -> Result<Response> {
        let mut stream = match UnixStream::connect(&self.socket_path) {
            Ok(s) => s,
            Err(err) if is_timeout_error(&err) => {
                bail!("{DAEMON_NOT_RESPONDING}");
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "failed to connect to sttd at {} (is the daemon running?)",
                        self.socket_path.display()
                    )
                });
            }
        };

        let timeout = request_timeout(request);
        if let Err(err) = stream.set_read_timeout(Some(timeout)) {
            return Err(map_io_error(err, "set_read_timeout"));
        }
        if let Err(err) = stream.set_write_timeout(Some(timeout)) {
            return Err(map_io_error(err, "set_write_timeout"));
        }

        let payload = encode_line(request).context("encode request")?;
        if let Err(err) = stream.write_all(payload.as_bytes()) {
            return Err(map_io_error(err, "write request"));
        }
        if let Err(err) = stream.flush() {
            return Err(map_io_error(err, "flush request"));
        }

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(err) => return Err(map_io_error(err, "read response line")),
        };
        if n == 0 {
            return Err(anyhow!("daemon closed connection without a response"));
        }

        decode_line::<Response>(&line).context("decode response")
    }
}

fn request_timeout(request: &Request) -> Duration {
    match request {
        Request::Status => Duration::from_millis(CLIENT_TIMEOUT_MS),
        Request::Start { .. }
        | Request::Stop
        | Request::Toggle { .. }
        | Request::SetMode { .. }
        | Request::Pause
        | Request::Resume => Duration::from_millis(OPERATION_TIMEOUT_MS),
    }
}

/// Socket path: CLI override → `$STTD_SOCK` → `$XDG_RUNTIME_DIR/sttd.sock`.
fn resolve_socket_path(cli_override: Option<PathBuf>) -> PathBuf {
    if let Some(path) = cli_override {
        return path;
    }
    if let Ok(path) = std::env::var("STTD_SOCK") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    default_socket_path()
}

fn is_timeout_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

fn map_io_error(err: io::Error, context: &'static str) -> anyhow::Error {
    if is_timeout_error(&err) {
        anyhow!("{DAEMON_NOT_RESPONDING}")
    } else {
        anyhow::Error::new(err).context(context)
    }
}
