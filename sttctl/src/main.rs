//! `sttctl` — thin control client for the `sttd` Wayland STT daemon.
//!
//! Sends one newline-delimited JSON request over the daemon Unix socket,
//! prints one response line, and exits. No daemon logic lives here.

mod client;

use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use client::{Client, DAEMON_NOT_RESPONDING};
use proto::{Mode, Request, Response};

#[derive(Debug, Parser)]
#[command(
    name = "sttctl",
    about = "Control the sttd Wayland speech-to-text daemon",
    version
)]
struct Cli {
    /// Override the Unix socket path.
    /// Default: `$STTD_SOCK`, else `$XDG_RUNTIME_DIR/sttd.sock`.
    #[arg(long, global = true)]
    socket: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start capture (default mode: live).
    Start {
        /// Capture mode: live | edit
        #[arg(long, value_enum, default_value_t = ModeArg::Live)]
        mode: ModeArg,
    },
    /// Stop capture.
    Stop,
    /// Toggle capture (start if idle, stop if recording).
    Toggle {
        /// Mode used when starting.
        #[arg(long, value_enum, default_value_t = ModeArg::Live)]
        mode: ModeArg,
    },
    /// Set operating mode without necessarily starting.
    Mode {
        #[arg(value_enum)]
        mode: ModeArg,
    },
    /// Pause VAD feeding (keep stream open).
    Pause,
    /// Resume VAD feeding after pause.
    Resume,
    /// Print daemon status as JSON (suitable for waybar).
    Status,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ModeArg {
    Live,
    Edit,
}

impl From<ModeArg> for Mode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Live => Mode::Live,
            ModeArg::Edit => Mode::Edit,
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            // Timeouts are a fixed, user-facing string (no "sttctl:" prefix).
            let msg = format!("{err:#}");
            if msg == DAEMON_NOT_RESPONDING || msg.ends_with(DAEMON_NOT_RESPONDING) {
                eprintln!("{DAEMON_NOT_RESPONDING}");
            } else {
                eprintln!("sttctl: {msg}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let client = Client::from_override(cli.socket);

    let request = match cli.command {
        Commands::Start { mode } => Request::Start { mode: mode.into() },
        Commands::Stop => Request::Stop,
        Commands::Toggle { mode } => Request::Toggle { mode: mode.into() },
        Commands::Mode { mode } => Request::SetMode { mode: mode.into() },
        Commands::Pause => Request::Pause,
        Commands::Resume => Request::Resume,
        Commands::Status => Request::Status,
    };

    let resp = client.call(&request)?;
    // Always emit one JSON response line (status is waybar-friendly).
    print!("{}", proto::encode_line(&resp)?);

    match resp {
        Response::Ok { .. } | Response::Status { error: None, .. } => Ok(ExitCode::SUCCESS),
        Response::Status { error: Some(_), .. } => Ok(ExitCode::from(2)),
        Response::Error { .. } => Ok(ExitCode::FAILURE),
    }
}
