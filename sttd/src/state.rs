//! Daemon state machine.
//!
//! # States
//!
//! ```text
//! Idle --Start(mode)--> Recording(Live|Edit)
//! Recording --Pause--> Paused --Resume--> Recording
//! Recording(Live) --Stop--> Idle   (flush, destroy overlay, close audio)
//! Recording(Edit) --Stop--> EditReview  (transcribe full buffer)
//! EditReview --Copy--> inject then Idle
//! EditReview --Discard--> Idle
//! ```
//!
//! # Modes
//!
//! - **Live** — VAD-closed chunks committed + injected; preview overlay-only.
//! - **Edit** — accumulate session audio; Stop → full STT → EditReview.
//!
//! # Edges
//!
//! - Start while already `Recording` → no-op Ok ([`StartResult::AlreadyRecording`]).
//! - Toggle: Idle/Paused/EditReview → Start(mode); Recording → Stop.
//! - SetMode mid-session updates the mode field; the daemon applies pipeline side effects.

use proto::{DaemonState, Mode, Response};
use tracing::debug;

use crate::config::ModelConfig;

/// In-memory application state shared across IPC and subsystems.
#[derive(Debug)]
pub struct AppState {
    pub state: DaemonState,
    pub mode: Mode,
    pub last_error: Option<String>,
    pub model_name: String,
    pub inject_available: bool,
    /// Edit-mode accumulated text (filled after full-buffer STT / EditReview edits).
    pub edit_buffer: String,
}

impl AppState {
    pub fn new(model: ModelConfig) -> Self {
        Self {
            state: DaemonState::Idle,
            mode: Mode::Live,
            last_error: None,
            model_name: model.name,
            inject_available: false, // set after inject probe
            edit_buffer: String::new(),
        }
    }

    pub fn set_error(&mut self, msg: impl Into<String>) {
        let m = msg.into();
        debug!(error = %m, "state: error");
        self.last_error = Some(m);
    }

    pub fn clear_error(&mut self) {
        self.last_error = None;
    }

    /// Start capture. Start while already Recording → no-op Ok.
    ///
    /// From Paused: resumes (returns [`StartResult::Resumed`]).
    /// From Idle / EditReview: begins a new session (clears edit buffer on Edit).
    pub fn start(&mut self, mode: Mode) -> StartResult {
        match self.state {
            DaemonState::Recording => {
                // Starting an already active session is deliberately a no-op.
                // Use SetMode when the caller explicitly wants to retarget a
                // live session; changing `mode` here would make status claim a
                // different pipeline than the one actually running.
                StartResult::AlreadyRecording
            }
            DaemonState::Paused => {
                self.mode = mode;
                self.state = DaemonState::Recording;
                debug!(?mode, "state: start → resumed from paused");
                StartResult::Resumed
            }
            DaemonState::Idle | DaemonState::EditReview => {
                let was_edit_review = self.state == DaemonState::EditReview;
                self.mode = mode;
                self.state = DaemonState::Recording;
                if mode == Mode::Edit || was_edit_review {
                    self.edit_buffer.clear();
                }
                debug!(?mode, "state: start → recording");
                StartResult::Started
            }
        }
    }

    /// Stop capture.
    ///
    /// Live → Idle. Edit → EditReview (caller transcribes full buffer).
    pub fn stop(&mut self) -> StopResult {
        match self.state {
            DaemonState::Idle => StopResult::AlreadyIdle,
            DaemonState::EditReview => StopResult::AlreadyIdle,
            DaemonState::Recording | DaemonState::Paused => {
                if self.mode == Mode::Edit {
                    self.state = DaemonState::EditReview;
                    debug!("state: stop → edit_review");
                    StopResult::EnteredEditReview
                } else {
                    self.state = DaemonState::Idle;
                    debug!("state: stop → idle");
                    StopResult::Stopped
                }
            }
        }
    }

    /// Toggle: Idle/Paused/EditReview → Start(mode); Recording → Stop.
    pub fn toggle(&mut self, mode: Mode) -> ToggleResult {
        match self.state {
            DaemonState::Recording => match self.stop() {
                StopResult::Stopped => ToggleResult::Stopped,
                StopResult::EnteredEditReview => ToggleResult::EnteredEditReview,
                StopResult::AlreadyIdle => ToggleResult::Stopped,
            },
            DaemonState::Idle | DaemonState::Paused | DaemonState::EditReview => {
                match self.start(mode) {
                    StartResult::Started => ToggleResult::Started,
                    StartResult::Resumed => ToggleResult::Resumed,
                    StartResult::AlreadyRecording => ToggleResult::AlreadyRecording,
                }
            }
        }
    }

    pub fn pause(&mut self) -> bool {
        if self.state == DaemonState::Recording {
            self.state = DaemonState::Paused;
            debug!("state: pause");
            true
        } else {
            false
        }
    }

    pub fn resume(&mut self) -> bool {
        if self.state == DaemonState::Paused {
            self.state = DaemonState::Recording;
            debug!("state: resume");
            true
        } else {
            false
        }
    }

    /// Change mode. Always updates `mode`; returns whether a live session is active
    /// so the daemon can retarget the pipeline (accumulate vs rolling Live).
    pub fn set_mode(&mut self, mode: Mode) -> SetModeResult {
        let prev = self.mode;
        self.mode = mode;
        debug!(?prev, ?mode, state = %self.state, "state: set_mode");

        match self.state {
            DaemonState::Recording | DaemonState::Paused if prev != mode => {
                SetModeResult::SessionModeChanged {
                    from: prev,
                    to: mode,
                }
            }
            _ => SetModeResult::Stored { mode },
        }
    }

    /// Leave EditReview (Copy or Discard) → Idle.
    pub fn finish_edit_review(&mut self) {
        self.edit_buffer.clear();
        self.state = DaemonState::Idle;
        debug!("state: finish_edit_review → idle");
    }

    /// Store transcribed / edited text while remaining in EditReview.
    pub fn set_edit_buffer(&mut self, text: String) {
        self.edit_buffer = text;
    }

    pub fn status_snapshot(&self) -> Response {
        Response::Status {
            state: self.state,
            mode: self.mode,
            error: self.last_error.clone(),
            model: Some(self.model_name.clone()),
            inject_available: self.inject_available,
        }
    }

    pub fn is_capturing(&self) -> bool {
        matches!(self.state, DaemonState::Recording | DaemonState::Paused)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartResult {
    Started,
    Resumed,
    AlreadyRecording,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopResult {
    Stopped,
    EnteredEditReview,
    AlreadyIdle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToggleResult {
    Started,
    Resumed,
    Stopped,
    EnteredEditReview,
    AlreadyRecording,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetModeResult {
    /// Mode stored; no active capture session to retarget.
    Stored { mode: Mode },
    /// Recording/Paused session should switch pipeline behaviour.
    SessionModeChanged { from: Mode, to: Mode },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn st() -> AppState {
        AppState::new(ModelConfig::default())
    }

    #[test]
    fn start_while_recording_is_noop() {
        let mut s = st();
        assert_eq!(s.start(Mode::Live), StartResult::Started);
        assert_eq!(s.start(Mode::Edit), StartResult::AlreadyRecording);
        assert_eq!(s.state, DaemonState::Recording);
        // Start while recording is a true no-op; use `mode edit` to retarget.
        assert_eq!(s.mode, Mode::Live);
    }

    #[test]
    fn live_stop_to_idle_edit_stop_to_review() {
        let mut s = st();
        s.start(Mode::Live);
        assert_eq!(s.stop(), StopResult::Stopped);
        assert_eq!(s.state, DaemonState::Idle);

        s.start(Mode::Edit);
        assert_eq!(s.stop(), StopResult::EnteredEditReview);
        assert_eq!(s.state, DaemonState::EditReview);
    }

    #[test]
    fn pause_resume_and_toggle() {
        let mut s = st();
        s.start(Mode::Live);
        assert!(s.pause());
        assert_eq!(s.state, DaemonState::Paused);
        assert!(s.resume());
        assert_eq!(s.state, DaemonState::Recording);

        assert_eq!(s.toggle(Mode::Live), ToggleResult::Stopped);
        assert_eq!(s.state, DaemonState::Idle);
        assert_eq!(s.toggle(Mode::Edit), ToggleResult::Started);
        assert_eq!(s.mode, Mode::Edit);
    }

    #[test]
    fn finish_edit_review() {
        let mut s = st();
        s.start(Mode::Edit);
        s.stop();
        s.set_edit_buffer("hello".into());
        s.finish_edit_review();
        assert_eq!(s.state, DaemonState::Idle);
        assert!(s.edit_buffer.is_empty());
    }

    #[test]
    fn starting_from_edit_review_abandons_stale_buffer() {
        let mut s = st();
        s.start(Mode::Edit);
        s.stop();
        s.set_edit_buffer("stale review".into());

        assert_eq!(s.start(Mode::Live), StartResult::Started);
        assert_eq!(s.state, DaemonState::Recording);
        assert!(s.edit_buffer.is_empty());
    }
}
