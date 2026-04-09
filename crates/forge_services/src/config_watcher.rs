//! Configuration file watcher service.
//!
//! The [`ConfigWatcher`] watches Forge's configuration files and
//! directories (`~/.forge/config.toml`, installed plugins, hooks,
//! skills, …) for on-disk changes, debounces the raw filesystem events,
//! and hands the resulting [`ConfigChange`] values to a user-supplied
//! callback so the orchestrator can fire the
//! [`forge_domain::LifecycleEvent::ConfigChange`] plugin hook.
//!
//! # Phase 6C scope
//!
//! This file currently ships a **minimal skeleton**: the public types,
//! the internal-write suppression map, and [`ConfigWatcher::classify_path`]
//! (used by callers to map a raw filesystem path into a
//! [`forge_domain::ConfigSource`]) are all functional, but the
//! constructor does **not** yet install real file watchers. Wiring
//! [`notify_debouncer_full::new_debouncer`] up to emit real
//! [`ConfigChange`] events — plus the atomic-save grace period for
//! delete+recreate sequences — is tracked as a Phase 6C follow-up.
//!
//! # Design notes
//!
//! - **Internal write suppression.** Every time Forge itself writes a watched
//!   config file it calls [`ConfigWatcher::mark_internal_write`] first. When
//!   the filesystem notification finally arrives, the fire loop (not yet
//!   implemented) will consult [`ConfigWatcher::is_internal_write`] and skip
//!   the event if the timestamp is still within the 5-second suppression
//!   window. This stops Forge from firing its own `ConfigChange` hook for saves
//!   it made itself.
//! - **Debouncing.** Raw `notify` events are noisy — a single `Save` from a
//!   text editor can produce half a dozen create/modify/rename events.
//!   `notify-debouncer-full` coalesces them into a single event per file per
//!   debounce tick.
//! - **Classification.** Plugin hooks filter on the wire string of
//!   [`forge_domain::ConfigSource`] (e.g. `"user_settings"`, `"plugins"`), so
//!   the watcher must know how to translate a raw absolute path back into a
//!   source. [`ConfigWatcher::classify_path`] does that mapping based on
//!   Forge's directory layout.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use forge_domain::ConfigSource;
use notify_debouncer_full::notify::RecommendedWatcher;
use notify_debouncer_full::{Debouncer, RecommendedCache};
use tokio::sync::Mutex;

/// How long after a `mark_internal_write` call the path stays
/// suppressed. Matches Claude Code's 5-second window.
const INTERNAL_WRITE_WINDOW: Duration = Duration::from_secs(5);

/// A debounced configuration change detected by [`ConfigWatcher`].
///
/// This is the value handed to the user-supplied callback registered
/// via [`ConfigWatcher::new`]. The orchestrator wraps it in a
/// [`forge_domain::ConfigChangePayload`] and fires the
/// `ConfigChange` lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigChange {
    /// Which config store changed.
    pub source: ConfigSource,
    /// Absolute path of the file (or directory) that changed.
    pub file_path: PathBuf,
}

/// Service that watches configuration files and directories for
/// changes, debounces the raw events, and suppresses events for paths
/// Forge itself just wrote.
///
/// # Phase 6C status
///
/// The minimal skeleton shipped here keeps the `_debouncer` field
/// permanently `None`. [`ConfigWatcher::new`] accepts a callback so the
/// public API is stable, but the callback is not yet wired to any real
/// filesystem event loop. The only functionality that is live today is
/// the internal-write tracking ([`mark_internal_write`] +
/// [`is_internal_write`]) and the [`classify_path`] helper — both of
/// which are already useful to call sites that need to record their
/// own writes or reason about config paths.
pub struct ConfigWatcher {
    /// Map of paths Forge just wrote → instant the write was recorded.
    /// Consulted by the (future) fire loop to suppress events that
    /// would otherwise re-enter Forge right after its own save.
    recent_internal_writes: Arc<Mutex<HashMap<PathBuf, Instant>>>,

    /// Holds the real debouncer instance once Phase 6C wires the fire
    /// loop. Today this is always `None`; the field exists so the
    /// `Debouncer` is dropped when the watcher is dropped (via
    /// `notify-debouncer-full`'s `Drop` impl), which is the correct
    /// shutdown contract once real watching is enabled.
    _debouncer: Option<Debouncer<RecommendedWatcher, RecommendedCache>>,
}

impl ConfigWatcher {
    /// Create a new [`ConfigWatcher`] with the given callback.
    ///
    /// # Arguments
    ///
    /// - `_callback` — user-supplied closure invoked once per debounced
    ///   [`ConfigChange`] event. Phase 6C accepts the callback to lock the
    ///   public API shape but does not yet invoke it, since the debouncer
    ///   wiring is deferred.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` so the future
    /// debouncer-wiring revision can surface `notify` setup errors
    /// without a breaking API change.
    pub fn new<F>(_callback: F) -> Result<Self>
    where
        F: Fn(ConfigChange) + Send + Sync + 'static,
    {
        // Phase 6C minimal version: the callback is accepted to freeze
        // the public API shape, but no real `notify` watchers are
        // installed yet. See the module-level docs for the follow-up
        // plan that will turn this into a real fire loop.
        Ok(Self {
            recent_internal_writes: Arc::new(Mutex::new(HashMap::new())),
            _debouncer: None,
        })
    }

    /// Record that Forge itself is about to write `path`, so any
    /// filesystem event that arrives within [`INTERNAL_WRITE_WINDOW`]
    /// can be suppressed by the fire loop.
    pub async fn mark_internal_write(&self, path: impl Into<PathBuf>) {
        let mut guard = self.recent_internal_writes.lock().await;
        guard.insert(path.into(), Instant::now());
    }

    /// Returns `true` if `path` was marked as an internal write within
    /// the last [`INTERNAL_WRITE_WINDOW`].
    pub async fn is_internal_write(&self, path: &Path) -> bool {
        let guard = self.recent_internal_writes.lock().await;
        guard
            .get(path)
            .map(|ts| ts.elapsed() < INTERNAL_WRITE_WINDOW)
            .unwrap_or(false)
    }

    /// Classify a filesystem path into a [`ConfigSource`] based on
    /// Forge's directory layout.
    ///
    /// This is a pure function so callers can use it without having to
    /// spin up a full [`ConfigWatcher`]. The mapping rules:
    ///
    /// | Path shape                         | Source           |
    /// |------------------------------------|------------------|
    /// | `…/.forge/local.toml`              | `LocalSettings`  |
    /// | `…/forge/.forge.toml`              | `UserSettings`   |
    /// | `…/.forge/config.toml`             | `ProjectSettings`|
    /// | `…hooks.json`                      | `Hooks`          |
    /// | `…/plugins/…`                      | `Plugins`        |
    /// | `…/skills/…`                       | `Skills`         |
    /// | anything else                      | `None`           |
    ///
    /// Policy settings are intentionally not classified here — the
    /// policy path is OS-specific and must be resolved by the caller
    /// before mapping.
    pub fn classify_path(path: &Path) -> Option<ConfigSource> {
        let s = path.to_string_lossy();
        if s.contains("/.forge/local.toml") || s.ends_with("local.toml") {
            Some(ConfigSource::LocalSettings)
        } else if s.contains("/forge/.forge.toml") || s.ends_with(".forge.toml") {
            Some(ConfigSource::UserSettings)
        } else if s.contains("/.forge/config.toml") {
            Some(ConfigSource::ProjectSettings)
        } else if s.contains("hooks.json") {
            Some(ConfigSource::Hooks)
        } else if s.contains("/plugins/") {
            Some(ConfigSource::Plugins)
        } else if s.contains("/skills/") {
            Some(ConfigSource::Skills)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;

    use super::*;

    // ---- classify_path ----

    #[test]
    fn test_classify_path_user_settings() {
        let path = PathBuf::from("/home/alice/forge/.forge.toml");
        let actual = ConfigWatcher::classify_path(&path);
        assert_eq!(actual, Some(ConfigSource::UserSettings));
    }

    #[test]
    fn test_classify_path_project_settings() {
        let path = PathBuf::from("/work/myproj/.forge/config.toml");
        let actual = ConfigWatcher::classify_path(&path);
        assert_eq!(actual, Some(ConfigSource::ProjectSettings));
    }

    #[test]
    fn test_classify_path_plugin_directory() {
        let path = PathBuf::from("/home/alice/forge/plugins/acme/plugin.toml");
        let actual = ConfigWatcher::classify_path(&path);
        assert_eq!(actual, Some(ConfigSource::Plugins));
    }

    #[test]
    fn test_classify_path_hooks_json() {
        let path = PathBuf::from("/home/alice/forge/hooks.json");
        let actual = ConfigWatcher::classify_path(&path);
        assert_eq!(actual, Some(ConfigSource::Hooks));
    }

    #[test]
    fn test_classify_path_unknown_returns_none() {
        let path = PathBuf::from("/tmp/some/random/file.txt");
        let actual = ConfigWatcher::classify_path(&path);
        assert_eq!(actual, None);
    }

    // ---- internal-write suppression ----

    /// Helper that constructs a minimal `ConfigWatcher` with a no-op
    /// callback so tests can exercise the internal-write API without
    /// depending on the (not-yet-wired) debouncer fire loop.
    fn test_watcher() -> ConfigWatcher {
        ConfigWatcher::new(|_change: ConfigChange| {}).expect("ctor is infallible in Phase 6C")
    }

    #[tokio::test]
    async fn test_mark_internal_write_then_is_internal_write_true() {
        let watcher = test_watcher();
        let path = PathBuf::from("/home/alice/forge/config.toml");

        watcher.mark_internal_write(path.clone()).await;

        assert!(watcher.is_internal_write(&path).await);
    }

    #[tokio::test(start_paused = true)]
    async fn test_is_internal_write_false_after_expiry() {
        // `start_paused = true` lets us advance tokio's mocked clock
        // past INTERNAL_WRITE_WINDOW without actually sleeping, but
        // `Instant::now()` in the internal-write map is the *real*
        // std::time instant — so we assert the natural behaviour by
        // using a tiny window via a freshly inserted stale entry.
        //
        // Workaround: seed the map directly with an Instant in the
        // past so we don't depend on wall-clock sleeping.
        let watcher = test_watcher();
        let path = PathBuf::from("/home/alice/forge/config.toml");

        {
            let mut guard = watcher.recent_internal_writes.lock().await;
            // 10 seconds ago — comfortably outside the 5-second window.
            guard.insert(path.clone(), Instant::now() - Duration::from_secs(10));
        }

        assert!(!watcher.is_internal_write(&path).await);
    }

    #[tokio::test]
    async fn test_is_internal_write_false_for_unknown_path() {
        let watcher = test_watcher();
        let path = PathBuf::from("/never/marked.toml");

        assert!(!watcher.is_internal_write(&path).await);
    }
}
