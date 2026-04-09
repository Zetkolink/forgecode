//! Wave B — Phase 6A/6B lifecycle fire helpers.
//!
//! This module hosts the out-of-orchestrator fire sites for
//! [`NotificationPayload`] and [`SetupPayload`]. Both helpers live in
//! `forge_app` (rather than `forge_services`) because they need direct
//! access to [`crate::hooks::PluginHookHandler`], which is crate-private
//! to `forge_app` through its private `hooks` module.
//!
//! The two entry points are:
//!
//! 1. [`ForgeNotificationService`] — concrete [`NotificationService`]
//!    implementation. Calling [`NotificationService::emit`] fires the
//!    `Notification` lifecycle event through the plugin hook dispatcher
//!    (observability only — hook errors never propagate) and, when the current
//!    stderr is a non-VS-Code TTY, emits a best-effort terminal bell so REPL
//!    users get a passive nudge.
//!
//! 2. [`fire_setup_hook`] — free function used by `ForgeAPI` to fire the
//!    `Setup` lifecycle event when the user invokes `forge --init` / `forge
//!    --maintenance`. Per Claude Code semantics (`hooksConfigManager.ts:175`)
//!    blocking errors from Setup hooks are intentionally discarded; the fire is
//!    observability-only.
//!
//! Both helpers construct a scratch [`Conversation`] because neither is
//! scoped to a live session — the orchestrator lifecycle isn't running
//! when a notification is emitted from the REPL prompt loop, and Setup
//! fires before any conversation has been initialized. The scratch
//! conversation is discarded immediately after the dispatch.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use forge_domain::{
    Agent, ConfigChangePayload, ConfigSource, Conversation, ConversationId, EventData, EventHandle,
    FileChangeEvent, FileChangedPayload, InstructionsLoadedPayload, LoadedInstructions, ModelId,
    NotificationPayload, SetupPayload, SetupTrigger,
};
use notify_debouncer_full::notify::RecursiveMode;
use tracing::{debug, warn};

use crate::hooks::PluginHookHandler;
use crate::services::{Notification, NotificationService, Services};

/// Runtime-settable accessor for the background
/// `FileChangedWatcher` used by the Phase 7C Wave E-2b dynamic
/// `watch_paths` wiring.
///
/// The orchestrator's `SessionStart` fire site needs to push
/// watch-path additions from a hook's
/// [`forge_domain::AggregatedHookResult::watch_paths`] back into the
/// running watcher, but `forge_app` cannot name the concrete
/// `FileChangedWatcherHandle` without creating a dependency cycle
/// (the handle lives in `forge_api`, which already depends on
/// `forge_app`). This trait gives `forge_app` a minimal, concrete-
/// handle-agnostic interface so the two crates fit together.
///
/// Implementations live in `forge_api::file_changed_watcher_handle`
/// and are registered with [`install_file_changed_watcher_ops`]
/// during `ForgeAPI::init`. The orchestrator later calls
/// [`add_file_changed_watch_paths`] from its `SessionStart`
/// aggregator — if no ops have been installed yet (e.g. in a unit
/// test that bypasses `ForgeAPI::init`), the call is a silent no-op.
pub trait FileChangedWatcherOps: Send + Sync {
    /// Install additional runtime watchers over the given paths.
    ///
    /// Implementations are responsible for splitting any pipe-
    /// separated hook matcher strings (e.g. `.envrc|.env`) into
    /// individual entries before calling this method — `watch_paths`
    /// here is expected to already be a flat list of absolute /
    /// cwd-resolved `(PathBuf, RecursiveMode)` pairs.
    fn add_paths(&self, watch_paths: Vec<(PathBuf, RecursiveMode)>);
}

/// Process-wide slot holding the runtime `FileChangedWatcher`
/// accessor. Populated exactly once by `ForgeAPI::init` via
/// [`install_file_changed_watcher_ops`]; read by the orchestrator's
/// `SessionStart` fire site via [`add_file_changed_watch_paths`].
///
/// This deliberately uses [`OnceLock`] rather than plumbing the
/// handle through every layer of the services stack: the watcher is
/// conceptually process-wide (there is one `ForgeAPI` per process),
/// it is installed before any orchestrator run, and the alternative —
/// adding a setter to the `Services` trait — would touch more than
/// a dozen crates for what is essentially a late-binding hook.
/// Mirrors the same pattern used by `ConfigWatcherHandle` in its own
/// `ForgeAPI::init` wiring.
static FILE_CHANGED_WATCHER_OPS: OnceLock<Arc<dyn FileChangedWatcherOps>> = OnceLock::new();

/// Register the live [`FileChangedWatcherOps`] implementation so the
/// orchestrator's `SessionStart` fire site can call
/// [`add_file_changed_watch_paths`] at runtime.
///
/// Called exactly once from `ForgeAPI::init` after
/// [`crate::file_changed_watcher_handle::FileChangedWatcherHandle::spawn`]
/// (in `forge_api`) succeeds. Subsequent calls are a silent no-op
/// because [`OnceLock::set`] returns `Err` on a second write — the
/// process-wide singleton is intentionally immutable.
///
/// # Test-harness behaviour
///
/// Unit tests that construct a `ForgeAPI` without a multi-threaded
/// tokio runtime never reach this installer, which is fine:
/// [`add_file_changed_watch_paths`] is a no-op when nothing has been
/// installed, so tests continue to run without needing to mock the
/// watcher.
pub fn install_file_changed_watcher_ops(ops: Arc<dyn FileChangedWatcherOps>) {
    if FILE_CHANGED_WATCHER_OPS.set(ops).is_err() {
        debug!(
            "install_file_changed_watcher_ops called twice; \
             ignoring the second install (OnceLock is already populated)"
        );
    }
}

/// Push runtime watch-path additions into the installed
/// [`FileChangedWatcherOps`] implementation.
///
/// Called by the orchestrator after a `SessionStart` hook returns
/// `watch_paths` in its [`forge_domain::AggregatedHookResult`]. If
/// no ops have been installed yet (e.g. in unit tests, or when
/// `ForgeAPI::init` degraded to a no-op watcher because no
/// multi-thread tokio runtime was active), this is a silent no-op —
/// dynamic watch_paths are observability-only and losing them is
/// never a correctness bug.
pub fn add_file_changed_watch_paths(watch_paths: Vec<(PathBuf, RecursiveMode)>) {
    if watch_paths.is_empty() {
        return;
    }
    if let Some(ops) = FILE_CHANGED_WATCHER_OPS.get() {
        ops.add_paths(watch_paths);
    } else {
        debug!(
            "add_file_changed_watch_paths called before \
             install_file_changed_watcher_ops — dropping runtime watch paths \
             (expected in unit tests that bypass ForgeAPI::init)"
        );
    }
}

/// Production implementation of [`NotificationService`].
///
/// Cheap to construct — holds only an `Arc<S>` to the services aggregate.
/// Construct one per call from the API layer; there is no persistent
/// state to cache.
pub struct ForgeNotificationService<S> {
    services: Arc<S>,
}

impl<S> ForgeNotificationService<S> {
    /// Create a new service backed by the given [`Services`] handle.
    pub fn new(services: Arc<S>) -> Self {
        Self { services }
    }
}

impl<S: Services> ForgeNotificationService<S> {
    /// Returns `true` if stderr is a TTY and we are **not** inside a VS
    /// Code integrated terminal.
    ///
    /// VS Code's integrated terminal forwards `\x07` as a loud modal
    /// alert, which is exactly the kind of disruption this function
    /// exists to avoid. The detection matches
    /// `crates/forge_main/src/vscode.rs:9-16` verbatim (duplicated here
    /// so `forge_app` does not need to depend on `forge_main`).
    fn should_beep() -> bool {
        if !io::stderr().is_terminal() {
            return false;
        }
        let in_vscode = std::env::var("TERM_PROGRAM")
            .map(|v| v == "vscode")
            .unwrap_or(false)
            || std::env::var("VSCODE_PID").is_ok()
            || std::env::var("VSCODE_GIT_ASKPASS_NODE").is_ok()
            || std::env::var("VSCODE_GIT_IPC_HANDLE").is_ok();
        !in_vscode
    }

    /// Best-effort BEL emission to stderr. Swallows IO errors — the bell
    /// is a nice-to-have and should never fail the caller.
    fn emit_bell() {
        let mut err = io::stderr();
        let _ = err.write_all(b"\x07");
        let _ = err.flush();
    }

    /// Look up an [`Agent`] to attach to the hook event. Prefers the
    /// active agent, falling back to the first registered agent when no
    /// active agent is configured. Returns `None` if the registry is
    /// empty — in that case the fire is skipped entirely because the
    /// hook infrastructure requires a non-`None` agent tag on every
    /// event.
    async fn resolve_agent(&self) -> Option<Agent> {
        use crate::services::AgentRegistry;

        // Prefer the active agent so the notification event reflects the
        // agent the user has selected.
        if let Ok(Some(active_id)) = self.services.get_active_agent_id().await
            && let Ok(Some(agent)) = self.services.get_agent(&active_id).await
        {
            return Some(agent);
        }

        // Fall back to any registered agent.
        self.services
            .get_agents()
            .await
            .ok()
            .and_then(|agents| agents.into_iter().next())
    }
}

#[async_trait]
impl<S: Services> NotificationService for ForgeNotificationService<S> {
    async fn emit(&self, notification: Notification) -> anyhow::Result<()> {
        debug!(
            kind = ?notification.kind,
            title = ?notification.title,
            message = %notification.message,
            "emit notification"
        );

        // 1. Fire the Notification hook. Per the trait docs, hook dispatcher errors are
        //    soft failures: log and continue.
        if let Err(err) = self.fire_hook(&notification).await {
            warn!(error = %err, "failed to fire Notification hook");
        }

        // 2. Best-effort terminal bell.
        if Self::should_beep() {
            Self::emit_bell();
        }

        Ok(())
    }
}

impl<S: Services> ForgeNotificationService<S> {
    /// Dispatches the Notification lifecycle event through
    /// [`PluginHookHandler`]. The aggregated result is intentionally
    /// discarded — Notification is an observability-only event per the
    /// trait documentation in `services.rs:538-540`.
    async fn fire_hook(&self, notification: &Notification) -> anyhow::Result<()> {
        let Some(agent) = self.resolve_agent().await else {
            debug!("no agent available — skipping Notification hook fire");
            return Ok(());
        };
        let model_id: ModelId = agent.model.clone();

        let environment = self.services.get_environment();
        // Scratch conversation — Notification fires out-of-band (e.g. on
        // REPL idle) so there is no live Conversation to update. The
        // resulting hook_result is drained and discarded below.
        let mut scratch = Conversation::new(ConversationId::generate());
        let session_id = scratch.id.into_string();
        let transcript_path = environment.transcript_path(&session_id);
        let cwd = environment.cwd.clone();

        let payload = NotificationPayload {
            message: notification.message.clone(),
            title: notification.title.clone(),
            notification_type: notification.kind.as_wire_str().to_string(),
        };

        let event =
            EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

        let plugin_handler = PluginHookHandler::new(self.services.clone());
        <PluginHookHandler<S> as EventHandle<EventData<NotificationPayload>>>::handle(
            &plugin_handler,
            &event,
            &mut scratch,
        )
        .await?;

        // Drain and discard the hook_result — Notification is
        // observability only, blocking_error does not apply.
        let _ = std::mem::take(&mut scratch.hook_result);
        Ok(())
    }
}

/// Fire the `Setup` lifecycle event with the given trigger.
///
/// Used by `ForgeAPI::fire_setup_hook` as the out-of-orchestrator entry
/// point for the `--init` / `--init-only` / `--maintenance` CLI flags.
/// Per Claude Code semantics (`hooksConfigManager.ts:175`) any blocking
/// error returned by a Setup hook is intentionally **discarded** — Setup
/// runs before a conversation exists, so there is nothing to block.
///
/// This function is safe to call even when no plugins are configured:
/// the hook dispatcher returns an empty result which is then drained.
pub async fn fire_setup_hook<S: Services>(
    services: Arc<S>,
    trigger: SetupTrigger,
) -> anyhow::Result<()> {
    use crate::services::AgentRegistry;

    // Resolve an agent for the event context. Setup fires before any
    // conversation has been established, so we use the active agent if
    // set, otherwise the first registered agent. If the registry is
    // empty, skip the fire entirely.
    let agent = if let Ok(Some(active_id)) = services.get_active_agent_id().await {
        match services.get_agent(&active_id).await {
            Ok(Some(agent)) => Some(agent),
            _ => None,
        }
    } else {
        None
    };
    let agent = match agent {
        Some(a) => a,
        None => match services
            .get_agents()
            .await
            .ok()
            .and_then(|a| a.into_iter().next())
        {
            Some(a) => a,
            None => {
                debug!("no agent available — skipping Setup hook fire");
                return Ok(());
            }
        },
    };
    let model_id: ModelId = agent.model.clone();

    let environment = services.get_environment();
    let mut scratch = Conversation::new(ConversationId::generate());
    let session_id = scratch.id.into_string();
    let transcript_path = environment.transcript_path(&session_id);
    let cwd = environment.cwd.clone();

    let payload = SetupPayload { trigger };
    let event = EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

    let plugin_handler = PluginHookHandler::new(services.clone());
    <PluginHookHandler<S> as EventHandle<EventData<SetupPayload>>>::handle(
        &plugin_handler,
        &event,
        &mut scratch,
    )
    .await?;

    // Drain and explicitly ignore the blocking_error per Claude Code
    // semantics (setup hooks cannot block — they run before any
    // conversation exists).
    let aggregated = std::mem::take(&mut scratch.hook_result);
    if let Some(err) = aggregated.blocking_error {
        debug!(
            trigger = ?trigger,
            error = %err.message,
            "Setup hook returned blocking_error; ignoring per Claude Code semantics"
        );
    }

    Ok(())
}

/// Fire the `ConfigChange` lifecycle event for a debounced config
/// file/directory change.
///
/// Used by `ForgeAPI` as the out-of-orchestrator entry point for the
/// `ConfigWatcher` service (Wave C Part 1). The watcher hands us a
/// classified [`ConfigSource`] and absolute `file_path`; we wrap them
/// in a [`ConfigChangePayload`] and dispatch through
/// [`PluginHookHandler`] on a scratch [`Conversation`].
///
/// Per the trait documentation in `services.rs:538-540`, ConfigChange
/// is an observability-only event — hook dispatcher errors are soft
/// failures (logged at `warn!`) and any `blocking_error` on the
/// aggregated result is drained and discarded. Config changes can
/// fire at any time (including from a background watcher thread),
/// long after the triggering conversation is gone, so there is
/// nothing to block.
///
/// This function is safe to call even when no plugins are configured:
/// the hook dispatcher returns an empty result which is then drained.
pub async fn fire_config_change_hook<S: Services>(
    services: Arc<S>,
    source: ConfigSource,
    file_path: Option<PathBuf>,
) {
    use crate::services::AgentRegistry;

    // Resolve an agent for the event context. ConfigChange fires
    // out-of-band (from a background filesystem watcher) so there is
    // no live Conversation bound to an agent — we use the active
    // agent if set, otherwise the first registered agent. If the
    // registry is empty, skip the fire entirely.
    let agent = if let Ok(Some(active_id)) = services.get_active_agent_id().await {
        match services.get_agent(&active_id).await {
            Ok(Some(agent)) => Some(agent),
            _ => None,
        }
    } else {
        None
    };
    let agent = match agent {
        Some(a) => a,
        None => match services
            .get_agents()
            .await
            .ok()
            .and_then(|a| a.into_iter().next())
        {
            Some(a) => a,
            None => {
                debug!("no agent available — skipping ConfigChange hook fire");
                return;
            }
        },
    };
    let model_id: ModelId = agent.model.clone();

    let environment = services.get_environment();
    // Scratch conversation — ConfigChange fires out-of-band from a
    // background watcher thread, so there is no live Conversation to
    // update. The resulting hook_result is drained and discarded
    // below.
    let mut scratch = Conversation::new(ConversationId::generate());
    let session_id = scratch.id.into_string();
    let transcript_path = environment.transcript_path(&session_id);
    let cwd = environment.cwd.clone();

    let payload = ConfigChangePayload { source, file_path };
    let event = EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

    let plugin_handler = PluginHookHandler::new(services.clone());
    if let Err(err) = <PluginHookHandler<S> as EventHandle<EventData<ConfigChangePayload>>>::handle(
        &plugin_handler,
        &event,
        &mut scratch,
    )
    .await
    {
        warn!(
            source = ?source,
            error = %err,
            "failed to dispatch ConfigChange hook; ignoring per Claude Code semantics"
        );
    }

    // Drain and explicitly ignore the blocking_error. ConfigChange is
    // observability-only — the watcher callback runs asynchronously
    // on a background thread with no conversation to block against.
    let aggregated = std::mem::take(&mut scratch.hook_result);
    if let Some(err) = aggregated.blocking_error {
        debug!(
            source = ?source,
            error = %err.message,
            "ConfigChange hook returned blocking_error; ignoring (observability only)"
        );
    }
}

/// Fire the `FileChanged` lifecycle event for a debounced filesystem
/// change under one of the user's watched paths.
///
/// Used by `ForgeAPI` as the out-of-orchestrator entry point for the
/// Phase 7C `FileChangedWatcher` service. The watcher hands us an
/// absolute `file_path` and a [`FileChangeEvent`] discriminator; we
/// wrap them in a [`FileChangedPayload`] and dispatch through
/// [`PluginHookHandler`] on a scratch [`Conversation`].
///
/// Per Claude Code's `FileChanged` semantics, the event is
/// **observability-only** for Wave E-2a — any `blocking_error`
/// returned by a plugin hook is drained and discarded, and dispatch
/// failures are logged at `warn!` but never propagated. Dynamic
/// extension of the watched-paths set based on hook results is
/// deferred to Wave E-2b.
///
/// This function is safe to call even when no plugins are configured:
/// the hook dispatcher returns an empty result which is then drained.
pub async fn fire_file_changed_hook<S: Services>(
    services: Arc<S>,
    file_path: PathBuf,
    event: FileChangeEvent,
) {
    use crate::services::AgentRegistry;

    // Resolve an agent for the event context. FileChanged fires from
    // a background filesystem watcher with no live Conversation bound
    // to an agent — we use the active agent if set, otherwise the
    // first registered agent. If the registry is empty, skip the
    // fire entirely.
    let agent = if let Ok(Some(active_id)) = services.get_active_agent_id().await {
        match services.get_agent(&active_id).await {
            Ok(Some(agent)) => Some(agent),
            _ => None,
        }
    } else {
        None
    };
    let agent = match agent {
        Some(a) => a,
        None => match services
            .get_agents()
            .await
            .ok()
            .and_then(|a| a.into_iter().next())
        {
            Some(a) => a,
            None => {
                debug!("no agent available — skipping FileChanged hook fire");
                return;
            }
        },
    };
    let model_id: ModelId = agent.model.clone();

    let environment = services.get_environment();
    // Scratch conversation — FileChanged fires out-of-band from a
    // background watcher thread, so there is no live Conversation to
    // update. The resulting hook_result is drained and discarded
    // below.
    let mut scratch = Conversation::new(ConversationId::generate());
    let session_id = scratch.id.into_string();
    let transcript_path = environment.transcript_path(&session_id);
    let cwd = environment.cwd.clone();

    let payload = FileChangedPayload { file_path: file_path.clone(), event };
    let event_data =
        EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

    let plugin_handler = PluginHookHandler::new(services.clone());
    if let Err(err) = <PluginHookHandler<S> as EventHandle<EventData<FileChangedPayload>>>::handle(
        &plugin_handler,
        &event_data,
        &mut scratch,
    )
    .await
    {
        warn!(
            path = %file_path.display(),
            event = ?event,
            error = %err,
            "failed to dispatch FileChanged hook; ignoring per Claude Code semantics"
        );
    }

    // Drain and explicitly ignore the blocking_error. FileChanged is
    // observability-only in Wave E-2a — the watcher callback runs
    // asynchronously on a background thread with no conversation to
    // block against. Dynamic watch-path extension based on hook
    // results is deferred to Wave E-2b.
    let aggregated = std::mem::take(&mut scratch.hook_result);
    if let Some(err) = aggregated.blocking_error {
        debug!(
            path = %file_path.display(),
            event = ?event,
            error = %err.message,
            "FileChanged hook returned blocking_error; ignoring (observability only)"
        );
    }
}

/// Fire the `InstructionsLoaded` lifecycle event for a single
/// instructions file that was just loaded into the agent's context.
///
/// Used by `ForgeApp::chat` to dispatch one hook event per AGENTS.md
/// file returned by
/// [`crate::CustomInstructionsService::get_custom_instructions_detailed`]. Pass
/// 1 of Wave D only fires with
/// [`forge_domain::InstructionsLoadReason::SessionStart`]; the nested
/// traversal, conditional-rule, `@include` and post-compact reasons
/// are deferred to Pass 2.
///
/// Per Claude Code semantics, `InstructionsLoaded` is an
/// **observability-only** event — any `blocking_error` returned by a
/// plugin hook is drained and discarded, and dispatch failures are
/// logged at `warn!` but never propagated to the caller. The memory
/// layer cannot veto a load of its own source files.
///
/// This function is safe to call even when no plugins are configured:
/// the hook dispatcher returns an empty result which is then drained.
pub async fn fire_instructions_loaded_hook<S: Services>(
    services: Arc<S>,
    loaded: LoadedInstructions,
) {
    use crate::services::AgentRegistry;

    // Resolve an agent for the event context. InstructionsLoaded fires
    // at session start from the chat pipeline, so we use the active
    // agent if set, otherwise the first registered agent. If the
    // registry is empty, skip the fire entirely — without an agent
    // tag the hook infrastructure cannot build an `EventData`.
    let agent = if let Ok(Some(active_id)) = services.get_active_agent_id().await {
        match services.get_agent(&active_id).await {
            Ok(Some(agent)) => Some(agent),
            _ => None,
        }
    } else {
        None
    };
    let agent = match agent {
        Some(a) => a,
        None => match services
            .get_agents()
            .await
            .ok()
            .and_then(|a| a.into_iter().next())
        {
            Some(a) => a,
            None => {
                debug!("no agent available — skipping InstructionsLoaded hook fire");
                return;
            }
        },
    };
    let model_id: ModelId = agent.model.clone();

    let environment = services.get_environment();
    // Scratch conversation — InstructionsLoaded fires from the chat
    // pipeline *before* the live conversation's orchestrator is
    // running, so we dispatch against a throwaway conversation that
    // gets dropped as soon as the hook call returns.
    let mut scratch = Conversation::new(ConversationId::generate());
    let session_id = scratch.id.into_string();
    let transcript_path = environment.transcript_path(&session_id);
    let cwd = environment.cwd.clone();

    // Project the LoadedInstructions into the wire payload. The
    // payload struct uses the typed enums directly (not strings), so
    // we pass `memory_type` / `load_reason` verbatim.
    let payload = InstructionsLoadedPayload {
        file_path: loaded.file_path,
        memory_type: loaded.memory_type,
        load_reason: loaded.load_reason,
        globs: loaded.globs,
        trigger_file_path: loaded.trigger_file_path,
        parent_file_path: loaded.parent_file_path,
    };

    let event = EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

    let plugin_handler = PluginHookHandler::new(services.clone());
    if let Err(err) =
        <PluginHookHandler<S> as EventHandle<EventData<InstructionsLoadedPayload>>>::handle(
            &plugin_handler,
            &event,
            &mut scratch,
        )
        .await
    {
        warn!(
            error = %err,
            "failed to dispatch InstructionsLoaded hook; ignoring per Claude Code semantics"
        );
    }

    // Drain and explicitly ignore the blocking_error — InstructionsLoaded
    // is observability-only. The memory layer cannot be vetoed by a
    // plugin.
    let aggregated = std::mem::take(&mut scratch.hook_result);
    if let Some(err) = aggregated.blocking_error {
        debug!(
            error = %err.message,
            "InstructionsLoaded hook returned blocking_error; ignoring (observability only)"
        );
    }
}

#[cfg(test)]
mod tests {
    // End-to-end dispatch behaviour for Notification and Setup is already
    // covered by the existing integration tests in
    // `crates/forge_app/src/hooks/plugin.rs`:
    //
    //   - `test_dispatch_notification_matches_notification_type`
    //   - `test_dispatch_setup_matches_trigger_string`
    //
    // Those tests exercise the same `PluginHookHandler` dispatcher that
    // `ForgeNotificationService` and `fire_setup_hook` call into, so we
    // rely on them for correctness.
    //
    // Unit tests for `should_beep` are intentionally omitted: the
    // detection reads env vars, which cannot be safely toggled from a
    // parallel test runner without serializing test threads. The
    // detection logic is a near-verbatim copy of the already-tested
    // `forge_main::vscode::is_vscode_terminal` function
    // (see `crates/forge_main/src/vscode.rs:86-110`).
}
