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
//!    (observability only — hook errors never propagate) and, when the
//!    current stderr is a non-VS-Code TTY, emits a best-effort terminal
//!    bell so REPL users get a passive nudge.
//!
//! 2. [`fire_setup_hook`] — free function used by `ForgeAPI` to fire the
//!    `Setup` lifecycle event when the user invokes
//!    `forge --init` / `forge --maintenance`. Per Claude Code semantics
//!    (`hooksConfigManager.ts:175`) blocking errors from Setup hooks are
//!    intentionally discarded; the fire is observability-only.
//!
//! Both helpers construct a scratch [`Conversation`] because neither is
//! scoped to a live session — the orchestrator lifecycle isn't running
//! when a notification is emitted from the REPL prompt loop, and Setup
//! fires before any conversation has been initialized. The scratch
//! conversation is discarded immediately after the dispatch.

use std::io::{self, IsTerminal, Write};
use std::sync::Arc;

use async_trait::async_trait;
use forge_domain::{
    Agent, Conversation, ConversationId, EventData, EventHandle, ModelId, NotificationPayload,
    SetupPayload, SetupTrigger,
};
use tracing::{debug, warn};

use crate::hooks::PluginHookHandler;
use crate::services::{Notification, NotificationService, Services};

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

        // 1. Fire the Notification hook. Per the trait docs, hook
        //    dispatcher errors are soft failures: log and continue.
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
        None => match services.get_agents().await.ok().and_then(|a| a.into_iter().next()) {
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
    let event =
        EventData::with_context(agent, model_id, session_id, transcript_path, cwd, payload);

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
