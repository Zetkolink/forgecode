//! Forge's implementation of [`ElicitationDispatcher`].
//!
//! Routes MCP server elicitation requests through the plugin hook
//! system first, then falls back to interactive UI when no plugin
//! handles the request.
//!
//! Wave F-1 lands the hook short-circuit path. The interactive UI
//! fallback is Wave F-2 — for now, non-hook-handled requests return
//! [`ElicitationAction::Decline`] so the MCP server gets a well-formed
//! response instead of hanging on a not-yet-implemented prompt.
//!
//! # Why `OnceLock`?
//!
//! `ForgeElicitationDispatcher<S>` needs an `Arc<S>` so it can call
//! `fire_elicitation_hook(self.services.clone(), ...)`, but
//! [`crate::ForgeServices`] owns the dispatcher as a field, which
//! creates a chicken-and-egg cycle (`ForgeServices::new` would need
//! `Arc<Self>` before the `Arc` has been constructed). To break the
//! cycle, the dispatcher stores `OnceLock<Arc<S>>` that is populated
//! via [`ForgeElicitationDispatcher::init`] after the `Arc<S>` exists
//! — typically immediately after `Arc::new(ForgeServices::new(...))`
//! returns at the `forge_api` layer. Until `init` runs, the dispatcher
//! declines all requests (with a warning log) so a bug in the wiring
//! degrades gracefully instead of panicking.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use forge_app::{
    ElicitationAction, ElicitationDispatcher, ElicitationRequest, ElicitationResponse, Services,
    fire_elicitation_hook, fire_elicitation_result_hook,
};
use forge_domain::{AggregatedHookResult, PermissionBehavior};

/// Production [`ElicitationDispatcher`] that fires the `Elicitation`
/// hook and short-circuits on plugin-provided auto-responses, falling
/// back to an interactive UI (Wave F-2) or a hardcoded `Decline`
/// (Wave F-1) when no plugin handles the request.
///
/// The struct-level bound on `S` is intentionally relaxed to
/// `Send + Sync + 'static` (instead of `Services`) so that
/// [`crate::ForgeServices`] can store
/// `Arc<ForgeElicitationDispatcher<ForgeServices<F>>>` as a field
/// without the struct definition requiring
/// `ForgeServices<F>: Services` — the Services impl lives in a
/// separate `impl` block, so demanding it at field-definition time
/// would create a where-clause cycle. The actual `S: Services`
/// requirement is enforced on the [`ElicitationDispatcher`] impl
/// block below, which is the only place that needs to call into
/// `fire_elicitation_hook`.
pub struct ForgeElicitationDispatcher<S: Send + Sync + 'static> {
    /// Late-initialized handle to the Services aggregate. Populated by
    /// [`ForgeElicitationDispatcher::init`] after the outer
    /// `Arc<Services>` exists. Reads use [`OnceLock::get`] and fall
    /// back to Decline with a warn log when the lock is still empty.
    services: OnceLock<Arc<S>>,
}

impl<S: Send + Sync + 'static> ForgeElicitationDispatcher<S> {
    /// Create a dispatcher with an empty services slot. Callers must
    /// invoke [`ForgeElicitationDispatcher::init`] before the first
    /// request arrives — see the module-level docs for the cycle
    /// rationale.
    pub fn new() -> Self {
        Self { services: OnceLock::new() }
    }

    /// Populate the services slot. First call wins — subsequent calls
    /// are silently ignored per the [`OnceLock`] contract. Called
    /// from `forge_api::forge_api.rs` immediately after
    /// `Arc::new(ForgeServices::new(...))` returns so the dispatcher
    /// can fire hooks against the fully-constructed services
    /// aggregate.
    pub fn init(&self, services: Arc<S>) {
        let _ = self.services.set(services);
    }
}

impl<S: Send + Sync + 'static> Default for ForgeElicitationDispatcher<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: Services + 'static> ElicitationDispatcher for ForgeElicitationDispatcher<S> {
    async fn elicit(&self, request: ElicitationRequest) -> ElicitationResponse {
        let Some(services) = self.services.get() else {
            tracing::warn!(
                server = %request.server_name,
                "ForgeElicitationDispatcher::elicit called before init; declining"
            );
            return ElicitationResponse { action: ElicitationAction::Decline, content: None };
        };

        let mode = if request.url.is_some() {
            Some("url".to_string())
        } else {
            None
        };

        // Step 1: fire the `Elicitation` plugin hook.
        let hook_result = fire_elicitation_hook(
            services.clone(),
            request.server_name.clone(),
            request.message.clone(),
            request.requested_schema.clone(),
            mode.clone(),
            request.url.clone(),
        )
        .await;

        // Step 2: inspect hook result for a plugin short-circuit.
        if let Some(response) = resolve_hook_response(&hook_result) {
            match response.action {
                ElicitationAction::Cancel => {
                    tracing::warn!(
                        server = %request.server_name,
                        "elicitation cancelled by plugin hook (blocking_error set)"
                    );
                }
                ElicitationAction::Decline => {
                    tracing::info!(
                        server = %request.server_name,
                        "elicitation auto-declined by plugin hook"
                    );
                }
                ElicitationAction::Accept => {
                    tracing::info!(
                        server = %request.server_name,
                        "elicitation auto-accepted by plugin hook with form data"
                    );
                }
            }
            fire_elicitation_result_hook(
                services.clone(),
                request.server_name.clone(),
                response.action.as_wire_str().to_string(),
                response.content.clone(),
            )
            .await;
            return response;
        }

        // Step 3: no plugin short-circuit — fall back.
        //
        // TODO(wave-f-2-interactive-ui): Wave F-2 replaces this Decline
        // with an interactive form (for form mode) or url-open prompt
        // (for url mode) driven through `forge_select` under
        // `spawn_blocking`. Until then, declining is the safest
        // default: the MCP server gets a well-formed response and the
        // user is not left staring at a hung tool call.
        tracing::warn!(
            server = %request.server_name,
            "no plugin hook handled elicitation and interactive UI fallback is not yet implemented (Wave F-2); declining"
        );

        let response = ElicitationResponse { action: ElicitationAction::Decline, content: None };
        fire_elicitation_result_hook(
            services.clone(),
            request.server_name,
            response.action.as_wire_str().to_string(),
            response.content.clone(),
        )
        .await;
        response
    }
}

/// Pure function that inspects an [`AggregatedHookResult`] for a
/// plugin-provided short-circuit response, returning `Some(response)`
/// when the hook unambiguously dictated an outcome and `None` when the
/// dispatcher should fall through to the interactive UI path.
///
/// Precedence mirrors Claude Code's `hooksConfigManager.ts` semantics:
///
/// 1. `blocking_error` → `Cancel` (highest priority — a blocked event must
///    never progress to an auto-accept path).
/// 2. `permission_behavior == Deny` → `Decline`.
/// 3. `permission_behavior == Allow` + `updated_input` present → `Accept` with
///    the plugin-provided content.
/// 4. `permission_behavior == Allow` without `updated_input` → no short-circuit
///    (plugin said "allow" but provided no form data, so the dispatcher should
///    still prompt the user).
/// 5. `permission_behavior == Ask` → no short-circuit.
/// 6. No permission behavior set → no short-circuit.
///
/// Extracted from [`ForgeElicitationDispatcher::elicit`] so the branch
/// logic can be unit-tested without constructing a full Services
/// mock.
fn resolve_hook_response(hook_result: &AggregatedHookResult) -> Option<ElicitationResponse> {
    if hook_result.blocking_error.is_some() {
        return Some(ElicitationResponse { action: ElicitationAction::Cancel, content: None });
    }

    match hook_result.permission_behavior {
        Some(PermissionBehavior::Deny) => {
            Some(ElicitationResponse { action: ElicitationAction::Decline, content: None })
        }
        Some(PermissionBehavior::Allow) => {
            hook_result
                .updated_input
                .as_ref()
                .map(|content| ElicitationResponse {
                    action: ElicitationAction::Accept,
                    content: Some(content.clone()),
                })
        }
        Some(PermissionBehavior::Ask) | None => None,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`resolve_hook_response`].
    //!
    //! The branch-testing logic was intentionally extracted from
    //! `ForgeElicitationDispatcher::elicit` into a pure function so it
    //! can be unit-tested without building a mock `Services` (the
    //! trait has 28+ associated types, which makes hand-rolling a
    //! mock impractical for a single-wave deliverable). End-to-end
    //! dispatch coverage — including the `init`/Decline path, the
    //! `fire_elicitation_result_hook` fan-out, and the interactive UI
    //! fallback — lands in Wave F-2 alongside the rmcp
    //! `ClientHandler` integration tests, which will have a mock MCP
    //! transport.
    use forge_domain::HookBlockingError;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_resolve_returns_none_for_empty_result() {
        let fixture = AggregatedHookResult::default();
        let actual = resolve_hook_response(&fixture);
        assert!(actual.is_none(), "empty result should fall through to UI");
    }

    #[test]
    fn test_resolve_returns_cancel_on_blocking_error() {
        let mut fixture = AggregatedHookResult::default();
        fixture.blocking_error = Some(HookBlockingError {
            message: "blocked by policy".to_string(),
            command: "test-plugin".to_string(),
        });
        let actual = resolve_hook_response(&fixture).expect("expected short-circuit");
        assert_eq!(actual.action, ElicitationAction::Cancel);
        assert!(actual.content.is_none());
    }

    #[test]
    fn test_resolve_returns_cancel_when_blocking_error_and_allow_both_set() {
        // blocking_error takes precedence over permission_behavior so
        // a blocked event never progresses to an auto-accept path.
        let mut fixture = AggregatedHookResult::default();
        fixture.blocking_error = Some(HookBlockingError {
            message: "blocked by policy".to_string(),
            command: "test-plugin".to_string(),
        });
        fixture.permission_behavior = Some(PermissionBehavior::Allow);
        fixture.updated_input = Some(json!({"user": "alice"}));
        let actual = resolve_hook_response(&fixture).expect("expected short-circuit");
        assert_eq!(actual.action, ElicitationAction::Cancel);
        assert!(actual.content.is_none());
    }

    #[test]
    fn test_resolve_returns_decline_on_deny() {
        let mut fixture = AggregatedHookResult::default();
        fixture.permission_behavior = Some(PermissionBehavior::Deny);
        let actual = resolve_hook_response(&fixture).expect("expected short-circuit");
        assert_eq!(actual.action, ElicitationAction::Decline);
        assert!(actual.content.is_none());
    }

    #[test]
    fn test_resolve_ignores_updated_input_when_denied() {
        // Even if a (misbehaving) plugin set both Deny and
        // updated_input, we should still Decline and never leak the
        // plugin's content into the MCP response.
        let mut fixture = AggregatedHookResult::default();
        fixture.permission_behavior = Some(PermissionBehavior::Deny);
        fixture.updated_input = Some(json!({"user": "alice"}));
        let actual = resolve_hook_response(&fixture).expect("expected short-circuit");
        assert_eq!(actual.action, ElicitationAction::Decline);
        assert!(actual.content.is_none());
    }

    #[test]
    fn test_resolve_returns_accept_on_allow_with_updated_input() {
        let mut fixture = AggregatedHookResult::default();
        fixture.permission_behavior = Some(PermissionBehavior::Allow);
        fixture.updated_input = Some(json!({"user": "alice", "role": "admin"}));
        let actual = resolve_hook_response(&fixture).expect("expected short-circuit");
        assert_eq!(actual.action, ElicitationAction::Accept);
        assert_eq!(
            actual.content,
            Some(json!({"user": "alice", "role": "admin"}))
        );
    }

    #[test]
    fn test_resolve_returns_none_on_allow_without_updated_input() {
        // Allow without form data cannot auto-accept — we need the
        // content payload to return to the MCP server. Fall through
        // to the interactive UI path so the user can fill the form.
        let mut fixture = AggregatedHookResult::default();
        fixture.permission_behavior = Some(PermissionBehavior::Allow);
        let actual = resolve_hook_response(&fixture);
        assert!(
            actual.is_none(),
            "Allow without content should fall through"
        );
    }

    #[test]
    fn test_resolve_returns_none_on_ask() {
        let mut fixture = AggregatedHookResult::default();
        fixture.permission_behavior = Some(PermissionBehavior::Ask);
        let actual = resolve_hook_response(&fixture);
        assert!(
            actual.is_none(),
            "Ask should fall through to interactive UI"
        );
    }

    #[test]
    fn test_as_wire_str_matches_claude_code_vocab() {
        assert_eq!(ElicitationAction::Accept.as_wire_str(), "accept");
        assert_eq!(ElicitationAction::Decline.as_wire_str(), "decline");
        assert_eq!(ElicitationAction::Cancel.as_wire_str(), "cancel");
    }
}
