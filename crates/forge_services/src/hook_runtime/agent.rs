//! Agent hook executor — **stub implementation**.
//!
//! A full agent hook spawns a sub-agent loop (multiple turns, tool
//! invocations, etc.) for agentic verification scenarios like "Verify
//! that the tests pass before continuing". Wiring this up requires
//! access to the full agent executor which is not straightforward from
//! the hook runtime layer.
//!
//! For now [`ForgeAgentHookExecutor::execute`] returns a stub
//! `NonBlockingError` result with a descriptive message so the
//! dispatcher can call it uniformly without special-casing.
//!
//! Reference: `claude-code/src/schemas/hooks.ts:128-163`

use forge_domain::{AgentHookCommand, HookExecResult, HookInput};

use crate::hook_runtime::HookOutcome;

/// Stub executor for sub-agent hooks.
///
/// Full implementation is deferred — this exists so [`HookCommand::Agent`]
/// can be dispatched without branching at every call site.
#[derive(Debug, Clone, Default)]
pub struct ForgeAgentHookExecutor;

impl ForgeAgentHookExecutor {
    /// Create a new stub executor.
    pub fn new() -> Self {
        Self
    }

    /// Returns a `NonBlockingError` result with a "not yet supported"
    /// stderr message. Signature matches the future fully-featured
    /// implementation so the dispatcher contract is stable.
    pub async fn execute(
        &self,
        _config: &AgentHookCommand,
        _input: &HookInput,
    ) -> anyhow::Result<HookExecResult> {
        Ok(HookExecResult {
            outcome: HookOutcome::NonBlockingError,
            output: None,
            raw_stdout: String::new(),
            raw_stderr: "Agent hooks are not yet supported in this build of Forge".to_string(),
            exit_code: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use forge_domain::{HookInputBase, HookInputPayload};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn sample_input() -> HookInput {
        HookInput {
            base: HookInputBase {
                session_id: "sess-agent".to_string(),
                transcript_path: PathBuf::from("/tmp/transcript.json"),
                cwd: PathBuf::from("/tmp"),
                permission_mode: None,
                agent_id: None,
                agent_type: None,
                hook_event_name: "PreToolUse".to_string(),
            },
            payload: HookInputPayload::PreToolUse {
                tool_name: "Bash".to_string(),
                tool_input: json!({"command": "cargo test"}),
                tool_use_id: "toolu_agent".to_string(),
            },
        }
    }

    fn agent_hook() -> AgentHookCommand {
        AgentHookCommand {
            prompt: "Verify tests pass".to_string(),
            condition: None,
            timeout: None,
            model: None,
            status_message: None,
            once: false,
        }
    }

    #[tokio::test]
    async fn test_agent_hook_executor_returns_stub_result() {
        let executor = ForgeAgentHookExecutor::new();
        let result = executor
            .execute(&agent_hook(), &sample_input())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.output.is_none());
        assert!(result.raw_stdout.is_empty());
        assert!(result.raw_stderr.contains("not yet supported"));
        assert_eq!(result.exit_code, None);
    }
}
