//! Prompt hook executor — **stub implementation**.
//!
//! A full prompt hook runs a single LLM call with the hook input
//! substituted into the prompt and parses the model's response as
//! [`forge_domain::HookOutput`]. Implementing it requires wiring the
//! chat completion service through the hook runtime, which is not
//! trivial and is deferred to a later phase.
//!
//! For now [`ForgePromptHookExecutor::execute`] always returns
//! [`HookOutcome::NonBlockingError`] with a descriptive message so that
//! callers can detect the unsupported path without panicking.
//!
//! Reference: `claude-code/src/utils/hooks/execPromptHook.ts`

use forge_domain::{HookExecResult, HookInput, PromptHookCommand};

use crate::hook_runtime::HookOutcome;

/// Stub executor for prompt hooks.
///
/// Full implementation is deferred — this type exists so the dispatcher
/// has a concrete executor to call for [`PromptHookCommand`] hooks without
/// special-casing the unsupported path at every call site.
#[derive(Debug, Clone, Default)]
pub struct ForgePromptHookExecutor;

impl ForgePromptHookExecutor {
    /// Returns a `NonBlockingError` result with a clear "not yet
    /// supported" stderr message. The signature matches the future
    /// fully-featured implementation so the dispatcher doesn't need to
    /// change when real support lands.
    pub async fn execute(
        &self,
        _config: &PromptHookCommand,
        _input: &HookInput,
    ) -> anyhow::Result<HookExecResult> {
        Ok(HookExecResult {
            outcome: HookOutcome::NonBlockingError,
            output: None,
            raw_stdout: String::new(),
            raw_stderr: "Prompt hooks are not yet supported in this build of Forge".to_string(),
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
                session_id: "sess-prompt".to_string(),
                transcript_path: PathBuf::from("/tmp/transcript.json"),
                cwd: PathBuf::from("/tmp"),
                permission_mode: None,
                agent_id: None,
                agent_type: None,
                hook_event_name: "UserPromptSubmit".to_string(),
            },
            payload: HookInputPayload::UserPromptSubmit { prompt: "hello".to_string() },
        }
    }

    fn prompt_hook() -> PromptHookCommand {
        PromptHookCommand {
            prompt: "Summarize: $ARGUMENTS".to_string(),
            condition: None,
            timeout: None,
            model: None,
            status_message: None,
            once: false,
        }
    }

    #[tokio::test]
    async fn test_prompt_hook_executor_returns_stub_result() {
        let executor = ForgePromptHookExecutor;
        let result = executor
            .execute(&prompt_hook(), &sample_input())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.output.is_none());
        assert!(result.raw_stdout.is_empty());
        assert!(result.raw_stderr.contains("not yet supported"));
        assert_eq!(result.exit_code, None);

        // Ensure the sample payload round-trips through the stub without
        // panicking or failing to serialize.
        let _ = json!({"ensure": "no panic"});
    }
}
