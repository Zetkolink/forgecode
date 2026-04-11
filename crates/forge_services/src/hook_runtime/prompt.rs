//! Prompt hook executor — single LLM call evaluation.
//!
//! A prompt hook sends one LLM request with a hardcoded system prompt,
//! the hook's prompt text (with `$ARGUMENTS` substituted), and parses
//! the model's `{ "ok": true }` / `{ "ok": false, "reason": "..." }`
//! response to decide whether to allow or block the action.
//!
//! Reference: `claude-code/src/utils/hooks/execPromptHook.ts`

use forge_app::HookExecutorInfra;
use forge_domain::{
    Context, ContextMessage, HookDecision, HookExecResult, HookInput, HookOutput, ModelId,
    PromptHookCommand, ResponseFormat, SyncHookOutput,
};

use crate::hook_runtime::HookOutcome;

/// Default model for prompt hooks when the config doesn't specify one.
/// Matches Claude Code's `getSmallFastModel()`.
const DEFAULT_PROMPT_HOOK_MODEL: &str = "claude-3-5-haiku-20241022";

/// Default timeout for prompt hooks in seconds.
const DEFAULT_PROMPT_HOOK_TIMEOUT_SECS: u64 = 30;

/// System prompt for evaluating hook conditions via LLM.
/// Exact match of Claude Code's `execPromptHook.ts:65-69`.
const HOOK_EVALUATION_SYSTEM_PROMPT: &str = r#"You are evaluating a hook in Claude Code.

Your response must be a JSON object matching one of the following schemas:
1. If the condition is met, return: {"ok": true}
2. If the condition is not met, return: {"ok": false, "reason": "Reason for why it is not met"}"#;

/// Executor for prompt hooks.
///
/// Uses a single LLM call to evaluate whether a hook condition is met.
/// The model receives the hook prompt (with `$ARGUMENTS` substituted)
/// and must respond with `{"ok": true}` or `{"ok": false, "reason": "..."}`.
#[derive(Debug, Clone, Default)]
pub struct ForgePromptHookExecutor;

/// JSON schema for the hook response: `{ "ok": bool, "reason"?: string }`.
fn hook_response_schema() -> schemars::Schema {
    schemars::json_schema!({
        "type": "object",
        "properties": {
            "ok": { "type": "boolean" },
            "reason": { "type": "string" }
        },
        "required": ["ok"],
        "additionalProperties": false
    })
}

/// Replace `$ARGUMENTS` in the prompt text with the JSON-serialized
/// hook input. Matches Claude Code's `addArgumentsToPrompt()` from
/// `claude-code/src/utils/hooks/hookHelpers.ts:6-30`.
fn substitute_arguments(prompt: &str, input: &HookInput) -> String {
    if !prompt.contains("$ARGUMENTS") {
        return prompt.to_string();
    }
    // Serialize the full input as JSON for substitution.
    let json_input = serde_json::to_string(input).unwrap_or_default();
    prompt.replace("$ARGUMENTS", &json_input)
}

/// Parsed model response.
#[derive(serde::Deserialize)]
struct HookResponse {
    ok: bool,
    reason: Option<String>,
}

impl ForgePromptHookExecutor {
    /// Execute a prompt hook by making a single LLM call.
    ///
    /// # Arguments
    /// - `config` — The prompt hook configuration (prompt text, model override,
    ///   timeout).
    /// - `input` — The hook input payload (tool name, args, etc.).
    /// - `executor` — The executor infra providing `query_model_for_hook`.
    pub async fn execute(
        &self,
        config: &PromptHookCommand,
        input: &HookInput,
        executor: &dyn HookExecutorInfra,
    ) -> anyhow::Result<HookExecResult> {
        // 1. Substitute $ARGUMENTS in the prompt text.
        let processed_prompt = substitute_arguments(&config.prompt, input);

        // 2. Determine the model to use.
        let model_id = ModelId::new(config.model.as_deref().unwrap_or(DEFAULT_PROMPT_HOOK_MODEL));

        // 3. Build the LLM context.
        let context = Context::default()
            .add_message(ContextMessage::system(
                HOOK_EVALUATION_SYSTEM_PROMPT.to_string(),
            ))
            .add_message(ContextMessage::user(
                processed_prompt.clone(),
                Some(model_id.clone()),
            ))
            .response_format(ResponseFormat::JsonSchema(Box::new(hook_response_schema())));

        // 4. Apply timeout.
        let timeout_secs = config.timeout.unwrap_or(DEFAULT_PROMPT_HOOK_TIMEOUT_SECS);
        let timeout_duration = std::time::Duration::from_secs(timeout_secs);

        // 5. Make the LLM call with timeout.
        let llm_result = tokio::time::timeout(
            timeout_duration,
            executor.query_model_for_hook(&model_id, context),
        )
        .await;

        match llm_result {
            // Timeout — cancelled outcome.
            Err(_elapsed) => {
                tracing::warn!(
                    prompt = %config.prompt,
                    timeout_secs,
                    "Prompt hook timed out"
                );
                Ok(HookExecResult {
                    outcome: HookOutcome::Cancelled,
                    output: None,
                    raw_stdout: String::new(),
                    raw_stderr: format!("Prompt hook timed out after {}s", timeout_secs),
                    exit_code: None,
                })
            }
            // LLM call error — non-blocking error.
            Ok(Err(err)) => {
                let err_msg = format!("Error executing prompt hook: {err}");
                tracing::warn!(
                    prompt = %config.prompt,
                    error = %err,
                    "Prompt hook LLM call failed"
                );
                Ok(HookExecResult {
                    outcome: HookOutcome::NonBlockingError,
                    output: None,
                    raw_stdout: String::new(),
                    raw_stderr: err_msg,
                    exit_code: Some(1),
                })
            }
            // LLM call succeeded — parse the response.
            Ok(Ok(response_text)) => {
                let trimmed = response_text.trim();
                tracing::debug!(
                    prompt = %config.prompt,
                    response = %trimmed,
                    "Prompt hook model response"
                );

                // Try to parse the JSON response.
                let parsed: Result<HookResponse, _> = serde_json::from_str(trimmed);
                match parsed {
                    Err(parse_err) => {
                        tracing::warn!(
                            response = %trimmed,
                            error = %parse_err,
                            "Prompt hook response is not valid JSON"
                        );
                        Ok(HookExecResult {
                            outcome: HookOutcome::NonBlockingError,
                            output: None,
                            raw_stdout: trimmed.to_string(),
                            raw_stderr: format!("JSON validation failed: {parse_err}"),
                            exit_code: Some(1),
                        })
                    }
                    Ok(hook_resp) if hook_resp.ok => {
                        // Condition was met — success.
                        tracing::debug!(prompt = %config.prompt, "Prompt hook condition was met");
                        Ok(HookExecResult {
                            outcome: HookOutcome::Success,
                            output: Some(HookOutput::Sync(SyncHookOutput {
                                should_continue: Some(true),
                                ..Default::default()
                            })),
                            raw_stdout: trimmed.to_string(),
                            raw_stderr: String::new(),
                            exit_code: Some(0),
                        })
                    }
                    Ok(hook_resp) => {
                        // Condition was not met — blocking.
                        let reason = hook_resp.reason.unwrap_or_default();
                        tracing::info!(
                            prompt = %config.prompt,
                            reason = %reason,
                            "Prompt hook condition was not met"
                        );
                        let output = HookOutput::Sync(SyncHookOutput {
                            should_continue: Some(false),
                            decision: Some(HookDecision::Block),
                            reason: Some(format!("Prompt hook condition was not met: {reason}")),
                            ..Default::default()
                        });
                        Ok(HookExecResult {
                            outcome: HookOutcome::Blocking,
                            output: Some(output),
                            raw_stdout: trimmed.to_string(),
                            raw_stderr: String::new(),
                            exit_code: Some(1),
                        })
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use forge_domain::{HookInputBase, HookInputPayload};
    use pretty_assertions::assert_eq;

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

    /// Mock executor that records the query and returns a canned response.
    struct MockExecutor {
        response: Mutex<String>,
        captured_model: Mutex<Option<String>>,
    }

    impl MockExecutor {
        fn with_response(response: &str) -> Self {
            Self {
                response: Mutex::new(response.to_string()),
                captured_model: Mutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl HookExecutorInfra for MockExecutor {
        async fn execute_shell(
            &self,
            _: &forge_domain::ShellHookCommand,
            _: &HookInput,
            _: std::collections::HashMap<String, String>,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_http(
            &self,
            _: &forge_domain::HttpHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_prompt(
            &self,
            _: &PromptHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_agent(
            &self,
            _: &forge_domain::AgentHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }

        async fn query_model_for_hook(
            &self,
            model_id: &ModelId,
            _context: Context,
        ) -> anyhow::Result<String> {
            *self.captured_model.lock().unwrap() = Some(model_id.as_str().to_string());
            Ok(self.response.lock().unwrap().clone())
        }
    }

    /// Mock that simulates an LLM error.
    struct ErrorExecutor;

    #[async_trait::async_trait]
    impl HookExecutorInfra for ErrorExecutor {
        async fn execute_shell(
            &self,
            _: &forge_domain::ShellHookCommand,
            _: &HookInput,
            _: std::collections::HashMap<String, String>,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_http(
            &self,
            _: &forge_domain::HttpHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_prompt(
            &self,
            _: &PromptHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_agent(
            &self,
            _: &forge_domain::AgentHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }

        async fn query_model_for_hook(
            &self,
            _model_id: &ModelId,
            _context: Context,
        ) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("provider connection refused"))
        }
    }

    /// Mock that hangs forever (for timeout tests).
    struct HangingExecutor;

    #[async_trait::async_trait]
    impl HookExecutorInfra for HangingExecutor {
        async fn execute_shell(
            &self,
            _: &forge_domain::ShellHookCommand,
            _: &HookInput,
            _: std::collections::HashMap<String, String>,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_http(
            &self,
            _: &forge_domain::HttpHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_prompt(
            &self,
            _: &PromptHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }
        async fn execute_agent(
            &self,
            _: &forge_domain::AgentHookCommand,
            _: &HookInput,
        ) -> anyhow::Result<HookExecResult> {
            unimplemented!()
        }

        async fn query_model_for_hook(
            &self,
            _model_id: &ModelId,
            _context: Context,
        ) -> anyhow::Result<String> {
            // Hang forever — let the timeout kick in.
            std::future::pending().await
        }
    }

    #[test]
    fn test_substitute_arguments_replaces_placeholder() {
        let input = sample_input();
        let result = substitute_arguments("Check: $ARGUMENTS", &input);
        assert!(result.contains("UserPromptSubmit"));
        assert!(result.contains("hello"));
        assert!(!result.contains("$ARGUMENTS"));
    }

    #[test]
    fn test_substitute_arguments_no_placeholder() {
        let input = sample_input();
        let result = substitute_arguments("Just a plain prompt", &input);
        assert_eq!(result, "Just a plain prompt");
    }

    #[tokio::test]
    async fn test_prompt_hook_ok_true() {
        let executor = MockExecutor::with_response(r#"{"ok": true}"#);
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Success);
        assert!(result.output.is_some());
        assert_eq!(result.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_prompt_hook_ok_false_with_reason() {
        let executor =
            MockExecutor::with_response(r#"{"ok": false, "reason": "Tests are failing"}"#);
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Blocking);
        assert_eq!(result.exit_code, Some(1));
        if let Some(HookOutput::Sync(sync)) = &result.output {
            assert_eq!(sync.should_continue, Some(false));
            assert!(sync.reason.as_ref().unwrap().contains("Tests are failing"));
        } else {
            panic!("Expected Sync output");
        }
    }

    #[tokio::test]
    async fn test_prompt_hook_invalid_json_response() {
        let executor = MockExecutor::with_response("not valid json at all");
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.raw_stderr.contains("JSON validation failed"));
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_prompt_hook_llm_error() {
        let executor = ErrorExecutor;
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.raw_stderr.contains("Error executing prompt hook"));
        assert!(result.raw_stderr.contains("connection refused"));
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_prompt_hook_timeout() {
        let executor = HangingExecutor;
        let prompt_executor = ForgePromptHookExecutor;
        let mut hook = prompt_hook();
        hook.timeout = Some(1); // 1 second timeout

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Cancelled);
        assert!(result.raw_stderr.contains("timed out"));
    }

    #[tokio::test]
    async fn test_prompt_hook_custom_model() {
        let executor = Arc::new(MockExecutor::with_response(r#"{"ok": true}"#));
        let prompt_executor = ForgePromptHookExecutor;
        let mut hook = prompt_hook();
        hook.model = Some("claude-3-opus-20240229".to_string());

        prompt_executor
            .execute(&hook, &sample_input(), executor.as_ref())
            .await
            .unwrap();

        assert_eq!(
            *executor.captured_model.lock().unwrap(),
            Some("claude-3-opus-20240229".to_string())
        );
    }

    #[tokio::test]
    async fn test_prompt_hook_default_model() {
        let executor = Arc::new(MockExecutor::with_response(r#"{"ok": true}"#));
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        prompt_executor
            .execute(&hook, &sample_input(), executor.as_ref())
            .await
            .unwrap();

        assert_eq!(
            *executor.captured_model.lock().unwrap(),
            Some(DEFAULT_PROMPT_HOOK_MODEL.to_string())
        );
    }

    #[tokio::test]
    async fn test_prompt_hook_ok_false_without_reason() {
        let executor = MockExecutor::with_response(r#"{"ok": false}"#);
        let prompt_executor = ForgePromptHookExecutor;
        let hook = prompt_hook();

        let result = prompt_executor
            .execute(&hook, &sample_input(), &executor)
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Blocking);
        if let Some(HookOutput::Sync(sync)) = &result.output {
            assert!(sync.reason.as_ref().unwrap().contains("not met"));
        }
    }

    #[test]
    fn test_hook_response_schema_is_valid() {
        // Ensure the schema is valid JSON Schema.
        let schema = hook_response_schema();
        let json = serde_json::to_value(schema).unwrap();
        assert_eq!(json["type"], "object");
        assert!(json["properties"]["ok"]["type"] == "boolean");
    }
}
