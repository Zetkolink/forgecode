//! Shell hook executor — runs a `ShellHookCommand` as a subprocess.
//!
//! Implements the wire protocol described in
//! `claude-code/src/utils/hooks.ts:747-1335`:
//!
//! 1. Serialize the [`HookInput`] to JSON (snake_case fields matching the
//!    Claude Code wire format exactly).
//! 2. Spawn `bash -c <command>` (or `powershell -Command <command>` on Windows,
//!    if the config requests it).
//! 3. Write the JSON + a trailing `\n` to stdin. The newline is **critical** —
//!    shell hooks that use `read -r` patterns rely on it to complete their read
//!    loop.
//! 4. Close stdin immediately so the hook can exit without a partial read.
//! 5. Wait for the child with a timeout. Default timeout is 30 seconds to match
//!    Claude Code's `TOOL_HOOK_EXECUTION_TIMEOUT_MS`.
//! 6. Attempt to parse stdout as a [`HookOutput`] JSON document; fall back to
//!    treating the output as plain text when parsing fails.
//! 7. Classify the outcome using the JSON `decision` field when present,
//!    otherwise the raw exit code.
//!
//! The executor is stateless: `async` / `asyncRewake` modes and `once`
//! semantics are added in Phase 3 Part 3 once the dispatcher exists.

use std::collections::HashMap;
use std::time::Duration;

use forge_app::{HookExecResult, HookOutcome};
use forge_domain::{
    HookDecision, HookInput, HookOutput, ShellHookCommand, ShellType, SyncHookOutput,
};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

/// Default timeout when a hook doesn't set its own.
///
/// Matches Claude Code's `TOOL_HOOK_EXECUTION_TIMEOUT_MS = 30000`.
const DEFAULT_HOOK_TIMEOUT: Duration = Duration::from_secs(30);

/// Executes [`ShellHookCommand`] hooks.
///
/// This is the Part 2 shell-only executor. Part 3 adds HTTP, prompt,
/// and agent support behind the same [`forge_app::HookExecutorInfra`]
/// trait.
#[derive(Debug, Clone)]
pub struct ForgeShellHookExecutor {
    default_timeout: Duration,
}

impl Default for ForgeShellHookExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgeShellHookExecutor {
    /// Create a new shell executor using the default 30-second timeout.
    pub fn new() -> Self {
        Self { default_timeout: DEFAULT_HOOK_TIMEOUT }
    }

    /// Create a shell executor with a custom default timeout (used in
    /// tests to avoid sleeping for 30 s on the timeout path).
    #[cfg(test)]
    pub fn with_default_timeout(default_timeout: Duration) -> Self {
        Self { default_timeout }
    }

    /// Run `config` with `input` piped to stdin.
    ///
    /// `env_vars` are layered on top of the inherited parent environment.
    /// Variable substitution (`${FORGE_PLUGIN_ROOT}` etc.) is applied to
    /// `config.command` before spawning.
    pub async fn execute(
        &self,
        config: &ShellHookCommand,
        input: &HookInput,
        env_vars: HashMap<String, String>,
    ) -> anyhow::Result<HookExecResult> {
        // 1. Serialize the input.
        let input_json = serde_json::to_string(input)?;

        // 2. Substitute ${VAR} references in the command string.
        let command = substitute_variables(&config.command, &env_vars);

        // 3. Pick shell based on config (default bash on Unix, powershell on Windows is
        //    handled implicitly by the fallback on Windows builds; Part 2 defaults to
        //    bash everywhere because the test suite is gated to unix).
        let (program, shell_flag) = match config.shell {
            Some(ShellType::Powershell) => ("powershell", "-Command"),
            Some(ShellType::Bash) | None => ("bash", "-c"),
        };

        let mut cmd = Command::new(program);
        cmd.arg(shell_flag)
            .arg(&command)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        for (key, val) in &env_vars {
            cmd.env(key, val);
        }

        let mut child = cmd.spawn()?;

        // 4. Write JSON + "\n" to stdin, then drop the handle so the hook sees EOF.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input_json.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            // Dropping `stdin` here closes the pipe.
        }

        // 5. Wait with timeout.
        let timeout_duration = config
            .timeout
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);

        let output = match timeout(timeout_duration, child.wait_with_output()).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!("hook wait failed: {e}"));
            }
            Err(_) => {
                // Child is killed by `kill_on_drop` when we return here.
                return Ok(HookExecResult {
                    outcome: HookOutcome::Cancelled,
                    output: None,
                    raw_stdout: String::new(),
                    raw_stderr: format!("hook timed out after {}s", timeout_duration.as_secs()),
                    exit_code: None,
                });
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code();

        // 6. Try to parse stdout as a HookOutput JSON document.
        let parsed_output = if stdout.trim_start().starts_with('{') {
            serde_json::from_str::<HookOutput>(&stdout).ok()
        } else {
            None
        };

        // 7. Classify the outcome.
        let outcome = classify_outcome(exit_code, parsed_output.as_ref());

        Ok(HookExecResult {
            outcome,
            output: parsed_output,
            raw_stdout: stdout,
            raw_stderr: stderr,
            exit_code,
        })
    }
}

/// Decide the [`HookOutcome`] using (in priority order):
///
/// 1. A parsed [`SyncHookOutput`]'s `decision` field, if `Block`.
/// 2. The raw exit code: `0` → `Success`, `2` → `Blocking`, other non-zero /
///    missing → `NonBlockingError`.
fn classify_outcome(exit_code: Option<i32>, output: Option<&HookOutput>) -> HookOutcome {
    if let Some(HookOutput::Sync(SyncHookOutput { decision: Some(dec), .. })) = output
        && matches!(dec, HookDecision::Block)
    {
        return HookOutcome::Blocking;
    }

    match exit_code {
        Some(0) => HookOutcome::Success,
        Some(2) => HookOutcome::Blocking,
        Some(_) => HookOutcome::NonBlockingError,
        None => HookOutcome::NonBlockingError,
    }
}

/// Substitute `${VAR}` and `${user_config.KEY}` references in a command
/// string using the given environment variable map.
///
/// Only `${VAR}` (braced) references are substituted here — the bare
/// `$VAR` form is left for the shell itself to expand.
///
/// `${user_config.KEY}` is resolved by looking up
/// `FORGE_PLUGIN_OPTION_<KEY>` in `env_vars` (key is upper-cased, hyphens
/// become underscores). This mirrors Claude Code's plugin user-config
/// substitution at `claude-code/src/utils/hooks.ts:822-857`.
///
/// Reference: `claude-code/src/utils/hooks.ts:822-857`.
pub fn substitute_variables(command: &str, env_vars: &HashMap<String, String>) -> String {
    let mut result = command.to_string();

    // Handle ${user_config.KEY} substitutions first so they don't collide
    // with the generic ${VAR} pass below.
    let prefix = "${user_config.";
    while let Some(start) = result.find(prefix) {
        if let Some(rel_end) = result[start..].find('}') {
            let key = &result[start + prefix.len()..start + rel_end];
            let env_key = format!(
                "FORGE_PLUGIN_OPTION_{}",
                key.to_uppercase().replace('-', "_")
            );
            let replacement = env_vars.get(&env_key).map(String::as_str).unwrap_or("");
            result = format!("{}{}{}", &result[..start], replacement, &result[start + rel_end + 1..]);
        } else {
            break;
        }
    }

    // Handle regular ${VAR} substitutions.
    for (key, val) in env_vars {
        let braced = format!("${{{key}}}");
        if result.contains(&braced) {
            result = result.replace(&braced, val);
        }
    }
    result
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use forge_domain::{HookInputBase, HookInputPayload};
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    fn sample_input() -> HookInput {
        HookInput {
            base: HookInputBase {
                session_id: "sess-test".to_string(),
                transcript_path: PathBuf::from("/tmp/transcript.json"),
                cwd: PathBuf::from("/tmp"),
                permission_mode: None,
                agent_id: None,
                agent_type: None,
                hook_event_name: "PreToolUse".to_string(),
            },
            payload: HookInputPayload::PreToolUse {
                tool_name: "Bash".to_string(),
                tool_input: json!({"command": "ls"}),
                tool_use_id: "toolu_test".to_string(),
            },
        }
    }

    fn shell_hook(command: &str) -> ShellHookCommand {
        ShellHookCommand {
            command: command.to_string(),
            condition: None,
            shell: Some(ShellType::Bash),
            timeout: None,
            status_message: None,
            once: false,
            async_mode: false,
            async_rewake: false,
        }
    }

    #[tokio::test]
    async fn test_hook_with_json_stdout_parses_to_hook_output() {
        let executor = ForgeShellHookExecutor::new();
        let config = shell_hook(r#"echo '{"continue": true, "systemMessage": "from hook"}'"#);
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Success);
        assert_eq!(result.exit_code, Some(0));
        assert!(matches!(result.output, Some(HookOutput::Sync(_))));
        match result.output {
            Some(HookOutput::Sync(sync)) => {
                assert_eq!(sync.should_continue, Some(true));
                assert_eq!(sync.system_message.as_deref(), Some("from hook"));
            }
            other => panic!("expected Sync output, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_hook_with_plain_text_stdout_is_success() {
        let executor = ForgeShellHookExecutor::new();
        let config = shell_hook("echo hello world");
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Success);
        assert_eq!(result.exit_code, Some(0));
        assert!(result.output.is_none());
        assert_eq!(result.raw_stdout.trim(), "hello world");
    }

    #[tokio::test]
    async fn test_hook_exit_code_2_is_blocking() {
        let executor = ForgeShellHookExecutor::new();
        let config = shell_hook("echo 'nope' 1>&2; exit 2");
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Blocking);
        assert_eq!(result.exit_code, Some(2));
        assert_eq!(result.raw_stderr.trim(), "nope");
    }

    #[tokio::test]
    async fn test_hook_exit_code_1_is_non_blocking_error() {
        let executor = ForgeShellHookExecutor::new();
        let config = shell_hook("exit 1");
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_hook_stdin_receives_exact_snake_case_json() {
        let temp = TempDir::new().unwrap();
        let captured = temp.path().join("captured.json");
        let executor = ForgeShellHookExecutor::new();

        // The hook writes its stdin contents to a file so the test can
        // inspect them.
        let command = format!("cat > {}", captured.display());
        let config = shell_hook(&command);
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Success);

        let contents = std::fs::read_to_string(&captured).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();

        assert_eq!(parsed["session_id"], "sess-test");
        assert_eq!(parsed["hook_event_name"], "PreToolUse");
        assert_eq!(parsed["tool_name"], "Bash");
        assert_eq!(parsed["tool_use_id"], "toolu_test");
        assert_eq!(parsed["tool_input"]["command"], "ls");
    }

    #[tokio::test]
    async fn test_hook_env_vars_are_set_in_subprocess() {
        let temp = TempDir::new().unwrap();
        let captured = temp.path().join("env.txt");
        let executor = ForgeShellHookExecutor::new();

        let command = format!(
            "printf '%s|%s' \"$FORGE_PROJECT_DIR\" \"$FORGE_SESSION_ID\" > {}",
            captured.display()
        );
        let config = shell_hook(&command);

        let mut env = HashMap::new();
        env.insert("FORGE_PROJECT_DIR".to_string(), "/proj-test".to_string());
        env.insert("FORGE_SESSION_ID".to_string(), "sess-env".to_string());

        executor
            .execute(&config, &sample_input(), env)
            .await
            .unwrap();

        let captured_text = std::fs::read_to_string(&captured).unwrap();
        assert_eq!(captured_text, "/proj-test|sess-env");
    }

    #[tokio::test]
    async fn test_command_substitution_replaces_braced_variable() {
        let temp = TempDir::new().unwrap();
        let captured = temp.path().join("plugin-root.txt");
        let executor = ForgeShellHookExecutor::new();

        // The literal ${FORGE_PLUGIN_ROOT} is substituted by us (not the
        // shell) before spawning.
        let command = format!("echo '${{FORGE_PLUGIN_ROOT}}' > {}", captured.display());
        let config = shell_hook(&command);

        let mut env = HashMap::new();
        env.insert("FORGE_PLUGIN_ROOT".to_string(), "/plugins/demo".to_string());

        executor
            .execute(&config, &sample_input(), env)
            .await
            .unwrap();

        let contents = std::fs::read_to_string(&captured).unwrap();
        assert_eq!(contents.trim(), "/plugins/demo");
    }

    #[tokio::test]
    async fn test_hook_timeout_produces_cancelled() {
        // Use a very short timeout and a long-running hook.
        let executor = ForgeShellHookExecutor::with_default_timeout(Duration::from_millis(100));
        let config = shell_hook("sleep 5");
        let result = executor
            .execute(&config, &sample_input(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, HookOutcome::Cancelled);
        assert!(result.exit_code.is_none());
        assert!(result.raw_stderr.contains("timed out"));
    }

    #[test]
    fn test_substitute_variables_replaces_braced_references() {
        let mut env = HashMap::new();
        env.insert("FORGE_PLUGIN_ROOT".to_string(), "/plugins/x".to_string());
        env.insert("FORGE_SESSION_ID".to_string(), "sess-1".to_string());

        let actual = substitute_variables(
            "run ${FORGE_PLUGIN_ROOT}/bin --session ${FORGE_SESSION_ID}",
            &env,
        );
        assert_eq!(actual, "run /plugins/x/bin --session sess-1");
    }

    #[test]
    fn test_substitute_variables_leaves_unknown_vars_alone() {
        let env = HashMap::new();
        let actual = substitute_variables("echo ${UNKNOWN}", &env);
        assert_eq!(actual, "echo ${UNKNOWN}");
    }

    #[test]
    fn test_classify_outcome_json_block_overrides_exit_zero() {
        let output = HookOutput::Sync(SyncHookOutput {
            decision: Some(HookDecision::Block),
            ..Default::default()
        });
        let outcome = classify_outcome(Some(0), Some(&output));
        assert_eq!(outcome, HookOutcome::Blocking);
    }

    #[test]
    fn test_classify_outcome_exit_0_no_json_is_success() {
        assert_eq!(classify_outcome(Some(0), None), HookOutcome::Success);
    }

    #[test]
    fn test_classify_outcome_exit_2_no_json_is_blocking() {
        assert_eq!(classify_outcome(Some(2), None), HookOutcome::Blocking);
    }

    #[test]
    fn test_classify_outcome_exit_1_no_json_is_non_blocking_error() {
        assert_eq!(
            classify_outcome(Some(1), None),
            HookOutcome::NonBlockingError
        );
    }
}
