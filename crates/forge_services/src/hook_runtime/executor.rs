//! Top-level hook executor — fans [`forge_app::HookExecutorInfra`] method
//! calls out to the four per-kind executors (`shell`, `http`, `prompt`,
//! `agent`).
//!
//! The dispatcher ([`forge_app::hooks::plugin::PluginHookHandler`]) never
//! touches the per-kind executors directly. It holds a single
//! `HookExecutorInfra` trait object and calls `execute_shell` /
//! `execute_http` / `execute_prompt` / `execute_agent` based on the
//! [`forge_domain::HookCommand`] variant that came out of the merged
//! config. This file is the glue that makes that dispatch work.

use std::collections::HashMap;

use async_trait::async_trait;
use forge_app::{EnvironmentInfra, HookExecutorInfra};
use forge_domain::{
    AgentHookCommand, HookExecResult, HookInput, HttpHookCommand, PromptHookCommand,
    ShellHookCommand,
};

use crate::hook_runtime::agent::ForgeAgentHookExecutor;
use crate::hook_runtime::http::{ForgeHttpHookExecutor, map_env_lookup};
use crate::hook_runtime::prompt::ForgePromptHookExecutor;
use crate::hook_runtime::shell::ForgeShellHookExecutor;

/// Concrete implementation of [`HookExecutorInfra`].
///
/// Generic over the environment infrastructure `F` so the HTTP executor
/// can use `F::get_env_var` for header substitution. The three other
/// executors are parameter-free and held as plain values.
///
/// The struct is cheap to clone — every field is either `Copy` or a
/// small handle.
#[derive(Debug, Clone)]
pub struct ForgeHookExecutor<F> {
    infra: std::sync::Arc<F>,
    shell: ForgeShellHookExecutor,
    http: ForgeHttpHookExecutor,
    prompt: ForgePromptHookExecutor,
    agent: ForgeAgentHookExecutor,
}

impl<F> ForgeHookExecutor<F> {
    /// Creates a new executor with all four per-kind executors in their
    /// default configuration.
    pub fn new(infra: std::sync::Arc<F>) -> Self {
        Self {
            infra,
            shell: ForgeShellHookExecutor::default(),
            http: ForgeHttpHookExecutor::default(),
            prompt: ForgePromptHookExecutor,
            agent: ForgeAgentHookExecutor,
        }
    }
}

#[async_trait]
impl<F> HookExecutorInfra for ForgeHookExecutor<F>
where
    F: EnvironmentInfra + Send + Sync + 'static,
{
    async fn execute_shell(
        &self,
        config: &ShellHookCommand,
        input: &HookInput,
        env_vars: HashMap<String, String>,
    ) -> anyhow::Result<HookExecResult> {
        self.shell.execute(config, input, env_vars).await
    }

    async fn execute_http(
        &self,
        config: &HttpHookCommand,
        input: &HookInput,
    ) -> anyhow::Result<HookExecResult> {
        // Build a lookup that calls into `EnvironmentInfra::get_env_var`.
        // Using a `HashMap` snapshot of the allow-listed names keeps the
        // closure `'static` and `Send`/`Sync` for free, which matters
        // because the HTTP executor is called from inside the dispatcher
        // across thread boundaries.
        let mut snapshot = HashMap::new();
        if let Some(allowed) = config.allowed_env_vars.as_ref() {
            for name in allowed {
                if let Some(value) = self.infra.get_env_var(name) {
                    snapshot.insert(name.clone(), value);
                }
            }
        }
        let lookup = map_env_lookup(snapshot);
        self.http.execute(config, input, lookup).await
    }

    async fn execute_prompt(
        &self,
        config: &PromptHookCommand,
        input: &HookInput,
    ) -> anyhow::Result<HookExecResult> {
        self.prompt.execute(config, input).await
    }

    async fn execute_agent(
        &self,
        config: &AgentHookCommand,
        input: &HookInput,
    ) -> anyhow::Result<HookExecResult> {
        self.agent.execute(config, input).await
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use fake::{Fake, Faker};
    use forge_domain::{Environment, HookInputBase, HookInputPayload, HookOutcome};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    /// Tiny environment stub that satisfies `EnvironmentInfra` for the
    /// executor-wiring tests in this module. We only care that the
    /// trait object constructs and that each dispatch path routes to
    /// the correct per-kind executor — the real implementations have
    /// their own unit tests.
    #[derive(Clone)]
    struct StubInfra;

    impl EnvironmentInfra for StubInfra {
        type Config = forge_config::ForgeConfig;

        fn get_environment(&self) -> Environment {
            Faker.fake()
        }

        fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
            std::collections::BTreeMap::new()
        }
    }

    fn sample_input() -> HookInput {
        HookInput {
            base: HookInputBase {
                session_id: "sess".to_string(),
                transcript_path: PathBuf::from("/tmp/t.json"),
                cwd: PathBuf::from("/tmp"),
                permission_mode: None,
                agent_id: None,
                agent_type: None,
                hook_event_name: "PreToolUse".to_string(),
            },
            payload: HookInputPayload::PreToolUse {
                tool_name: "Bash".to_string(),
                tool_input: json!({}),
                tool_use_id: "toolu_1".to_string(),
            },
        }
    }

    #[tokio::test]
    async fn test_prompt_hook_routes_through_stub_executor() {
        let infra = Arc::new(StubInfra);
        let exec = ForgeHookExecutor::new(infra);
        let config = PromptHookCommand {
            prompt: "hi".to_string(),
            condition: None,
            timeout: None,
            model: None,
            status_message: None,
            once: false,
        };
        let result = exec.execute_prompt(&config, &sample_input()).await.unwrap();
        // The stub always returns NonBlockingError with "not yet supported".
        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.raw_stderr.contains("not yet supported"));
    }

    #[tokio::test]
    async fn test_agent_hook_routes_through_stub_executor() {
        let infra = Arc::new(StubInfra);
        let exec = ForgeHookExecutor::new(infra);
        let config = AgentHookCommand {
            prompt: "verify".to_string(),
            condition: None,
            timeout: None,
            model: None,
            status_message: None,
            once: false,
        };
        let result = exec.execute_agent(&config, &sample_input()).await.unwrap();
        assert_eq!(result.outcome, HookOutcome::NonBlockingError);
        assert!(result.raw_stderr.contains("not yet supported"));
    }
}
