//! Hook runtime — the infrastructure for executing hook commands
//! declared in `hooks.json`.
//!
//! This module is split into sub-modules by executor kind plus the
//! dispatch plumbing that wires them together:
//!
//! - [`env`] — builds the `HashMap<String, String>` of `FORGE_*` env vars
//!   injected into every shell hook subprocess.
//! - [`shell`] — the `tokio::process::Command` shell executor.
//! - [`http`] — the HTTP webhook executor (POSTs the input JSON and parses the
//!   response body).
//! - [`prompt`] — stub LLM prompt executor. Full support is deferred.
//! - [`agent`] — stub sub-agent executor. Full support is deferred.
//! - [`config_loader`] — merges `hooks.json` from user/project/plugin sources
//!   into a single [`forge_app::hook_runtime::MergedHooksConfig`] used by the
//!   dispatcher.
//! - [`executor`] — the top-level [`forge_app::HookExecutorInfra`] impl that
//!   fans out to the per-kind executors.
//!
//! `HookExecResult` and `HookOutcome` live in `forge_domain` (not here)
//! so [`forge_domain::AggregatedHookResult::merge`] can consume them
//! without a circular crate dependency. They are re-exported here for
//! convenience so every hook runtime file can `use
//! crate::hook_runtime::HookOutcome;` without pulling in the full
//! `forge_domain::` prefix.
//!
//! The merged-config types (`MergedHooksConfig`, `HookMatcherWithSource`,
//! `HookConfigSource`) live in `forge_app::hook_runtime` so they can be
//! consumed by both the dispatcher (upstream) and the loader
//! (downstream). They are re-exported from this module for backwards
//! compatibility with existing call sites.

pub mod agent;
pub mod config_loader;
pub mod env;
pub mod executor;
pub mod http;
pub mod prompt;
pub mod shell;

pub use config_loader::ForgeHookConfigLoader;
pub use executor::ForgeHookExecutor;
#[allow(unused_imports)]
pub use forge_app::hook_runtime::{
    HookConfigLoaderService, HookConfigSource, HookMatcherWithSource, MergedHooksConfig,
};
#[allow(unused_imports)]
pub use forge_domain::{HookExecResult, HookOutcome};
