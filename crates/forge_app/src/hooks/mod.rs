mod compaction;
mod doom_loop;
mod pending_todos;
// Phase 4 Part 2a: `PluginHookHandler` is now wired into the Hook builder
// in `ForgeApp::chat`, so the module no longer needs `#[allow(dead_code)]`.
// Part 2b will add the matching fire sites in `Orchestrator::run` that
// actually trigger each T1 lifecycle event.
pub mod plugin;
mod skill_listing;
mod title_generation;
mod tracing;

pub use compaction::CompactionHandler;
pub use doom_loop::DoomLoopDetector;
pub use pending_todos::PendingTodosHandler;
pub use plugin::PluginHookHandler;
// Only the two lifecycle hooks themselves are re-exported at crate level.
// Internal helpers (`format_invocables_within_budget`, `build_skill_reminder`,
// `DEFAULT_BUDGET_FRACTION`, `DEFAULT_CONTEXT_TOKENS`) stay private to the
// `skill_listing` module and are only reachable through
// `crate::hooks::skill_listing::*` if a future caller genuinely needs them.
// This keeps the public surface area minimal and avoids `unused_imports`
// warnings for symbols nothing outside the module consumes today.
pub use skill_listing::{SkillCacheInvalidator, SkillListingHandler};
pub use title_generation::TitleGenerationHandler;
pub use tracing::TracingHandler;
