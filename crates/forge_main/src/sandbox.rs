use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use forge_app::Services;
use forge_domain::TitleFormat;
use forge_services::worktree_manager;

use crate::title_display::TitleDisplayExt;

/// Thin wrapper around [`worktree_manager::create_worktree`] that
/// handles the `--worktree <name>` CLI flag path.
///
/// Responsibilities on top of the plain manager:
///
/// 1. Fires the `WorktreeCreate` plugin hook via
///    [`forge_app::fire_worktree_create_hook`] so plugins can veto the creation
///    or hand back a custom path.
/// 2. Prints a user-facing status title with [`TitleFormat::info`] — something
///    the manager itself must not do because the manager is shared with the
///    future runtime `EnterWorktreeTool` path (deferred).
/// 3. Canonicalizes a plugin-provided path before returning it.
pub struct Sandbox<'a, S: Services + 'static> {
    dir: &'a str,
    services: Arc<S>,
}

impl<'a, S: Services + 'static> Sandbox<'a, S> {
    pub fn new(dir: &'a str, services: Arc<S>) -> Self {
        Self { dir, services }
    }

    /// Handles worktree creation and returns the path to the worktree
    /// directory.
    ///
    /// # Plugin hook semantics
    ///
    /// - If a `WorktreeCreate` plugin hook sets `blocking_error`, the creation
    ///   is aborted with that error message.
    /// - If a plugin hook provides a `worktreePath` override, that path is
    ///   validated (must exist) and canonicalized; the built-in `git worktree
    ///   add` path is skipped entirely.
    /// - Otherwise the manager's `git worktree add` path runs normally and the
    ///   resulting path is returned.
    ///
    /// Hook dispatch errors are never fatal: they are handled
    /// fail-open inside [`forge_app::fire_worktree_create_hook`],
    /// which returns an empty aggregate on failure. Only git
    /// errors from the fallback path and plugin-reported
    /// `blocking_error`s propagate out of this function.
    pub async fn create(&self) -> Result<PathBuf> {
        let worktree_name = self.dir;

        // Fire the WorktreeCreate plugin hook BEFORE touching git so
        // plugins have a chance to veto or re-route the creation.
        let hook_result =
            forge_app::fire_worktree_create_hook(self.services.clone(), worktree_name.to_string())
                .await;

        // Check blocking_error first — plugin can veto worktree creation.
        if let Some(err) = hook_result.blocking_error {
            bail!("Worktree creation blocked by plugin hook: {}", err.message);
        }

        // If a plugin provided a worktreePath override, use it verbatim
        // and skip the built-in `git worktree add` fallback.
        let worktree_path: PathBuf = if let Some(path) = hook_result.worktree_path {
            tracing::info!(
                path = %path.display(),
                "worktree path provided by WorktreeCreate plugin hook, skipping git worktree add"
            );
            if !path.exists() {
                bail!(
                    "Plugin-provided worktree path does not exist: {}",
                    path.display()
                );
            }
            path.canonicalize()
                .context("Failed to canonicalize plugin-provided worktree path")?
        } else {
            // No plugin override — fall back to the manager's
            // built-in `git worktree add` flow. The manager is
            // deliberately side-effect-free on stdout so the status
            // print lives here in the wrapper.
            let result = worktree_manager::create_worktree(worktree_name)?;
            let title = if result.created {
                "Worktree [Created]"
            } else {
                "Worktree [Reused]"
            };
            println!(
                "{}",
                TitleFormat::info(title)
                    .sub_title(result.path.display().to_string())
                    .display()
            );
            result.path
        };

        Ok(worktree_path)
    }

    /// Remove a previously-created worktree and fire the `WorktreeRemove`
    /// plugin hook.
    ///
    /// # TODO(hooks)
    ///
    /// This method is a stub — worktree cleanup is not yet implemented.
    /// To complete this:
    ///   1. Accept the `worktree_path: PathBuf` of the worktree to remove.
    ///   2. Fire the hook BEFORE removal so plugins can veto:
    ///        let hook_result = forge_app::fire_worktree_remove_hook(
    ///            self.services.clone(), worktree_path.clone()
    ///        ).await;
    ///   3. If `hook_result.blocking_error` is set, abort the removal.
    ///   4. Otherwise, run `git worktree remove --force <path>` (with an
    ///      `rm -rf` fallback for non-git worktrees).
    ///   5. Call this from the session exit path in `main.rs` or from a
    ///      future `ExitWorktreeTool` / `--sandbox-remove` CLI flag.
    ///
    /// See: `forge_app::fire_worktree_remove_hook`
    #[allow(dead_code)]
    pub async fn remove(services: Arc<S>, worktree_path: PathBuf) -> Result<()> {
        // TODO(hooks): Implement worktree removal with hook integration.
        // Fire `forge_app::fire_worktree_remove_hook(services, worktree_path)`
        // before executing `git worktree remove`.
        let _ = (services, worktree_path);
        bail!("Worktree removal is not yet implemented")
    }
}
