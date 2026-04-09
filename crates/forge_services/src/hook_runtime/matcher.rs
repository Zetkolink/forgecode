//! Hook matcher evaluation.
//!
//! This module ports two distinct matchers from Claude Code into Forge:
//!
//! 1. [`matches_pattern`] — evaluates the `matcher` field of a
//!    [`forge_domain::HookMatcher`] against a tool name. Supports exact
//!    strings, wildcards, pipe-separated alternatives, and regexes.
//!    Source of truth: `claude-code/src/utils/hooks.ts:1346-1390`.
//!
//! 2. [`matches_condition`] — evaluates the `if` field of a hook command
//!    against the current `tool_name` and `tool_input`. Uses the
//!    permission-rule syntax `ToolName(argument_pattern)` (e.g.
//!    `"Bash(git *)"`). Mirrors Claude Code's permission rule engine.
//!
//! Both matchers are pure and side-effect free. Unknown/empty conditions
//! always match so that misconfigured rules don't silently block hooks.

use glob::Pattern;
use regex::Regex;

/// Evaluate a hook `matcher` pattern against a tool name.
///
/// Order of checks (mirrors Claude Code):
/// 1. Empty or `"*"` → matches everything.
/// 2. Regex-like pattern (detected heuristically via special characters)
///    → compiled with the `regex` crate and tested. Checked before the
///    pipe-list branch so that a regex alternation like `^(Read|Write)$`
///    isn't mis-split into exact alternatives.
/// 3. Pipe-separated list (`"Write|Edit|Bash"`) → any exact alternative
///    matches.
/// 4. Exact case-sensitive match.
///
/// The `regex` crate provides linear-time matching with no catastrophic
/// backtracking, so untrusted plugin patterns are safe.
pub fn matches_pattern(pattern: &str, tool_name: &str) -> bool {
    let trimmed = pattern.trim();

    // 1. Empty or "*" → match everything.
    if trimmed.is_empty() || trimmed == "*" {
        return true;
    }

    // 2. Regex. Heuristic: if the pattern contains any regex special
    //    char that wouldn't appear in a plain identifier or a simple
    //    pipe-list, treat it as a regex. This must run before the
    //    pipe-split branch so that `^(Read|Write)$` is handled as a
    //    regex rather than split into two alternatives.
    if contains_regex_metachars(trimmed) {
        if let Ok(re) = Regex::new(trimmed) {
            return re.is_match(tool_name);
        }
    }

    // 3. Pipe list — any exact alternative matches.
    if trimmed.contains('|') {
        return trimmed
            .split('|')
            .map(str::trim)
            .any(|alt| alt == tool_name);
    }

    // 4. Exact match.
    trimmed == tool_name
}

/// Evaluate a hook `if` condition (permission-rule syntax) against the
/// current tool invocation.
///
/// The condition may be one of two forms:
/// - `"ToolName"` — matches whenever `tool_name` equals the name.
/// - `"ToolName(argument_pattern)"` — matches when the tool name equals
///   the name AND a tool-specific argument extracted from `tool_input`
///   matches `argument_pattern` using glob-style matching.
///
/// Argument extraction rules (per Claude Code):
/// - `Bash` — the argument is `tool_input["command"]`.
/// - `Read` / `Write` / `Edit` / `MultiEdit` / `NotebookEdit` — the
///   argument is `tool_input["file_path"]` or `tool_input["path"]`
///   (whichever exists).
/// - Any other tool — the argument is the JSON-serialized `tool_input`.
///
/// An empty or unparseable condition always matches so that a typo in a
/// plugin's `hooks.json` doesn't silently swallow hook events.
pub fn matches_condition(
    condition: &str,
    tool_name: &str,
    tool_input: &serde_json::Value,
) -> bool {
    let trimmed = condition.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Parse "ToolName" or "ToolName(argument_pattern)".
    let (name_part, arg_pattern) = match trimmed.find('(') {
        Some(open) if trimmed.ends_with(')') => {
            let name = trimmed[..open].trim();
            let inner = &trimmed[open + 1..trimmed.len() - 1];
            (name, Some(inner))
        }
        _ => (trimmed, None),
    };

    if name_part != tool_name {
        return false;
    }

    let Some(pattern) = arg_pattern else {
        // Bare "ToolName" form — tool name match is sufficient.
        return true;
    };

    let argument = extract_condition_argument(tool_name, tool_input);
    glob_match(pattern, &argument)
}

/// Extract the argument string used to evaluate a condition glob.
fn extract_condition_argument(tool_name: &str, tool_input: &serde_json::Value) -> String {
    match tool_name {
        "Bash" => tool_input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "Read" | "Write" | "Edit" | "MultiEdit" | "NotebookEdit" => tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
            .or_else(|| tool_input.get("path").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => serde_json::to_string(tool_input).unwrap_or_default(),
    }
}

/// Glob-match a pattern against a target string.
///
/// Uses the `glob` crate's `Pattern` if the pattern compiles; falls back
/// to literal equality otherwise. Matching is case-sensitive and uses
/// default glob options (no case-folding, path separators treated as
/// regular characters so `*` spans `/`).
fn glob_match(pattern: &str, target: &str) -> bool {
    match Pattern::new(pattern) {
        Ok(p) => p.matches(target),
        Err(_) => pattern == target,
    }
}

/// Cheap heuristic: does this string contain a character that would only
/// appear in a regex, not in a plain tool name?
fn contains_regex_metachars(pattern: &str) -> bool {
    pattern
        .chars()
        .any(|c| matches!(c, '^' | '$' | '[' | ']' | '(' | ')' | '\\' | '.' | '+' | '?' | '{' | '}'))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_empty_matcher_matches_any_tool_name() {
        assert!(matches_pattern("", "Bash"));
        assert!(matches_pattern("", "Write"));
        assert!(matches_pattern("   ", "Anything"));
    }

    #[test]
    fn test_star_matcher_matches_any() {
        assert!(matches_pattern("*", "Bash"));
        assert!(matches_pattern("*", "ReadFile"));
    }

    #[test]
    fn test_exact_match_only_matches_same_name() {
        assert!(matches_pattern("Write", "Write"));
        assert_eq!(matches_pattern("Write", "Bash"), false);
        assert_eq!(matches_pattern("Write", "write"), false);
    }

    #[test]
    fn test_pipe_list_matches_any_alternative() {
        assert!(matches_pattern("Write|Edit|Bash", "Write"));
        assert!(matches_pattern("Write|Edit|Bash", "Edit"));
        assert!(matches_pattern("Write|Edit|Bash", "Bash"));
        assert_eq!(matches_pattern("Write|Edit|Bash", "Read"), false);
    }

    #[test]
    fn test_pipe_list_tolerates_whitespace() {
        assert!(matches_pattern("Write | Edit | Bash", "Edit"));
    }

    #[test]
    fn test_regex_matches_suffix_pattern() {
        assert!(matches_pattern("^.*File$", "ReadFile"));
        assert!(matches_pattern("^.*File$", "WriteFile"));
        assert_eq!(matches_pattern("^.*File$", "Bash"), false);
    }

    #[test]
    fn test_regex_with_character_class() {
        assert!(matches_pattern("^(Read|Write)$", "Read"));
        assert!(matches_pattern("^(Read|Write)$", "Write"));
        assert_eq!(matches_pattern("^(Read|Write)$", "Bash"), false);
    }

    #[test]
    fn test_condition_bash_git_prefix_matches() {
        let input = json!({"command": "git status"});
        assert!(matches_condition("Bash(git *)", "Bash", &input));
    }

    #[test]
    fn test_condition_bash_git_rejects_other_commands() {
        let input = json!({"command": "ls -la"});
        assert_eq!(
            matches_condition("Bash(git *)", "Bash", &input),
            false
        );
    }

    #[test]
    fn test_condition_read_ts_extension_matches() {
        let input_path = json!({"path": "src/main.ts"});
        assert!(matches_condition("Read(*.ts)", "Read", &input_path));

        let input_file_path = json!({"file_path": "src/main.ts"});
        assert!(matches_condition("Read(*.ts)", "Read", &input_file_path));
    }

    #[test]
    fn test_condition_read_ts_rejects_other_extensions() {
        let input = json!({"path": "src/main.rs"});
        assert_eq!(
            matches_condition("Read(*.ts)", "Read", &input),
            false
        );
    }

    #[test]
    fn test_condition_bare_tool_name_matches_any_input() {
        let input = json!({"path": "anything"});
        assert!(matches_condition("Write", "Write", &input));
        assert_eq!(matches_condition("Write", "Read", &input), false);
    }

    #[test]
    fn test_empty_condition_always_matches() {
        let input = json!({});
        assert!(matches_condition("", "Bash", &input));
        assert!(matches_condition("   ", "Read", &input));
    }

    #[test]
    fn test_condition_rejects_when_tool_name_differs() {
        let input = json!({"command": "git status"});
        assert_eq!(
            matches_condition("Bash(git *)", "Write", &input),
            false
        );
    }
}
