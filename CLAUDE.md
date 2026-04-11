# CLAUDE.md

## Testing

Use `cargo nextest run` instead of `cargo test`. The project is configured for nextest (see `.config/nextest.toml`).

Always pass `--no-input-handler` to avoid a crossterm panic in non-interactive environments (e.g. when run by an LLM agent).

```bash
# Run all tests
cargo nextest run --no-input-handler

# Only unit tests (fast feedback loop)
cargo nextest run --no-input-handler --lib

# Specific crate
cargo nextest run --no-input-handler -p forge_domain

# Integration tests only
cargo nextest run --no-input-handler --test '*'

# Watch mode (auto-rerun on file changes)
cargo watch -x "nextest run --no-input-handler --lib"
```

Do NOT silently skip work. If a task is out of scope for the current change, place the TODO and mention it in your response summary.
