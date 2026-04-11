# CLAUDE.md

## Testing

Use `cargo nextest run` instead of `cargo test`. The project is configured for nextest (see `.config/nextest.toml`).

```bash
# Run all tests
cargo nextest run

# Only unit tests (fast feedback loop)
cargo nextest run --lib

# Specific crate
cargo nextest run -p forge_domain

# Integration tests only
cargo nextest run --test '*'

# Watch mode (auto-rerun on file changes)
cargo watch -x "nextest run --lib"
```
