# Contributing to NanoWasm

Thank you for your interest in contributing to NanoWasm! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include steps to reproduce
- Provide environment details (OS, Rust version, target)
- If possible, include the WASM module that causes the issue

### Suggesting Features

- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Clearly describe the use case
- Explain how it fits with NanoWasm's design philosophy

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Follow the coding standards** (see below)
5. **Add tests** if applicable
6. **Update documentation** as needed
7. **Run tests**: `cargo test`
8. **Check formatting**: `cargo fmt --check`
9. **Run clippy**: `cargo clippy --all-targets --all-features`
10. **Commit your changes**: Use clear, descriptive commit messages
11. **Push to your fork**: `git push origin feature/amazing-feature`
12. **Open a Pull Request**: Use the PR template

## Coding Standards

### Rust Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common mistakes
- Document public APIs with `///` doc comments

### Design Principles

NanoWasm follows the **"Nano"** philosophy:

- ✅ **Zero-Allocation**: No dynamic memory allocation at runtime
- ✅ **Zero-Copy**: Execute directly from source buffer
- ✅ **Deterministic**: Fixed-size structures, predictable memory usage
- ✅ **Safety**: Comprehensive bounds checking, memory safety

When adding features, consider:
- Does it maintain zero-allocation guarantees?
- Does it fit within fixed-size constraints?
- Is it necessary for critical/embedded systems?
- Can it be optional via feature flags?

### Code Organization

- Keep the interpreter logic in `src/interpreter.rs`
- Public API should be minimal and well-documented
- Use modules to organize related functionality
- Add tests alongside code

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run tests for specific target
cargo test --target thumbv7em-none-eabihf
```

### Writing Tests

- Add unit tests in the same file using `#[cfg(test)]`
- Add integration tests in `tests/` directory
- Test edge cases and error conditions
- Test with real WASM modules when possible

### no_std Testing

NanoWasm must work in `no_std` environments. Test with:

```bash
cargo build --target thumbv7em-none-eabihf --no-default-features
cargo build --target riscv32imac-unknown-none-elf --no-default-features
```

## Documentation

### Code Documentation

- Document all public APIs
- Use examples in doc comments
- Explain design decisions in comments
- Keep README.md up to date

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add support for i64 operations
fix: correct stack overflow detection
docs: update README with performance benchmarks
refactor: simplify opcode dispatch
test: add tests for f64 conversions
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nano_wasm.git
   cd nano_wasm
   ```

2. Install Rust (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. Install required tools:
   ```bash
   rustup component add rustfmt clippy
   ```

4. Build the project:
   ```bash
   cargo build
   ```

5. Run tests:
   ```bash
   cargo test
   ```

## Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md` with new features/fixes
3. Create a git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Create GitHub release with changelog

## Questions?

- Open an issue for questions or discussions
- Check existing issues and PRs
- Review the README.md for usage examples

Thank you for contributing to NanoWasm! 🚀

