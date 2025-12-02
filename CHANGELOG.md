# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of NanoWasm interpreter
- Zero-copy, zero-allocation execution model
- Full i32/i64 integer support
- Complete f64 floating-point support
- Control flow (blocks, loops, conditionals, branches)
- Memory operations with bounds checking
- Global variables support
- Internal and host function calls
- Extended opcodes (memory.copy, memory.fill, table.init, etc.)
- Pluggable logging system via Logger trait
- Comprehensive README with comparison to Wasmi and Wasm3
- MIT License

### Features
- Streaming execution without AST construction
- Fixed-size runtime structures for deterministic memory usage
- Memory isolation for multi-tenant scenarios
- Overflow-safe address calculations
- Comprehensive bounds checking

### Limitations
- Fixed stack size (64 values)
- Fixed locals (16 per function)
- Fixed control depth (16 nested blocks)
- Fixed call depth (8 function calls)
- No tables (indirect calls) support
- No SIMD support
- Single-threaded execution only

## [0.1.0] - 2025-12-02

### Added
- Initial release

[Unreleased]: https://github.com/AureClai/nano_wasm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AureClai/nano_wasm/releases/tag/v0.1.0

