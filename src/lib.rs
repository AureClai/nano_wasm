#![no_std]

//! A "Nano" WebAssembly Interpreter
//!
//! - Zero-Copy: Executes directly from the source buffer.
//! - Zero-Allocation: Uses fixed-size arrays for stack and locals.
//! - Streaming: Fetch-Decode-Execute loop (no AST).
//! - Constraints: Limited stack/locals, supports f64 for Brain logic.

/// Logger trait for optional logging support
pub trait Logger {
    fn log(&self, msg: &str);
}

/// No-op logger (default)
pub struct NoOpLogger;

impl Logger for NoOpLogger {
    fn log(&self, _msg: &str) {
        // No-op
    }
}

// Re-export the main types
pub use interpreter::{NanoInterpreter, MEMORY_SIZE};

// Type alias for convenience (default no-op logger)
pub type NanoInterpreterDefault<'a> = NanoInterpreter<'a, NoOpLogger>;

mod interpreter;
