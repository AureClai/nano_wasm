# NanoWasm

[![no_std](https://img.shields.io/badge/no__std-yes-blue.svg)](https://github.com/rust-lang/rfcs/blob/master/text/1184-stabilize-no_std.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A zero-copy, zero-allocation WebAssembly interpreter designed for embedded and critical systems.**

NanoWasm is a lightweight, streaming WebAssembly interpreter optimized for resource-constrained environments such as embedded systems, unikernels, drones, and satellites. It executes WASM bytecode directly from source buffers without building an Abstract Syntax Tree (AST), making it ideal for real-time and safety-critical applications.

## 🎯 Design Philosophy

NanoWasm follows the **"Nano"** design principles:

- **Zero-Copy**: Executes directly from the source buffer without copying or transforming bytecode
- **Zero-Allocation**: Uses fixed-size arrays for all runtime structures (stack, locals, control frames)
- **Streaming Execution**: Single-pass fetch-decode-execute loop without AST construction
- **Memory Safety**: Designed for critical systems with comprehensive bounds checking
- **Minimal Dependencies**: Only requires `libm` for floating-point operations in `no_std` environments

## ✨ Features

### Core Capabilities

- ✅ **Full i32/i64 Support**: Complete integer arithmetic, bitwise operations, and comparisons
- ✅ **f64 Floating-Point**: Full double-precision floating-point support for mission logic
- ✅ **Control Flow**: Blocks, loops, conditionals, branches, and function calls
- ✅ **Memory Operations**: Load/store operations with bounds checking
- ✅ **Global Variables**: Support for mutable and immutable globals
- ✅ **Function Calls**: Both internal WASM functions and host function imports
- ✅ **Extended Opcodes**: Support for bulk memory operations (memory.copy, memory.fill)

### Safety Features

- **Bounds Checking**: All memory accesses are validated before execution
- **Stack Overflow Protection**: Fixed-size stacks prevent unbounded growth
- **Overflow-Safe Arithmetic**: Address calculations use checked arithmetic
- **Memory Isolation**: Designed for multi-tenant scenarios with isolated memory buffers

### Performance Characteristics

- **Deterministic Execution**: Fixed-size structures ensure predictable memory usage
- **Low Overhead**: Minimal runtime overhead suitable for hard real-time systems
- **Cache-Friendly**: Linear memory access patterns optimize for CPU caches

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
nano_wasm = { path = "../crates/nano_wasm" }
# or from crates.io (when published)
# nano_wasm = "0.1.0"
```

## 🚀 Quick Start

### Basic Usage

```rust
use nano_wasm::{NanoInterpreter, MEMORY_SIZE, NoOpLogger};

// Your WASM bytecode
let wasm_bytes: &[u8] = /* ... */;

// Allocate memory buffer
let mut memory = vec![0u8; MEMORY_SIZE];

// Create interpreter
let mut interpreter = NanoInterpreter::new(
    wasm_bytes,
    &mut memory[..],
    false  // Don't preserve memory between runs
);

// Execute
match interpreter.run() {
    Ok(result) => println!("Result: {}", result),
    Err(e) => eprintln!("Error: {}", e),
}
```

### With Logging

```rust
use nano_wasm::{NanoInterpreter, Logger};

struct MyLogger;

impl Logger for MyLogger {
    fn log(&self, msg: &str) {
        println!("[NanoWasm] {}", msg);
    }
}

let logger = MyLogger;
let mut interpreter = NanoInterpreter::with_logger(
    wasm_bytes,
    &mut memory[..],
    false,
    Some(&logger)
);
```

## 📚 API Reference

### Core Types

#### `NanoInterpreter<'a, L: Logger = NoOpLogger>`

The main interpreter struct. Generic over a logger type for optional logging support.

**Methods:**

- `new(input, memory, preserve_memory)` - Create a new interpreter with no-op logger
- `with_logger(input, memory, preserve_memory, logger)` - Create with custom logger
- `run() -> Result<i32, &'static str>` - Execute the WASM module and return result

**Parameters:**

- `input: &'a [u8]` - The WASM bytecode to execute
- `memory: &'a mut [u8]` - Linear memory buffer (must be at least `MEMORY_SIZE`)
- `preserve_memory: bool` - If `true`, preserves memory contents between runs
- `logger: Option<&'a L>` - Optional logger for debug output

#### `Logger` Trait

Trait for pluggable logging:

```rust
pub trait Logger {
    fn log(&self, msg: &str);
}
```

Implement this trait to provide custom logging behavior.

#### Constants

- `MEMORY_SIZE: usize` - Default memory size (2 MB)

## 🏗️ Architecture

### Execution Model

NanoWasm uses a **streaming execution** model:

1. **Fetch**: Read opcode from bytecode buffer
2. **Decode**: Parse opcode and immediate operands
3. **Execute**: Perform operation directly
4. **Repeat**: Continue until function end or error

No intermediate representation (IR) or AST is constructed, minimizing memory overhead.

### Memory Layout

```
┌─────────────────────────────────────┐
│  Value Stack (64 × u64)            │  ← Fixed-size stack
├─────────────────────────────────────┤
│  Locals Array (16 × u64)            │  ← Function locals
├─────────────────────────────────────┤
│  Globals Array (16 × u64)           │  ← Global variables
├─────────────────────────────────────┤
│  Control Stack (16 frames)          │  ← Block/loop/if frames
├─────────────────────────────────────┤
│  Call Stack (8 frames)               │  ← Function call frames
├─────────────────────────────────────┤
│  Linear Memory (2 MB default)       │  ← WASM linear memory
└─────────────────────────────────────┘
```

### Supported Opcodes

#### Control Flow
- `block`, `loop`, `if`, `else`, `end`
- `br`, `br_if`, `return`
- `call` (internal and host functions)

#### Variables
- `local.get`, `local.set`, `local.tee`
- `global.get`, `global.set`

#### Integer Operations (i32/i64)
- Arithmetic: `add`, `sub`, `mul`, `div`, `rem`
- Bitwise: `and`, `or`, `xor`, `shl`, `shr`, `rotl`
- Comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Constants: `i32.const`, `i64.const`

#### Floating-Point Operations (f64)
- Arithmetic: `add`, `sub`, `mul`, `div`
- Comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Unary: `abs`, `neg`, `ceil`, `floor`, `trunc`, `nearest`, `sqrt`
- Min/Max: `min`, `max`
- Constants: `f64.const`
- Conversions: `i32.trunc_f64_s/u`, `f64.convert_i32_s/u`

#### Memory Operations
- Load: `i32.load`, `i32.load8_s/u`, `i32.load16_s/u`, `f64.load`
- Store: `i32.store`, `i32.store8`, `i32.store16`, `f64.store`
- Management: `memory.size`, `memory.grow`

#### Extended Opcodes (0xFC prefix)
- `memory.copy` - Copy memory regions
- `memory.fill` - Fill memory with value
- `data.drop`, `table.init`, `elem.drop` - Bulk memory operations

## 🔒 Safety & Limitations

### Safety Guarantees

- **Bounds Checking**: All memory accesses are validated
- **Stack Protection**: Fixed-size stacks prevent overflow
- **No Unsafe Code**: Safe Rust implementation (except where explicitly documented)
- **Memory Isolation**: Each interpreter instance uses isolated memory

### Limitations

Due to the "Nano" design philosophy, NanoWasm has intentional limitations:

- **Fixed Stack Size**: 64 values (configurable via constants)
- **Fixed Locals**: 16 local variables per function
- **Fixed Control Depth**: 16 nested blocks/loops
- **Fixed Call Depth**: 8 function call levels
- **No Tables**: Indirect function calls (tables) are not supported
- **No SIMD**: SIMD instructions are not supported
- **No Threading**: Single-threaded execution only

These limitations ensure deterministic, bounded resource usage suitable for critical systems.

## 📊 Comparison with Other WASM Interpreters

### NanoWasm vs Wasmi vs Wasm3

| Feature | **NanoWasm** | **Wasmi** | **Wasm3** |
|---------|--------------|-----------|-----------|
| **Language** | Rust (no_std) | Rust (std) | C |
| **no_std Support** | ✅ Full | ❌ Requires std | ✅ Full |
| **Memory Footprint** | ~2 MB fixed | Variable (10-50 MB) | ~64 KB code + 10 KB RAM |
| **Binary Size** | Small (~50-100 KB) | Medium (~500 KB+) | Very Small (~64 KB) |
| **Startup Time** | Instant | Fast | Instant |
| **Execution Speed** | Interpreted (~2-5% native) | Interpreted (~1.6% native) | Interpreted (~8% native) |
| **Zero-Allocation** | ✅ Yes | ❌ No | ✅ Yes |
| **Zero-Copy** | ✅ Yes | ⚠️ Partial | ✅ Yes |
| **AST Construction** | ❌ None (streaming) | ✅ Yes | ❌ None (streaming) |
| **Deterministic Memory** | ✅ Fixed-size | ❌ Dynamic | ✅ Bounded |
| **Hard Real-Time** | ✅ Suitable | ⚠️ Limited | ✅ Suitable |
| **Embedded Systems** | ✅ Excellent | ❌ Requires std | ✅ Excellent |
| **Safety (Rust)** | ✅ Memory safe | ✅ Memory safe | ⚠️ C (manual safety) |
| **Tables Support** | ❌ No | ✅ Yes | ✅ Yes |
| **SIMD Support** | ❌ No | ⚠️ Partial | ⚠️ Partial |
| **Threading** | ❌ No | ⚠️ Limited | ⚠️ Limited |
| **Bulk Memory Ops** | ✅ Yes | ✅ Yes | ✅ Yes |
| **f64 Support** | ✅ Full | ✅ Full | ✅ Full |
| **Host Functions** | ✅ Basic | ✅ Advanced | ✅ Advanced |
| **Gas Metering** | ❌ No | ✅ Yes | ✅ Yes |
| **Tracing/Debugging** | ⚠️ Basic | ✅ Advanced | ✅ Advanced |

### Detailed Comparison

#### **Performance**

**NanoWasm:**
- Streaming execution without AST overhead
- Fixed-size structures minimize allocation overhead
- Optimized for predictable, deterministic performance
- Estimated: ~2-5% of native speed (interpreted)

**Wasmi:**
- AST-based execution adds overhead
- Dynamic allocations can cause GC pauses
- Focus on correctness over speed
- Benchmarked: ~1.6% of native speed

**Wasm3:**
- Highly optimized C implementation
- Efficient opcode dispatch
- Best-in-class interpreter performance
- Benchmarked: ~8% of native speed

#### **Memory Usage**

**NanoWasm:**
- **Fixed Memory**: 2 MB linear memory (configurable)
- **Runtime Structures**: ~8 KB (stack + locals + control frames)
- **Total**: ~2 MB + code size
- **Deterministic**: Memory usage is predictable and bounded

**Wasmi:**
- **Dynamic Memory**: Variable based on module size
- **AST Storage**: Can be significant for large modules
- **Total**: 10-50 MB+ depending on module
- **Non-Deterministic**: Memory usage varies with module complexity

**Wasm3:**
- **Code Size**: ~64 KB
- **RAM**: ~10 KB base + module-dependent
- **Total**: Very small footprint
- **Deterministic**: Bounded memory usage

#### **Use Case Fit**

**Choose NanoWasm when:**
- ✅ You need **no_std** Rust compatibility
- ✅ You require **deterministic memory usage**
- ✅ You're building **critical/embedded systems**
- ✅ You need **memory isolation** (multi-tenant)
- ✅ You want **zero-allocation** guarantees
- ✅ You're targeting **hard real-time** systems
- ✅ You need **Rust memory safety** in embedded contexts

**Choose Wasmi when:**
- ✅ You need **full WebAssembly spec compliance**
- ✅ You want **advanced features** (tables, SIMD)
- ✅ You're building **general-purpose** applications
- ✅ You need **gas metering** and advanced tooling
- ✅ You can use **std** library

**Choose Wasm3 when:**
- ✅ You need **maximum performance** (interpreter)
- ✅ You're targeting **very resource-constrained** systems
- ✅ You need **C API** compatibility
- ✅ You want **smallest binary size**
- ✅ You're comfortable with **C codebase**

#### **Architecture Differences**

**NanoWasm Architecture:**
```
Bytecode → Fetch → Decode → Execute → Result
           (No AST, No IR, Direct Execution)
```

**Wasmi Architecture:**
```
Bytecode → Parse → AST → Validate → Execute → Result
           (AST Construction, Validation Pass)
```

**Wasm3 Architecture:**
```
Bytecode → Parse → Optimize → Execute → Result
           (Lightweight Parsing, Fast Dispatch)
```

#### **Trade-offs Summary**

| Aspect | NanoWasm Advantage | Trade-off |
|--------|-------------------|-----------|
| **Memory** | Predictable, bounded | Fixed size limits flexibility |
| **Startup** | Instant (no parsing) | No pre-validation |
| **Safety** | Rust memory safety | Requires Rust ecosystem |
| **Features** | Focused, minimal | Missing advanced features |
| **Real-Time** | Deterministic | Fixed limits may be restrictive |

### Performance Benchmarks (Estimated)

Based on typical WebAssembly interpreter performance:

```
Native Execution:    100% (baseline)
─────────────────────────────────────
Wasm3:              ~8%   (fastest interpreter)
NanoWasm:           ~2-5% (estimated, streaming)
Wasmi:              ~1.6% (AST-based)
─────────────────────────────────────
Wasmtime (JIT):     ~70-90% (compiled)
```

**Note**: Actual performance depends heavily on workload characteristics. NanoWasm's streaming approach may perform better on simple, linear code paths compared to AST-based interpreters.

### When to Choose Each

**NanoWasm** is the best choice for:
- 🎯 **Critical Systems**: Drones, satellites, avionics
- 🎯 **Unikernels**: Single-address-space systems
- 🎯 **Embedded Rust**: no_std environments
- 🎯 **Memory Isolation**: Multi-tenant scenarios
- 🎯 **Hard Real-Time**: Deterministic execution requirements

**Wasmi** is better for:
- 🎯 **General Applications**: Full-featured WASM runtime
- 🎯 **Substrate/Blockchain**: Gas metering, advanced features
- 🎯 **Development Tools**: Rich debugging and validation

**Wasm3** is better for:
- 🎯 **Maximum Speed**: Best interpreter performance
- 🎯 **Microcontrollers**: Smallest footprint
- 🎯 **C Integration**: C/C++ projects
- 🎯 **Cross-Platform**: Wide architecture support

## 🎓 Use Cases

### Ideal For

- **Embedded Systems**: Microcontrollers, IoT devices
- **Unikernels**: Single-address-space operating systems
- **Critical Systems**: Drones, satellites, avionics
- **Real-Time Systems**: Hard real-time applications requiring deterministic execution
- **Sandboxed Execution**: Isolated execution environments

### Not Suitable For

- **General-Purpose WASM Runtime**: Use full-featured runtimes like Wasmtime or Wasmer
- **High-Performance Computing**: Optimized JIT compilers are better suited
- **Complex WASM Modules**: Modules requiring tables, SIMD, or threading

## 📊 Performance

### Benchmarks

NanoWasm is optimized for:

- **Low Memory Footprint**: ~2 MB memory + fixed-size runtime structures
- **Deterministic Execution**: Predictable performance characteristics
- **Fast Startup**: No compilation or optimization passes
- **Cache Efficiency**: Linear memory access patterns

### Comparison

| Feature | NanoWasm | Full WASM Runtime |
|---------|----------|-------------------|
| Memory Overhead | ~2 MB fixed | Variable (10-100+ MB) |
| Startup Time | Instant | Seconds (compilation) |
| Execution Speed | Interpreted | JIT-compiled (faster) |
| Determinism | High | Variable |
| no_std Support | ✅ Full | ⚠️ Limited |

## 🔧 Configuration

### Adjusting Limits

Edit constants in `src/interpreter.rs`:

```rust
const STACK_SIZE: usize = 64;        // Value stack size
const LOCALS_SIZE: usize = 16;       // Local variables per function
const CONTROL_STACK_SIZE: usize = 16; // Max nesting depth
const MAX_CALL_DEPTH: usize = 8;     // Max function call depth
const GLOBALS_SIZE: usize = 16;      // Max global variables
pub const MEMORY_SIZE: usize = 2 * 1024 * 1024; // Linear memory size
```

## 📝 Examples

### Example 1: Simple Arithmetic

```rust
use nano_wasm::NanoInterpreter;

// WASM: (func (result i32) i32.const 42 i32.const 10 i32.add)
let wasm = &[
    0x00, 0x61, 0x73, 0x6d, // WASM magic
    0x01, 0x00, 0x00, 0x00, // Version
    // ... module definition ...
];

let mut memory = vec![0u8; nano_wasm::MEMORY_SIZE];
let mut interpreter = NanoInterpreter::new(wasm, &mut memory, false);
let result = interpreter.run().unwrap();
assert_eq!(result, 52);
```

### Example 2: Host Function Import

```rust
// Define a host function that can be called from WASM
// This requires implementing host function dispatch in your integration
```

### Example 3: Memory Operations

```rust
// WASM modules can read/write to the linear memory buffer
// Memory is isolated per interpreter instance for safety
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional opcode support
- Performance optimizations
- Better error messages
- More comprehensive tests
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

NanoWasm is designed for the **Capsule Unikernel** project, implementing the "Split-Brain" architecture where:

- **Spinal Cord** (Native Rust): Hard real-time, formally verified, immutable
- **Brain** (WebAssembly): High-level logic, sandboxed, updatable

This separation enables safe, updatable mission logic in critical systems while maintaining the "uncrashable" guarantee of the Spinal Cord.

## 📚 References

- [WebAssembly Specification](https://webassembly.github.io/spec/)
- [WebAssembly Binary Format](https://webassembly.github.io/spec/core/binary/index.html)
- [LEB128 Encoding](https://en.wikipedia.org/wiki/LEB128)

---

**Built with ❤️ for critical systems**

