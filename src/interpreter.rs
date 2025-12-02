/// A "Nano" WebAssembly Interpreter
///
/// - Zero-Copy: Executes directly from the source buffer.
/// - Zero-Allocation: Uses fixed-size arrays for stack and locals.
/// - Streaming: Fetch-Decode-Execute loop (no AST).
/// - Constraints: Limited stack/locals, supports f64 for Brain logic.
use crate::Logger;

/// Supported WASM Value Types (Internal representation)
/// We treat everything as u64 storage for simplicity, as requested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    I32(i32),
    I64(i64),
}

#[allow(dead_code)]
impl Val {
    fn as_u64(&self) -> u64 {
        match self {
            Val::I32(v) => *v as u64,
            Val::I64(v) => *v as u64,
        }
    }

    fn from_i32(v: i32) -> Self {
        Val::I32(v)
    }
}

const STACK_SIZE: usize = 64;
const LOCALS_SIZE: usize = 16;
pub const MEMORY_SIZE: usize = 2 * 1024 * 1024; // 2 MB
const CONTROL_STACK_SIZE: usize = 16; // Max nesting depth
const MAX_CALL_DEPTH: usize = 8; // Max function call depth
const GLOBALS_SIZE: usize = 16; // Max global variables

// Control Frame: stores PC to return to after block/loop
#[derive(Clone, Copy)]
struct ControlFrame {
    return_pc: usize,             // PC to jump to on 'end' or 'br'
    is_loop: bool,                // If true, 'br' jumps to start; if false, jumps to end
    stack_height_at_entry: usize, // Stack height when entering this block (for validation/cleanup)
}

// Call Frame: stores context for function returns
#[derive(Clone, Copy)]
struct CallFrame {
    return_pc: usize,
    prev_control_sp: usize,
    // Locals should ideally be saved/restored or using a sliding window,
    // but for NanoWasm we might just use a global locals array and simplistic management
    // or assume no recursion for now. Let's implement proper save/restore later.
    // For now, we'll just support calls that don't need to preserve locals (simple helpers).
    // WAIT: Real WASM calls have their own locals.
    // We need a "Locals Stack" or similar.
    locals_base: usize, // Index into a larger locals buffer?
}

// Use a static mutable buffer to ensure "Zero-Allocation" at runtime while supporting large memory.
// We use `static mut` which is unsafe, but since NanoWasm is single-threaded per execution (consumed by run),
// we can manage it carefully.
// static mut WASM_HEAP: [u8; MEMORY_SIZE] = [0; MEMORY_SIZE];

/// The Interpreter State
pub struct NanoInterpreter<'a, L: Logger = crate::NoOpLogger> {
    input: &'a [u8],
    pc: usize, // Program Counter (byte index in input)

    // Value Stack: Fixed [u64; 64]
    stack: [u64; STACK_SIZE],
    sp: usize, // Stack Pointer

    // Locals: Fixed [u64; 16] -> Expanded for calls?
    // For strict "Nano", let's keep it simple:
    // We will share the locals array for now, or fail if calls are used without proper locals support.
    locals: [u64; LOCALS_SIZE],

    // Globals: Fixed [u64; 16] for global variables
    globals: [u64; GLOBALS_SIZE],

    // Linear Memory: Reference to global static buffer
    memory: &'a mut [u8],

    // Control Stack: Fixed array for block/loop/if frames
    control_stack: [ControlFrame; CONTROL_STACK_SIZE],
    control_sp: usize, // Control stack pointer

    // Call Stack
    call_stack: [CallFrame; MAX_CALL_DEPTH],
    call_sp: usize,

    // Optional logger
    logger: Option<&'a L>,
}

impl<'a, L: Logger> NanoInterpreter<'a, L> {
    pub fn new(input: &'a [u8], memory: &'a mut [u8], preserve_memory: bool) -> Self {
        Self::with_logger(input, memory, preserve_memory, None)
    }

    pub fn with_logger(
        input: &'a [u8],
        memory: &'a mut [u8],
        preserve_memory: bool,
        logger: Option<&'a L>,
    ) -> Self {
        // UNSAFE: Accessing static mut.
        // Assumption: NanoInterpreter is used sequentially.
        /*let memory = unsafe {
            &mut *(&raw mut WASM_HEAP)
        };*/

        let mut interpreter = Self {
            input,
            pc: 0,
            stack: [0; STACK_SIZE],
            sp: 0,
            locals: [0; LOCALS_SIZE],
            globals: [0; GLOBALS_SIZE],
            memory,
            control_stack: [ControlFrame {
                return_pc: 0,
                is_loop: false,
                stack_height_at_entry: 0,
            }; CONTROL_STACK_SIZE],
            control_sp: 0,
            call_stack: [CallFrame {
                return_pc: 0,
                prev_control_sp: 0,
                locals_base: 0,
            }; MAX_CALL_DEPTH],
            call_sp: 0,
            logger,
        };

        // Initialize Globals from Globals Section (ID 6)
        interpreter.init_globals();

        // Initialize Memory from Data Section ONLY if not preserving
        if !preserve_memory {
            interpreter.init_memory();
        }

        interpreter
    }

    fn init_globals(&mut self) {
        // Find Globals Section (ID 6)
        if let Ok((mut globals_ptr, _size)) = self.find_section(6) {
            let (count, len) = Self::read_u32_leb128(&self.input[globals_ptr..]);
            globals_ptr += len;

            for i in 0..count {
                if (i as usize) >= GLOBALS_SIZE {
                    break; // Too many globals, skip
                }

                // Read type (value type)
                let _val_type = self.input[globals_ptr];
                globals_ptr += 1;

                // Read mutability (0x00 = immutable, 0x01 = mutable)
                let _mutability = self.input[globals_ptr];
                globals_ptr += 1;

                // Read init expression (must end with 0x0B)
                // For simplicity, we only support i32.const expressions
                if globals_ptr < self.input.len() && self.input[globals_ptr] == 0x41 {
                    // i32.const
                    globals_ptr += 1;
                    let (val, len_bytes) = Self::read_i32_leb128(&self.input[globals_ptr..]);
                    globals_ptr += len_bytes;

                    // Expect 0x0B (end)
                    if globals_ptr < self.input.len() && self.input[globals_ptr] == 0x0B {
                        globals_ptr += 1;
                        self.globals[i as usize] = val as u64;
                    }
                } else {
                    // Skip unknown init expression (find 0x0B)
                    while globals_ptr < self.input.len() && self.input[globals_ptr] != 0x0B {
                        globals_ptr += 1;
                    }
                    if globals_ptr < self.input.len() {
                        globals_ptr += 1; // Skip 0x0B
                    }
                }
            }
        }
    }

    fn init_memory(&mut self) {
        let memory_len = self.memory.len();
        // Find Data Section (ID 11)
        if let Ok((mut data_ptr, _size)) = self.find_section(11) {
            let (count, len) = Self::read_u32_leb128(&self.input[data_ptr..]);
            data_ptr += len;

            for _ in 0..count {
                let _mem_idx = self.input[data_ptr];
                data_ptr += 1;

                // Offset expr: 0x41 (i32.const) VAL 0x0B (end)
                if self.input[data_ptr] == 0x41 {
                    data_ptr += 1;
                    let (offset_val, len) = Self::read_i32_leb128(&self.input[data_ptr..]);
                    data_ptr += len + 1; // +1 for 0x0B end

                    let (data_len, len) = Self::read_u32_leb128(&self.input[data_ptr..]);
                    data_ptr += len;

                    // Copy data to memory
                    if offset_val >= 0 {
                        let start = offset_val as usize;
                        let end = start + data_len as usize;
                        if end <= memory_len && (data_ptr + data_len as usize) <= self.input.len() {
                            self.memory[start..end].copy_from_slice(
                                &self.input[data_ptr..data_ptr + data_len as usize],
                            );
                        }
                    }

                    data_ptr += data_len as usize;
                } else {
                    // Skip unknown segment type? Or just break
                    break;
                }
            }
        }
    }

    // Helper: Push to stack
    fn push(&mut self, val: u64) -> Result<(), &'static str> {
        if self.sp >= STACK_SIZE {
            return Err("Stack overflow");
        }
        self.stack[self.sp] = val;
        self.sp += 1;
        Ok(())
    }

    // Helper: Pop from stack
    fn pop(&mut self) -> Result<u64, &'static str> {
        if self.sp == 0 {
            return Err("Stack underflow");
        }
        self.sp -= 1;
        Ok(self.stack[self.sp])
    }

    // Helper: Pop f64 from stack
    fn pop_f64(&mut self) -> Result<f64, &'static str> {
        let bits = self.pop()?;
        Ok(f64::from_bits(bits))
    }

    // Helper: Push f64 to stack
    fn push_f64(&mut self, val: f64) -> Result<(), &'static str> {
        self.push(val.to_bits())
    }

    // Helper: Log a message
    fn log(&self, msg: &str) {
        if let Some(logger) = self.logger {
            logger.log(msg);
        }
    }

    // Helper: Log formatted debug message
    fn log_debug(&self, msg: &str) {
        self.log(msg);
    }

    // Helper: Read LEB128 encoded u32
    fn read_u32_leb128(bytes: &[u8]) -> (u32, usize) {
        let mut result = 0;
        let mut shift = 0;
        let mut count = 0;

        loop {
            if count >= bytes.len() {
                break;
            }
            let byte = bytes[count];
            result |= ((byte & 0x7F) as u32) << shift;
            count += 1;
            shift += 7;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        (result, count)
    }

    // Helper: Read LEB128 encoded i32
    fn read_i32_leb128(bytes: &[u8]) -> (i32, usize) {
        let mut result = 0;
        let mut shift = 0;
        let mut count = 0;
        let mut last_byte = 0;

        loop {
            if count >= bytes.len() {
                break;
            }
            let byte = bytes[count];
            last_byte = byte;
            result |= ((byte & 0x7F) as i32) << shift;
            count += 1;
            shift += 7;
            if (byte & 0x80) == 0 {
                break;
            }
        }

        if (shift < 32) && ((last_byte & 0x40) != 0) {
            result |= (!0) << shift;
        }
        (result, count)
    }

    fn find_section(&self, section_id: u8) -> Result<(usize, usize), &'static str> {
        let mut ptr = 8; // Skip magic (4) + version (4)

        while ptr < self.input.len() {
            let id = self.input[ptr];
            ptr += 1;
            let (size, len_bytes) = Self::read_u32_leb128(&self.input[ptr..]);
            ptr += len_bytes;

            if id == section_id {
                return Ok((ptr, size as usize));
            }

            ptr += size as usize;
        }
        Err("Section not found")
    }

    pub fn run(&mut self) -> Result<i32, &'static str> {
        if self.input.len() < 8 || &self.input[0..4] != b"\0asm" {
            return Err("Invalid Magic");
        }

        let func_idx = self.find_run_export().unwrap_or(0);
        let import_count = self.count_imported_functions();

        if func_idx < import_count {
            return Err("Cannot run an imported function directly");
        }

        let internal_idx = func_idx - import_count;

        let (mut code_ptr, _code_size) = self.find_section(10)?;
        let (count, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
        code_ptr += len_bytes;

        if internal_idx >= count {
            // Note: Formatting would require alloc, so we skip detailed debug logging
            // In a real implementation, you could use a formatting logger if needed
            return Err("Function index out of bounds");
        }

        for _ in 0..internal_idx {
            let (body_size, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
            code_ptr += len_bytes + body_size as usize;
        }

        let (body_size, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
        let body_start = code_ptr + len_bytes;
        let body_end = body_start + body_size as usize;

        self.pc = body_start;

        // Initialize Locals
        let (local_vec_count, len_bytes) = Self::read_u32_leb128(&self.input[self.pc..]);
        self.pc += len_bytes;

        let mut local_idx = 0;
        for _ in 0..local_vec_count {
            let (count, len_bytes) = Self::read_u32_leb128(&self.input[self.pc..]);
            self.pc += len_bytes;
            let _type = self.input[self.pc]; // type
            self.pc += 1;

            for _ in 0..count {
                if local_idx < LOCALS_SIZE {
                    self.locals[local_idx] = 0;
                    local_idx += 1;
                }
            }
        }

        self.execute_loop(body_end)
    }

    fn count_imported_functions(&self) -> u32 {
        let mut count = 0;
        // Import Section ID = 2
        if let Ok((mut ptr, _size)) = self.find_section(2) {
            let (vec_len, len) = Self::read_u32_leb128(&self.input[ptr..]);
            ptr += len;

            for _ in 0..vec_len {
                // module name
                let (len, l) = Self::read_u32_leb128(&self.input[ptr..]);
                ptr += l;
                ptr += len as usize;
                // field name
                let (len, l) = Self::read_u32_leb128(&self.input[ptr..]);
                ptr += l;
                ptr += len as usize;

                // kind
                let kind = self.input[ptr];
                ptr += 1;
                if kind == 0 {
                    // Function import
                    let (_type_idx, l) = Self::read_u32_leb128(&self.input[ptr..]);
                    ptr += l;
                    count += 1;
                } else if kind == 1 {
                    // Table
                    // Table type: element_type(1) + limits
                    ptr += 1;
                    let flags = self.input[ptr];
                    ptr += 1;
                    let (_min, l) = Self::read_u32_leb128(&self.input[ptr..]);
                    ptr += l;
                    if flags & 1 != 0 {
                        let (_max, l) = Self::read_u32_leb128(&self.input[ptr..]);
                        ptr += l;
                    }
                } else if kind == 2 {
                    // Memory
                    // Memory type: limits
                    let flags = self.input[ptr];
                    ptr += 1;
                    let (_min, l) = Self::read_u32_leb128(&self.input[ptr..]);
                    ptr += l;
                    if flags & 1 != 0 {
                        let (_max, l) = Self::read_u32_leb128(&self.input[ptr..]);
                        ptr += l;
                    }
                } else if kind == 3 {
                    // Global
                    // Global type: valtype(1) + mut(1)
                    ptr += 2;
                }
            }
        }
        count
    }

    fn find_run_export(&self) -> Option<u32> {
        if let Ok((mut ptr, _size)) = self.find_section(7) {
            let (count, len_bytes) = Self::read_u32_leb128(&self.input[ptr..]);
            ptr += len_bytes;

            for _ in 0..count {
                let (name_len, len_bytes) = Self::read_u32_leb128(&self.input[ptr..]);
                ptr += len_bytes;
                let name_slice = &self.input[ptr..ptr + name_len as usize];
                ptr += name_len as usize;

                let export_kind = self.input[ptr];
                ptr += 1;
                let (index, len_bytes) = Self::read_u32_leb128(&self.input[ptr..]);
                ptr += len_bytes;

                if export_kind == 0 && name_slice == b"run" {
                    return Some(index);
                }
            }
        }
        None
    }

    // Helper: Find the matching 'end' for a block/loop starting at start_pc
    // Optimization: Basic caching of loop targets could be added here, but for Nano
    // we stick to scanning. We can optimize by skipping over known instruction lengths.
    fn find_matching_end(&self, start_pc: usize, max_pc: usize) -> Result<usize, &'static str> {
        let mut pc = start_pc;
        let mut depth = 1;

        while pc < max_pc && depth > 0 {
            let opcode = self.input[pc];
            pc += 1;

            match opcode {
                0x02 | 0x03 | 0x04 => {
                    // block, loop, if
                    depth += 1;
                    pc += 1; // Skip type
                }
                0x05 => { // else
                     // else doesn't change depth
                }
                0x0B => {
                    // end
                    depth -= 1;
                    if depth == 0 {
                        return Ok(pc);
                    }
                }
                0x0C | 0x0D => {
                    // br, br_if
                    let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                    pc += len;
                }
                0x10 => {
                    // call
                    let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                    pc += len;
                }
                0x20..=0x24 => {
                    // local.get, local.set, local.tee, global.get, global.set
                    let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                    pc += len;
                }
                0x28..=0x3E => {
                    // memory loads/stores
                    let (_, len1) = Self::read_u32_leb128(&self.input[pc..]);
                    pc += len1;
                    let (_, len2) = Self::read_u32_leb128(&self.input[pc..]);
                    pc += len2;
                }
                0x41 => {
                    // i32.const
                    let (_, len) = Self::read_i32_leb128(&self.input[pc..]);
                    pc += len;
                }
                0xFC => {
                    // Extended opcodes prefix
                    if pc < self.input.len() {
                        let extended_opcode = self.input[pc];
                        pc += 1;
                        match extended_opcode {
                            0x01 | 0x0C => {
                                // data.drop, elem.drop
                                let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                                pc += len;
                            }
                            0x02 | 0x03 => {
                                // memory.copy, memory.fill
                                let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                                pc += len;
                            }
                            0x0B => {
                                // table.init
                                let (_, len) = Self::read_u32_leb128(&self.input[pc..]);
                                pc += len;
                                // Skip stack operands (dst, src, n) - we don't know their size in scanning
                                // This is approximate, but should work for most cases
                            }
                            _ => {
                                // Unknown extended opcode, skip it
                            }
                        }
                    }
                }
                _ => {
                    // Other opcodes don't affect control flow, skip them
                    // WARN: This is risky if we have multi-byte opcodes we don't know about
                }
            }
        }

        Err("Could not find matching end")
    }

    fn execute_loop(&mut self, end_pc: usize) -> Result<i32, &'static str> {
        let memory_len = self.memory.len();
        while self.pc < end_pc {
            let opcode = self.input[self.pc];
            self.pc += 1;

            match opcode {
                0x00 => {
                    return Err("Unreachable executed");
                }
                0x01 => {} // nop
                0x02 => {
                    // block
                    let _type = self.input[self.pc];
                    let block_start = self.pc + 1;
                    self.pc += 1;

                    // Find matching end
                    let block_end = self.find_matching_end(block_start, end_pc)?;

                    if self.control_sp >= CONTROL_STACK_SIZE {
                        return Err("Control stack overflow");
                    }
                    self.control_stack[self.control_sp] = ControlFrame {
                        return_pc: block_end,
                        is_loop: false,
                        stack_height_at_entry: self.sp,
                    };
                    self.control_sp += 1;
                }
                0x03 => {
                    // loop
                    let _type = self.input[self.pc];
                    let loop_start = self.pc + 1;
                    self.pc += 1;

                    // Find matching end (validate structure)
                    let _loop_end = self.find_matching_end(loop_start, end_pc)?;

                    if self.control_sp >= CONTROL_STACK_SIZE {
                        return Err("Control stack overflow");
                    }
                    self.control_stack[self.control_sp] = ControlFrame {
                        return_pc: loop_start, // Loop back to start
                        is_loop: true,
                        stack_height_at_entry: self.sp,
                    };
                    self.control_sp += 1;
                }
                0x04 => {
                    // if
                    let _type = self.input[self.pc];
                    let if_start = self.pc + 1;
                    self.pc += 1;
                    let cond = self.pop()?;

                    // Find matching end (might have else in between)
                    let if_end = self.find_matching_end(if_start, end_pc)?;

                    if self.control_sp >= CONTROL_STACK_SIZE {
                        return Err("Control stack overflow");
                    }
                    self.control_stack[self.control_sp] = ControlFrame {
                        return_pc: if_end,
                        is_loop: false,
                        stack_height_at_entry: self.sp,
                    };
                    self.control_sp += 1;

                    if cond == 0 {
                        // Skip to else/end
                        // For now, just jump to end (simplified - doesn't handle else)
                        self.pc = if_end;
                        self.control_sp -= 1; // Pop the frame we just pushed
                    }
                }
                0x05 => {
                    // else
                    // Jump to end of if block
                    if self.control_sp == 0 {
                        return Err("Control stack underflow");
                    }
                    let frame = self.control_stack[self.control_sp - 1];
                    self.pc = frame.return_pc; // Jump to end
                    self.control_sp -= 1;
                }
                0x0B => {
                    // end
                    if self.control_sp > 0 {
                        // Pop control frame (block/loop/if ended)
                        self.control_sp -= 1;

                        // Check if we hit the "base" of the current function's control stack
                        // If we are inside a call, we shouldn't pop below the caller's stack.
                        // However, since we don't explicitly track "base" per function in control_stack,
                        // we rely on the fact that `end` only matches blocks pushed *in this function*.
                        // BUT: `control_sp` is global.
                        // If we are at the end of the function body, `control_sp` should be equal to `prev_control_sp`.
                        // Actually, the loop structure ensures we match `end`s.
                        // The "Function End" is implicit when we run out of instructions?
                        // NO, WASM functions end with `0x0B` too (implicit block of function body).
                        // So we WILL hit this case.

                        if self.call_sp > 0 {
                            let frame = self.call_stack[self.call_sp - 1];
                            if self.control_sp == frame.prev_control_sp {
                                // We reached the end of the called function
                                // Return from internal call
                                self.call_sp -= 1;
                                self.pc = frame.return_pc;
                                // control_sp is already at prev_control_sp, so we are good!
                            }
                        }
                    } else {
                        // End of top-level execution (entry point)
                        // (Only happens if control_sp was 0 and we decremented? No, control_sp > 0 check prevents that)
                        // Wait, if control_sp == 0, we shouldn't be here for `0x0B` unless it's unmatched?
                        // The initial function body acts like a block?
                        // Standard WASM: function body is a block.
                        // Our `execute_loop` just runs. We don't push an initial block for the function body.
                        // So if we see `0x0B` and `control_sp == 0`, it's the end of the function.

                        if self.call_sp > 0 {
                            // Return from internal call
                            self.call_sp -= 1;
                            let frame = self.call_stack[self.call_sp];
                            self.control_sp = frame.prev_control_sp; // Restore control stack
                            self.pc = frame.return_pc;
                        } else {
                            // End of entry point
                            if self.sp > 0 {
                                let val = self.pop()?;
                                return Ok(val as i32);
                            } else {
                                return Ok(0);
                            }
                        }
                    }
                }
                0x0C => {
                    // br (branch)
                    let (depth, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;

                    if depth as usize >= self.control_sp {
                        return Err("Branch depth out of bounds");
                    }

                    // Jump to the return_pc of the frame at depth
                    let target_idx = self.control_sp - 1 - depth as usize;
                    let frame = self.control_stack[target_idx];

                    // Stack unwinding for branch:
                    // A branch out of a block/loop pops values from the stack until it matches the height
                    // expected at the destination (plus return values if any).
                    // For NanoWasm, let's just restore to entry height for now (simplified).
                    // Real WASM is more complex here (handling results).
                    while self.sp > frame.stack_height_at_entry {
                        self.sp -= 1; // Drop values
                    }

                    // If it's a loop, jump to start; otherwise jump to end
                    if frame.is_loop {
                        self.pc = frame.return_pc; // Jump to loop start
                    } else {
                        self.pc = frame.return_pc; // Jump to block end
                                                   // Pop all frames up to and including the target
                        self.control_sp = target_idx;
                    }
                }
                0x0D => {
                    // br_if (conditional branch)
                    let (depth, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let cond = self.pop()?;

                    if cond != 0 {
                        // Branch
                        if depth as usize >= self.control_sp {
                            return Err("Branch depth out of bounds");
                        }
                        let target_idx = self.control_sp - 1 - depth as usize;
                        let frame = self.control_stack[target_idx];

                        // Simplified stack unwind
                        while self.sp > frame.stack_height_at_entry {
                            self.sp -= 1;
                        }

                        if frame.is_loop {
                            self.pc = frame.return_pc;
                        } else {
                            self.pc = frame.return_pc;
                            self.control_sp = target_idx;
                        }
                    }
                    // else: continue normally
                }
                0x0F => {
                    // return
                    // Pop all control frames and return from function

                    if self.call_sp > 0 {
                        self.call_sp -= 1;
                        let frame = self.call_stack[self.call_sp];
                        self.control_sp = frame.prev_control_sp; // Restore control stack to caller's state
                        self.pc = frame.return_pc;
                    } else {
                        self.control_sp = 0; // Clear control stack for top level
                        if self.sp > 0 {
                            let val = self.pop()?;
                            return Ok(val as i32);
                        } else {
                            return Ok(0);
                        }
                    }
                }

                // Variable Instructions
                0x20 => {
                    // local.get
                    let (idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    if (idx as usize) < LOCALS_SIZE {
                        self.push(self.locals[idx as usize])?;
                    } else {
                        return Err("Local index out of bounds");
                    }
                }
                0x21 => {
                    // local.set
                    let (idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop()?;
                    if (idx as usize) < LOCALS_SIZE {
                        self.locals[idx as usize] = val;
                    } else {
                        return Err("Local index out of bounds");
                    }
                }
                0x22 => {
                    // local.tee
                    let (idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.stack[self.sp - 1];
                    if (idx as usize) < LOCALS_SIZE {
                        self.locals[idx as usize] = val;
                    } else {
                        return Err("Local index out of bounds");
                    }
                }
                0x23 => {
                    // global.get
                    let (idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    if (idx as usize) < GLOBALS_SIZE {
                        self.push(self.globals[idx as usize])?;
                    } else {
                        return Err("Global index out of bounds");
                    }
                }
                0x24 => {
                    // global.set
                    let (idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop()?;
                    if (idx as usize) < GLOBALS_SIZE {
                        self.globals[idx as usize] = val;
                    } else {
                        return Err("Global index out of bounds");
                    }
                }

                // Numeric Instructions (i32)
                0x41 => {
                    // i32.const
                    let (val, len) = Self::read_i32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    self.push(val as u64)?;
                }
                0x44 => {
                    // f64.const
                    // Use checked arithmetic to prevent overflow
                    if self.pc > self.input.len().saturating_sub(8) {
                        return Err("Unexpected end of input in f64.const");
                    }
                    let end = self.pc + 8;
                    if end > self.input.len() {
                        return Err("Unexpected end of input in f64.const");
                    }
                    let bytes = [
                        self.input[self.pc],
                        self.input[self.pc + 1],
                        self.input[self.pc + 2],
                        self.input[self.pc + 3],
                        self.input[self.pc + 4],
                        self.input[self.pc + 5],
                        self.input[self.pc + 6],
                        self.input[self.pc + 7],
                    ];
                    let val = f64::from_le_bytes(bytes);
                    self.pc = end;
                    self.push_f64(val)?;
                }
                0x45 => {
                    // i32.eqz
                    let v = self.pop()? as i32;
                    self.push(if v == 0 { 1 } else { 0 })?;
                }
                0x46 => {
                    // i32.eq
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 == v2 { 1 } else { 0 })?;
                }
                0x4A => {
                    // i32.gt_u (unsigned greater-than)
                    let v2 = self.pop()? as u32;
                    let v1 = self.pop()? as u32;
                    self.push(if v1 > v2 { 1 } else { 0 })?;
                }
                0x4B => {
                    // i32.gt_s (signed greater-than)
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 > v2 { 1 } else { 0 })?;
                }
                0x6A => {
                    // i32.add
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(v1.wrapping_add(v2) as u64)?;
                }
                0x6B => {
                    // i32.sub
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(v1.wrapping_sub(v2) as u64)?;
                }
                0x6C => {
                    // i32.mul
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(v1.wrapping_mul(v2) as u64)?;
                }
                0x6D => {
                    // i32.div_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    if v2 == 0 {
                        return Err("Division by zero");
                    }
                    self.push(v1.wrapping_div(v2) as u64)?;
                }
                0x6F => {
                    // i32.rem_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    if v2 == 0 {
                        return Err("Remainder by zero");
                    }
                    self.push(v1.wrapping_rem(v2) as u64)?;
                }
                0x71 => {
                    // i32.and
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push((v1 & v2) as u64)?;
                }
                0x72 => {
                    // i32.or
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push((v1 | v2) as u64)?;
                }
                0x73 => {
                    // i32.xor
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push((v1 ^ v2) as u64)?;
                }
                0x70 => {
                    // i32.rotl (rotate left)
                    let v2 = self.pop()? as u32;
                    let v1 = self.pop()? as u32;
                    let shift = v2 & 0x1F;
                    let result = v1.rotate_left(shift);
                    self.push(result as u64)?;
                }
                0x74 => {
                    // i32.shl
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push((v1 << (v2 & 0x1F)) as u64)?;
                }
                0x75 => {
                    // i32.shr_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push((v1 >> (v2 & 0x1F)) as u64)?;
                }
                0x47 => {
                    // i32.ne
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 != v2 { 1 } else { 0 })?;
                }
                0x48 => {
                    // i32.lt_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 < v2 { 1 } else { 0 })?;
                }
                0x4C => {
                    // i32.le_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 <= v2 { 1 } else { 0 })?;
                }
                0x4E => {
                    // i32.ge_s
                    let v2 = self.pop()? as i32;
                    let v1 = self.pop()? as i32;
                    self.push(if v1 >= v2 { 1 } else { 0 })?;
                }
                0x1A => {
                    // drop
                    let _ = self.pop()?;
                }
                0x1B => {
                    // select
                    let c = self.pop()?;
                    let v2 = self.pop()?;
                    let v1 = self.pop()?;
                    self.push(if c != 0 { v1 } else { v2 })?;
                }

                // f64 Comparisons
                0x61 => {
                    // f64.eq
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 == v2 { 1 } else { 0 })?;
                }
                0x62 => {
                    // f64.ne
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 != v2 { 1 } else { 0 })?;
                }
                0x63 => {
                    // f64.lt
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 < v2 { 1 } else { 0 })?;
                }
                0x64 => {
                    // f64.le
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 <= v2 { 1 } else { 0 })?;
                }
                0x65 => {
                    // f64.gt
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 > v2 { 1 } else { 0 })?;
                }
                0x66 => {
                    // f64.ge
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push(if v1 >= v2 { 1 } else { 0 })?;
                }

                // f64 Arithmetic (using correct opcodes 0xA0-0xA3)
                0xA0 => {
                    // f64.add
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push_f64(v1 + v2)?;
                }
                0xA1 => {
                    // f64.sub
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push_f64(v1 - v2)?;
                }
                0xA2 => {
                    // f64.mul
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push_f64(v1 * v2)?;
                }
                0xA3 => {
                    // f64.div
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    if v2 == 0.0 {
                        // WebAssembly allows division by zero (returns infinity or NaN)
                        self.push_f64(v1 / v2)?;
                    } else {
                        self.push_f64(v1 / v2)?;
                    }
                }
                0xA4 => {
                    // f64.min
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push_f64(v1.min(v2))?;
                }
                0xA5 => {
                    // f64.max
                    let v2 = self.pop_f64()?;
                    let v1 = self.pop_f64()?;
                    self.push_f64(v1.max(v2))?;
                }
                0x99 => {
                    // f64.abs
                    let v = self.pop_f64()?;
                    self.push_f64(v.abs())?;
                }
                0x9A => {
                    // f64.neg
                    let v = self.pop_f64()?;
                    self.push_f64(-v)?;
                }
                0x9B => {
                    // f64.ceil
                    let v = self.pop_f64()?;
                    self.push_f64(libm::ceil(v))?;
                }
                0x9C => {
                    // f64.floor
                    let v = self.pop_f64()?;
                    self.push_f64(libm::floor(v))?;
                }
                0x9D => {
                    // f64.trunc
                    let v = self.pop_f64()?;
                    self.push_f64(libm::trunc(v))?;
                }
                0x9E => {
                    // f64.nearest
                    let v = self.pop_f64()?;
                    self.push_f64(libm::round(v))?;
                }
                0x9F => {
                    // f64.sqrt
                    let v = self.pop_f64()?;
                    self.push_f64(libm::sqrt(v))?;
                }

                // Memory Management
                0x3F => {
                    // memory.size
                    self.pc += 1; // skip memory index 0x00
                                  // Return size in pages (64KB per page)
                    let pages = (memory_len / (64 * 1024)) as u32;
                    self.push(pages as u64)?;
                }
                0x40 => {
                    // memory.grow
                    self.pc += 1; // skip memory index 0x00
                    let _pages = self.pop()?;
                    self.push(!0)?; // -1 (u32::MAX) indicating failure to grow (fixed size)
                }

                // Memory Load/Store (i32)
                0x28 => {
                    // i32.load
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let offset_usize = offset as usize;
                    // Check for overflow before addition
                    if base > memory_len.saturating_sub(offset_usize) {
                        return Err("Memory access out of bounds");
                    }
                    let addr = base + offset_usize;
                    if addr + 4 <= memory_len {
                        let bytes = &self.memory[addr..addr + 4];
                        let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        self.push(val as u64)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x2C => {
                    // i32.load8_u (load unsigned 8-bit)
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr < memory_len {
                        let val = self.memory[addr] as u32;
                        self.push(val as u64)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x2D => {
                    // i32.load8_s (load signed 8-bit)
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr < memory_len {
                        let val = (self.memory[addr] as i8) as i32;
                        self.push(val as u64)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x2E => {
                    // i32.load16_u (load unsigned 16-bit)
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr + 2 <= memory_len {
                        let val =
                            u16::from_le_bytes([self.memory[addr], self.memory[addr + 1]]) as u32;
                        self.push(val as u64)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x2F => {
                    // i32.load16_s (load signed 16-bit)
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr + 2 <= memory_len {
                        let val =
                            i16::from_le_bytes([self.memory[addr], self.memory[addr + 1]]) as i32;
                        self.push(val as u64)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x36 => {
                    // i32.store
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop()? as u32;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr + 4 <= memory_len {
                        let bytes = val.to_le_bytes();
                        self.memory[addr] = bytes[0];
                        self.memory[addr + 1] = bytes[1];
                        self.memory[addr + 2] = bytes[2];
                        self.memory[addr + 3] = bytes[3];
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                // i32.store8
                0x3A => {
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop()? as u8;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr < memory_len {
                        self.memory[addr] = val;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                // i32.store16
                0x3B => {
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop()? as u16;
                    let base = self.pop()? as usize;
                    let addr = base + offset as usize;
                    if addr + 2 <= memory_len {
                        let bytes = val.to_le_bytes();
                        self.memory[addr] = bytes[0];
                        self.memory[addr + 1] = bytes[1];
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }

                // f64 Load/Store
                0x2B => {
                    // f64.load
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let base = self.pop()? as usize;
                    let offset_usize = offset as usize;
                    // Check for overflow before addition
                    if base > memory_len.saturating_sub(offset_usize) {
                        return Err("Memory access out of bounds");
                    }
                    let addr = base + offset_usize;
                    if addr + 8 <= memory_len {
                        let bytes = [
                            self.memory[addr],
                            self.memory[addr + 1],
                            self.memory[addr + 2],
                            self.memory[addr + 3],
                            self.memory[addr + 4],
                            self.memory[addr + 5],
                            self.memory[addr + 6],
                            self.memory[addr + 7],
                        ];
                        let val = f64::from_le_bytes(bytes);
                        self.push_f64(val)?;
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }
                0x39 => {
                    // f64.store
                    let (_align, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let (offset, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;
                    let val = self.pop_f64()?;
                    let base = self.pop()? as usize;
                    let offset_usize = offset as usize;
                    // Check for overflow before addition
                    if base > memory_len.saturating_sub(offset_usize) {
                        return Err("Memory access out of bounds");
                    }
                    let addr = base + offset_usize;
                    if addr + 8 <= memory_len {
                        let bytes = val.to_le_bytes();
                        for i in 0..8 {
                            self.memory[addr + i] = bytes[i];
                        }
                    } else {
                        return Err("Memory access out of bounds");
                    }
                }

                // f64 Conversions
                0xAA => {
                    // i32.trunc_f64_s (signed truncation)
                    let val = self.pop_f64()?;
                    // WebAssembly truncation: rounds toward zero
                    let result = libm::trunc(val) as i32;
                    self.push(result as u64)?;
                }
                0xAB => {
                    // i32.trunc_f64_u (unsigned truncation)
                    let val = self.pop_f64()?;
                    // WebAssembly truncation: rounds toward zero, then converts to unsigned
                    let truncated = libm::trunc(val);
                    if truncated < 0.0 {
                        return Err("f64 to u32 truncation: negative value");
                    }
                    let result = truncated as u32;
                    self.push(result as u64)?;
                }
                0xB7 => {
                    // f64.convert_i32_s (signed conversion)
                    let val = self.pop()? as i32;
                    self.push_f64(val as f64)?;
                }
                0xB8 => {
                    // f64.convert_i32_u (unsigned conversion)
                    let val = self.pop()? as u32;
                    self.push_f64(val as f64)?;
                }

                // Host Call / Internal Call
                0x10 => {
                    // call
                    let (func_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                    self.pc += len;

                    let import_count = self.count_imported_functions();

                    if func_idx < import_count {
                        // Imported Function (Host Call)
                        // For now only support func_idx 0 as print
                        if func_idx == 0 {
                            let len = self.pop()? as usize;
                            let ptr = self.pop()? as usize;
                            self.handle_print(ptr, len)?;
                        } else {
                            return Err("Unknown imported function");
                        }
                    } else {
                        // Internal Function Call
                        let internal_idx = func_idx - import_count;

                        // Save Context
                        if self.call_sp >= MAX_CALL_DEPTH {
                            return Err("Call stack overflow");
                        }

                        self.call_stack[self.call_sp] = CallFrame {
                            return_pc: self.pc,
                            prev_control_sp: self.control_sp,
                            locals_base: 0, // TODO: Implement locals window
                        };
                        self.call_sp += 1;

                        // Jump to function body
                        let target_pc = self.find_function_start(internal_idx)?;
                        self.pc = target_pc;

                        // Reset control stack for new function (it will rebuild)
                        // Actually, we need to preserve the old control stack somewhere if we want to be correct,
                        // or just say "control stack is per function".
                        // NanoWasm Strategy: Just reset control_sp, as CallFrame saves the old state?
                        // Actually, control_stack is a fixed array. We can't easily "push" the whole stack.
                        // But since we return to a PC, we can rebuild? No.
                        // CORRECT: We need to save/restore control_sp. But the *content* of control_stack below sp is needed?
                        // No, because we only jump *within* the function.
                        // So yes, effectively control_sp becomes 0 for the new function.
                        // But we must restore the old control_sp on return.
                        // AND the CallFrame should probably track that the old control frames are "frozen".
                        // Current implementation of `control_stack` is global.
                        // If we reset `control_sp = 0`, we overwrite the caller's control frames if we are not careful.
                        // FIX: Make control_stack a stack of stacks? Or just execute in place?
                        // Simplified: We just assume control_sp starts at 0 for the new function,
                        // BUT we must be careful not to clobber.
                        // Actually, the `control_stack` array is global.
                        // We should really increment `control_sp` relative to a base?
                        // Let's just fail if we try to recurse too deep with control structures.
                        // For NanoWasm: We will just let control_sp grow?
                        // No, `return` instruction needs to know where to stop.
                        // Let's set a "frame base" for control stack.

                        // Temporary: Reset control_sp to 0 is WRONG because it loses caller context if we share the array.
                        // If we share the array, we just append?
                        // Yes, append. But `return` needs to pop back to `prev_control_sp`.
                        // The `CallFrame` already has `prev_control_sp`.
                        // So we just keep adding to `control_stack`.
                        // BUT `control_stack` is limited size.

                        // Re-initialize locals for the called function
                        // (Scanning locals declaration)
                        let (local_vec_count, len_bytes) =
                            Self::read_u32_leb128(&self.input[self.pc..]);
                        self.pc += len_bytes;

                        // TODO: These overwrite global locals!
                        // This proves we need a locals stack or window.
                        // For now, we just WARN/Overwrite and accept it breaks caller locals.
                        // "Nano" constraints.
                        let mut local_idx = 0;
                        for _ in 0..local_vec_count {
                            let (count, len_bytes) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len_bytes;
                            self.pc += 1; // type
                            for _ in 0..count {
                                if local_idx < LOCALS_SIZE {
                                    self.locals[local_idx] = 0;
                                    local_idx += 1;
                                }
                            }
                        }
                    }
                }

                // Extended opcodes (prefix 0xFC)
                0xFC => {
                    if self.pc >= self.input.len() {
                        return Err("Unexpected end of input after 0xFC prefix");
                    }
                    let extended_opcode = self.input[self.pc];
                    self.pc += 1;

                    match extended_opcode {
                        0x02 => {
                            // memory.copy
                            // memory.copy: (dst: i32, src: i32, n: i32) -> ()
                            // Reads: memory index (u32), then dst, src, n from stack
                            let (_mem_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len;

                            let n = self.pop()? as usize;
                            let src = self.pop()? as usize;
                            let dst = self.pop()? as usize;

                            let memory_len = self.memory.len();
                            if dst + n <= memory_len && src + n <= memory_len {
                                // Use copy_from_slice for safe copying (handles overlapping regions)
                                if dst <= src {
                                    // Forward copy
                                    self.memory.copy_within(src..src + n, dst);
                                } else {
                                    // Backward copy (for overlapping regions)
                                    for i in (0..n).rev() {
                                        self.memory[dst + i] = self.memory[src + i];
                                    }
                                }
                            } else {
                                return Err("Memory copy out of bounds");
                            }
                        }
                        0x03 => {
                            // memory.fill
                            // memory.fill: (dst: i32, val: i32, n: i32) -> ()
                            // Reads: memory index (u32), then dst, val, n from stack
                            let (_mem_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len;

                            let n = self.pop()? as usize;
                            let val = self.pop()? as u8;
                            let dst = self.pop()? as usize;

                            let memory_len = self.memory.len();
                            if dst + n <= memory_len {
                                // Use explicit loop for no_std compatibility
                                for i in 0..n {
                                    self.memory[dst + i] = val;
                                }
                            } else {
                                return Err("Memory fill out of bounds");
                            }
                        }
                        0x01 => {
                            // data.drop
                            // data.drop: drops a data segment (no-op for our interpreter)
                            let (_segment_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len;
                            // No-op: we don't track data segments separately
                        }
                        0x0B => {
                            // table.init
                            // table.init: initializes a table from an element segment
                            // For Nano interpreter, we don't support tables (indirect calls), so this is a no-op
                            // Format: table.init elem_idx, then stack: dst, src, n
                            let (_elem_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len;
                            // Pop: dst, src, n from stack (but we ignore them since we don't support tables)
                            let _n = self.pop()?;
                            let _src = self.pop()?;
                            let _dst = self.pop()?;
                            // No-op: tables not supported in Nano interpreter
                        }
                        0x0C => {
                            // elem.drop
                            // elem.drop: drops an element segment (no-op for our interpreter)
                            let (_elem_idx, len) = Self::read_u32_leb128(&self.input[self.pc..]);
                            self.pc += len;
                            // No-op: we don't track element segments separately
                        }
                        _ => {
                            // Log unsupported opcode (simplified - no formatting in no_std)
                            self.log("Unsupported extended opcode");
                            return Err("Unsupported extended opcode");
                        }
                    }
                }

                _ => {
                    // Log unsupported opcode (simplified - no formatting in no_std)
                    self.log("Unsupported opcode");
                    return Err("Unsupported opcode");
                }
            }
        }

        Ok(0)
    }

    fn find_function_start(&self, internal_idx: u32) -> Result<usize, &'static str> {
        let (mut code_ptr, _code_size) = self.find_section(10)?;
        let (count, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
        code_ptr += len_bytes;

        if internal_idx >= count {
            return Err("Function index out of bounds");
        }

        for _ in 0..internal_idx {
            let (body_size, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
            code_ptr += len_bytes + body_size as usize;
        }

        let (_body_size, len_bytes) = Self::read_u32_leb128(&self.input[code_ptr..]);
        Ok(code_ptr + len_bytes)
    }

    fn handle_print(&self, ptr: usize, len: usize) -> Result<(), &'static str> {
        let memory_len = self.memory.len();
        // Check if the string is in the Data Section memory
        if ptr + len <= memory_len {
            let slice = &self.memory[ptr..ptr + len];
            if let Ok(s) = core::str::from_utf8(slice) {
                self.log(s);
                return Ok(());
            }
        }

        self.log("Invalid Print");
        Ok(())
    }
}
