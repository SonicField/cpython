# Phoenix-ASM Status Report

## How We Got Here

### Terminal Goal
Extract the CinderX JIT from Meta's CPython fork into upstream CPython 3.12.13. ARM64 + Intel. Pure C rewrite — zero C++ dependencies.

### Session Journey (2026-03-30 to 2026-03-31)

**Phase 2 Bug Fixes (session start):**
- Fixed 3 ASAN-verified memory safety bugs:
  1. `static PyModuleDef` — stack-use-after-return in jit::initialize() (f8564ccc2b)
  2. Generator dealloc routing — free-list allocator mismatch for deopted JIT generators (f772800991)
  3. `contains()` guard — routing non-pool generators to CPython dealloc (d98c7f90a1)
- Fixed inline exception handler deopt stack mismatch — JIT's exception handler skipped PUSH_EXC_INFO but deopt snapshot lacked prev_exc placeholder (28b4ee14b3)
- Result: 472-473/480 CPython tests pass, zero JIT regressions, ASAN quarantine=0 clean

**Phase 3A: phoenix-asm Design (this session):**
- Architect (theologian) designed 6-chunk implementation with dependency ordering
- Cataloged complete instruction surface: 52 x86_64 + 55 ARM64 instructions
- Identified arch.h as the abstraction seam — CRTP wrapper approach (Approach A)

**Phase 3A: phoenix-asm Implementation (this session):**
- Chunk 1: Core data structures + node-list infrastructure (common.c, 443 lines)
- Chunk 2: x86_64 encoding backend (x86_64.c, 2088 lines, 52 instructions)
- Chunk 3: ARM64 encoding backend (arm64.c, 2294 lines, 55 instructions)
- Chunk 4: Finalize pass — 3-pass label resolution for both architectures
- Chunk 5: RWX memory allocator (alloc.c, 386 lines)
- Chunk 6: C++ CRTP wrapper (phoenix_asm_wrapper.h 990 lines, asmjit_compat.h 200 lines)
- Total: ~8500 lines of new code (7500 C + 1000 C++)

**Phase 3B: Integration (this session):**
- x86_64: All 8 verification gates pass (473/480 tests, 56/56 byte-identical encoding, ASAN clean, benchmark parity)
- ARM64: Gates 1-5 pass on devgpu004 (14/14 test modules), Gate 6 in progress

## Current State

### What Works
- phoenix-asm compiles and produces correct executable code on both x86_64 and ARM64
- 56/56 byte-identical encoding match against asmjit on x86_64
- 473/480 CPython tests pass on x86_64 with phoenix-asm (exceeds asmjit baseline of 472)
- 14/14 ARM64 test modules pass on devgpu004 (Gate 6 full suite in progress)
- JIT initializes correctly: deopt trampolines emitted via phoenix-asm on both architectures
- JIT-compiled functions execute correctly on both architectures (verified: returns correct results)
- Benchmark harness (Tools/benchmark_phoenix.py) works with phoenix-asm, shows performance parity with asmjit
- End-to-end execution verified: assemble → finalize → allocate to RWX → execute → correct result
- All stubs implemented: ptr(Label), setSegment, shifted registers — zero abort-guarded paths remain

### What's Verified
- x86_64 encoding: 56/56 byte-identical + 48-entry static reference corpus (survives asmjit deletion)
- ARM64 encoding: compiles clean, 14/14 modules pass, 103-entry corpus spec ready (pending generation)
- ASAN quarantine=0: 17/17 PASS on x86_64 (zero use-after-free)
- Fallback toggle: PHOENIX_ASM=OFF builds cleanly (asmjit path preserved for corpus generation)
- CPython version on devgpu004: 3.12.13 (independently verified by testkeeper)
- Compiler on devgpu004: CC=clang CXX=clang++ (verified from Makefile)

### What's Pending
- ARM64 Gate 6 (full subprocess suite) — running now
- ARM64 Gates 7-8 (ASAN + byte-comparison corpus)
- ARM64 ABBA benchmarks against cinderx_dev
- Phase 3C-3E: C++ to C conversion of codegen files (planned, not started at scale)

## Next Steps (Phase 3C-3E: Zero C++)

### Conversion Order (instruction-emitting files first)
1. register_preserver.cpp → .c (prototyped, compiles standalone)
2. gen_asm_utils.cpp → .c (6 refs)
3. frame_asm.cpp → .c (27 refs, prologue/epilogue)
4. code_allocator.cpp → .c (partially C already with PhxRuntime)
5. annotations.cpp → .c (STL-heavy, debug infrastructure)
6. code_section.cpp → .c (minimal)
7. debug_info.cpp → .c (debug infrastructure)
8. gen_asm.cpp → .c (largest codegen file)
9. autogen.cpp → .c (CRTP templates → C function pointer tables, hardest)

### Key Architectural Constraint
- Environ struct (codegen/environ.h) is shared by all codegen files. Contains C++ types (std::vector, UnorderedMap). Must become C-compatible (PhxEnviron) for full conversion.
- Hybrid approach works for interim: keep .cpp, replace as_-> calls with phx_ C API calls internally.
- ASM() macro in autogen.cpp is autogen-only (verified by grep). Other files use direct as_-> calls. Incremental conversion is feasible for 8 of 9 files.

### Estimated Effort (AI velocity)
- Phase 3C (9 codegen files): 1-2 sessions
- Phase 3D (30+ JIT infrastructure files): 3-5 sessions
- Phase 3E (cleanup): 1 session
- Total: ~5-8 sessions

## Gotchas

### Build System
- CMake conditional: `-DPHOENIX_ASM=ON` excludes asmjit, includes phoenix-asm .c files
- Must `rm -f ./python` to force relink when .a changes (Make doesn't detect)
- Stale .o files: `find CMakeFiles/jit.dir -name '*.o' -delete` before clean rebuild
- devgpu004 requires CC=clang CXX=clang++ for ARM64 builds

### Encoding
- x86_64 RSP/R12 as base register: always needs SIB byte (handled)
- x86_64 RBP/R13 with offset=0: must use mod=01 with disp8=0 to avoid RIP-relative (handled)
- ARM64 logical immediates: bitmask encoding is complex (~30 lines), easy to get wrong
- ARM64 has no RIP-relative memory: use ADR (PC-relative address to register) instead
- ARM64 pre/post-indexed LDP/STP: different bit patterns from signed-offset form (bits [24:23])
- Accumulator short-forms (AND/TEST/XCHG with RAX): optional but needed for byte-identical encoding

### Runtime
- Shutdown SEGV: GC runs after JIT teardown, touching freed state. Dict watcher cleanup needs guard (watchers.cpp:81,89)
- NDEBUG strips assert(): all stubs must use fprintf+abort(), not assert(false)
- ARM64 requires __builtin___clear_cache() after copying code to RWX memory (icache coherence)
- Threshold=1000 masks codegen bugs: functions called <1000 times run interpreted, hiding JIT bugs

### Process
- Agent fabrication pattern: 8 incidents of agents claiming verification without reading files. Mitigation: require raw command + output for all test reports (Alex's directive). Medic catches post-hoc.
- Triple-duplication: agents independently start the same task without checking if someone else was assigned. Mitigation: ASSIGNED: prefix in chat messages.
- Cursor desync: agents miss recent messages due to --unread returning empty. Use --last=N as fallback.

## Lessons Learned

1. **Test-driven validation is the real gate.** Code review was compromised by fabrication (8 incidents). The 473/480 test result is unfabricatable — it is the ground truth. Design the process so tests, not reviews, are the gate.

2. **Byte-comparison is essential for assembler work.** Without byte-identical comparison against asmjit, encoding bugs are invisible until they manifest as codegen corruption — which looks identical to logic bugs. The 56/56 harness caught 3 mismatches (accumulator short-forms) before they reached production.

3. **Preserve the oracle before deleting it.** The x86_64 reference corpus (48 entries) was generated while asmjit was still available. ARM64 corpus (103 entries) must be generated before asmjit deletion. Once the oracle is gone, it's gone.

4. **ASAN finds what static analysis cannot.** ASAN quarantine=0 found the exception chain corruption in seconds. Static analysis of the deopt path was thorough but wrong — corruption originated in module init, not deopt.

5. **The arch.h abstraction seam worked.** Zero changes to 9 codegen logic files during the entire phoenix-asm integration. The CRTP wrapper absorbed all API differences. This validated the architecture.

6. **Silent stubs are the worst failure mode.** ptr(Label) returning empty Mem and JitRuntime::_add returning kErrorOk produced silent corruption, not crashes. assert(false) was stripped by -DNDEBUG. Only fprintf+abort() traps reliably.

7. **Node-list assembler was the right architecture.** The JIT emits code non-linearly (body first, then prologue via setCursor). A flat byte buffer cannot support this. The doubly-linked node list with deferred finalization was essential.

8. **The wrapper is transition scaffolding, not permanent infrastructure.** Every feature routed through the wrapper makes removal harder. Convert codegen files to C incrementally, starting with leaf files. The wrapper shrinks as files convert.
