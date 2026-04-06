# Phase 3D Status — C++ to Pure C Conversion

## Current State (2026-04-06, HEAD=584e4ceb1b)

### Progress: ~45/104 C++ files eliminated (~43%)

| Subsystem | Status | Files Remaining |
|-----------|--------|-----------------|
| LIR rewrites | COMPLETE | 0 (postgen.cpp, postalloc.cpp, rewrite.cpp deleted) |
| LIR rewrite framework | COMPLETE | 0 (rewrite_c.h + rewrite_impl.c replace C++) |
| autogen codegen | 54% DONE | autogen.cpp: 3117→1404 lines. C dispatch on both arches |
| frame_asm codegen | 25% DONE | 3/14 methods in frame_asm_c.c |
| gen_asm orchestrator | NOT STARTED | 3530 lines, depends on frame_asm + autogen |
| LIR core types | PARTIAL | C structs + C++ wrappers coexist (Phase B) |
| Hub (compiler/context) | SCOPED | 7 hierarchies verified, ~6-7 sessions |

### Conversion Order (locked)
1. **frame_asm.cpp** (current) → standalone C functions with (env, hir_func) params
2. **autogen.cpp x86_64 ASM()** → extend C dispatch to cover x86_64 ASM macro opcodes
3. **Trie deletion** → delete C++ DSL/trie infrastructure (~400 lines) + add opcode coverage test
4. **gen_asm.cpp** → last codegen file, depends on all callees being C

### Active Files

**frame_asm_c.c** (Python/jit/codegen/frame_asm_c.c)
- 3/14 methods converted: initTStateOffset, loadTState, linkNormalGeneratorFrame
- Next: incRef (lines 327-481, ~155 lines, complex: Py_REF_DEBUG + GIL_DISABLED branches)
- Then: storeConst, linkLightWeightFunctionFrame (400 lines, hardest method)
- Each method needs 2-5 new Environ C API wrappers in lir_c_api.h/cpp
- NOT wired — all methods unwired, zero behavioral change

**autogen_translate_c.c** (Python/jit/codegen/autogen_translate_c.c)
- 32+ ARM64 translate* functions (1496 lines)
- C dispatch table handles ALL non-yield opcodes on ARM64
- x86_64 C dispatch handles: Guard, Compare, DeoptPatchpoint, IntToBool
- Remaining x86_64: ~20 ASM() macro opcodes need C equivalents
- Yield functions deferred to generator.cpp conversion

### C API Surface Built This Session

**lir_c_api.h/cpp** — Environ accessors:
- jit_environ_get_phx_builder — PhxBuilder* from Environ
- jit_environ_get_max_arg_buffer / update — arg buffer tracking
- jit_environ_is_generator / saved_ip_fp_offset — ARM64 call codegen
- jit_environ_add_pending_debug_loc — debug location registration
- jit_environ_get_code_rt — CodeRuntime* accessor
- jit_environ_add_deopt_exit — deopt exit registration
- jit_environ_get_block_label — block→label mapping for branches
- jit_environ_shadow_frames_and_spill_size — frame setup
- jit_fill_live_value_locations — live value tracking for deopts
- jit_jump_patcher_stored_bytes — patchpoint bytes
- jit_environ_add_pending_deopt_patcher — deopt patcher registration
- jit_gen_data_footer_saved_ip_offset — GenDataFooter offset
- 10 arch register constant accessors (arg_reg, fp_arg_reg, scratch, etc.)

**phoenix-asm additions:**
- phx_fs_ptr — x86_64 FS segment TLS memory operand (584e4ceb1b)

### Known Issues

1. **ASAN latent UAF** — SEGV in JIT code under ASAN (forcedJitVectorcall, rdx=freed-memory poison). Does NOT crash in production (1M stress test PASS). Fingerprint baseline at docs/asan_known_uaf_fingerprint.txt
2. **Shutdown crash trace** — ARM64 test_phoenix_benchmark_correctness prints crash on stderr during shutdown but test PASSES (exit 0). Known dict watcher cleanup ordering issue
3. **PyMem_RawCalloc** — All LIR core types (OperandBase, Instruction, BasicBlock, MemoryIndirect) use PyMem_RawCalloc via operator new override. Matches C-side PyMem_RawFree

### Standing Rules
1. Gate-before-push — no exceptions
2. Gatekeeper symbol-existence grep before APPROVE
3. Clean builds (make clean && make) for unwired C code gate checks
4. Per-commit ARM64 compile for gen_asm.cpp (29 arch-conditional blocks)
5. Tool-output-in-chat for ALL file-level verification claims
6. File-level claims from any agent are hypotheses until verified with tool output

### Test Baselines
- x86_64: 462/483 PASS (10 env/infra failures)
- ARM64: 456/486 PASS (8 env/infra + 1 shutdown trace)
- Benchmark geo-mean: 1.10x (24 benchmarks, ~500ms each)
- ASAN: 7 alloc-dealloc mismatches FIXED, known UAF fingerprinted

### Hub Conversion Plan (future)
All 7 hierarchies verified monomorphic or static dispatch:
- H1: Symbolizer + GlobalCacheManager (monomorphic, delete interface) ~1 session
- H2: Pass system (static dispatch, direct C function calls) ~1 session
- H3: ICodeAllocator, IJitContext, IJITList (monomorphic) + IJitGenFreeList (dimorphic) ~1 session
- H4: Compiler/Context core (mutual refs) ~3-4 sessions
- Total: ~6-7 sessions

### C++ Container → C Replacement Map
| C++ Type | C Replacement |
|----------|--------------|
| std::vector<T> | T* + count + capacity, PyMem_RawMalloc |
| std::unique_ptr<T> | T* + explicit free |
| UnorderedMap<K,V> | _Py_hashtable_t or fixed array |
| std::optional<T> | T + has_value flag |
| std::function<R(Args)> | R (*)(Args, void* ctx) |
| std::shared_mutex | pthread_rwlock_t |
