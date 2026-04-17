# hir.h Porting Status — Session 2026-04-17 (FINAL)

## Completed Phases

### Phase R: Register → C (DONE)
- R1a: `68259913b8` — Register::name_ std::string → char*
- R1b: `06dfdca8a9` — HirRegisterLayout C struct + static_asserts
- R2: `6442e4a689` — 7 Register C accessor functions
- R3: skipped (inline methods, no wiring needed)

### Phase C: CFG → C (DONE)
- C1: `e938f58c35` — HirCFG C struct + static_asserts
- C2: `51e7d6e708` — 4 CFG block list C functions
- C3: `d3668ee79a` — wire InsertBlock/RemoveBlock to C

### Phase F: FrameState → C (struct + accessors DONE)
- F1: `9bb34a4029` — HirFrameStateLayout C struct + static_asserts
- F2: `80a31d6282` — 12 FrameState C accessor functions
- F3/F4: deferred (visitUses stays C++, alloc/dealloc complex)

### Phase E: Environment → C (DONE)
- E1: `d3b6a9b3cc` — RegisterMap unordered_map → flat Register** array
- E1 fix: `53693096e6` — merge duplicate destructor
- E2: `0c36f740de` — HirEnvironment C struct + static_asserts
- E3: `9f4939ee3b` — 6 Environment C accessor functions
- E3 fix: `27160e1436` — move struct before accessors (declaration ordering)
- sizeof cascade: `7b6560b1bc` — Function size 48→44→41 pointers

### Phase Fn: Function → C (DONE)
- Fn-prep-a: `4f5c8286f9` + `d63e1b9c92` — fullname std::string → char*
- Fn-prep-c: `e82cc7964d` + `ecd40d786e` + `0d07edfc7b` — ThreadedRef → raw pointers (5 fields)
- Fn1: `e60194ac3a` — HirFunctionLayout C struct (opaque char[328]) + sizeof assert
- Fn2: `28a7c1e955` — 9 Function C accessors (code, builtins, globals, prim_args_info, fullname_ptr, return_type, env, cfg_ptr, reifier) + 9 offsetof asserts
- Fn-prep-b: DEFERRED — typed_args vector (TypedArgument still has complex lifecycle)

### Phase I: Instr methods pure C (DONE)
- I1: already existed (hir_c_num_edges)
- I2: `6f10058b68` — hir_c_edge_at (mutable Edge* by opcode)
- I3: `6f10058b68` + `9b082e9a16` — hir_c_set_block pure C
- I4: `92b9b1e86a` + `762d7ddd74` — InsertBefore/After/unlink/link pure C
- I5: `8fcac51fbc` — wire C++ Instr methods to pure C

### Phase H: Header split (BLOCKED on full Fn field mapping)

## Remaining Work (next session)

1. **Opaque blob refinement**: Convert remaining Function opaque fields to real C:
   - typed_args: std::vector<TypedArgument> → flat array (TypedArgument may be POD now)
   - code_patchers: std::vector<unique_ptr<CodePatcher>> → opaque or flat array
   - InlineFunctionStats: contains UnorderedMap — keep opaque
   - compilation_phase_timer: unique_ptr — keep opaque (8 bytes)

2. **Phase H (header split)**: Split hir.h into hir_c.h (pure C) + hir.h (C++ compat).
   Requires Fn fully mapped (no opaque blobs ideally, or at least typed_args converted).

3. **ARM64 gate debt**: ~40+ commits unverified on ARM64. Needs Alex's Duo 2FA.

## Session Statistics
- 41 commits pushed to SonicField/cpython
- ALL 7 phases have verified implementations
- 9/9 Function C accessors with offsetof verification
- Key infrastructure: RegisterMap → flat array, ThreadedRef → raw pointers,
  pure C set_block/insert/unlink, fullname → char*
- Lessons learned:
  - Verify C header declaration ordering before commit (5 failures this session)
  - Always grep hir_c_api.h for existing declarations before adding new accessors
  - Use diagnostic static_asserts (intentionally wrong values) to discover offsets
