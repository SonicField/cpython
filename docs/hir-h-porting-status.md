# hir.h Porting Status — Session 2026-04-17

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
- sizeof cascade: `7b6560b1bc` — Function size 48→44 pointers

### Phase Fn: Function → C (PARTIAL)
- Fn-prep-a: `4f5c8286f9` + `d63e1b9c92` — fullname std::string → char*
- Fn-prep-b: DEFERRED — typed_args has ThreadedRef (RAII)
- Fn1: DEFERRED — too many opaque blobs (4x ThreadedRef, typed_args, code_patchers, InlineFunctionStats)

### Phase I: Instr methods pure C (DONE)
- I1: already existed (hir_c_num_edges)
- I2: `6f10058b68` — hir_c_edge_at (mutable Edge* by opcode)
- I3: `6f10058b68` + `9b082e9a16` — hir_c_set_block pure C
- I4: `92b9b1e86a` + `762d7ddd74` — InsertBefore/After/unlink/link pure C
- I5: `8fcac51fbc` — wire C++ Instr methods to pure C

### Phase H: Header split (BLOCKED on Fn1)

## Next Steps (for next session)

1. **ThreadedRef conversion**: Convert Function's 4 ThreadedRef<T> fields
   (code, builtins, globals, prim_args_info) to BorrowedRef<T> with explicit
   INCREF/DECREF. ~30 min. Unblocks Fn1.

2. **Fn-prep-b**: Convert typed_args std::vector<TypedArgument> to flat array.
   Requires TypedArgument ThreadedRef conversion first.

3. **Fn1**: Define HirFunction C struct once ThreadedRef fields are void*.

4. **Phase H**: Split hir.h into hir_c.h (pure C) + hir.h (C++ compat layer).
   Requires Fn1 completion.

## Session Statistics
- 32 commits pushed to SonicField/cpython
- 5/7 phases complete
- Key infrastructure: RegisterMap → flat array, pure C set_block/insert/unlink
- Lessons: verify C header declaration ordering before commit (3 consecutive failures)
