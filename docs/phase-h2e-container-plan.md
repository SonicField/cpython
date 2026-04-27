# Phase H2-E: DeoptBase Container Conversion (C++ → Pure C)

> **SUPERSEDED BY COMPLETION** (April 2026). The per-field DeoptBase
> container conversion landed: `descr_` → `char *descr` (E1),
> `live_regs_` → `void *live_regs_data; size_t live_regs_count;
> size_t live_regs_cap;` PhxRegStateArray (E2), and `frame_state_`
> → raw `void *frame_state` (E3) — see `HIR_DEOPT_FIELDS` in
> `Python/jit/hir/hir_instr_c.h:123` and the `hir_c_destroy_instr_impl`
> dispatcher in the same header for the resulting pure-C destroy
> path. Plan retained for the per-phase mixed-mode destruction
> rationale + ordering analysis.

## Goal
Convert DeoptBase's three C++ container fields to C equivalents, enabling
truly pure C instruction allocation and destruction.

## Current State (Post H2-B)
- ~136 factories use pure C allocation (hir_c_alloc_instr + hir_c_init_deopt)
- hir_c_init_deopt uses placement new for std::vector and std::string (#ifdef __cplusplus)
- Instr::Destroy (C++) handles destruction via opcode-dispatched ~DeoptBase

## Strategy: Per-Field Incremental (NOT Big-Bang)
Convert one container type at a time across ALL DeoptBase instructions.
Each phase is independently testable and gateable.

### Phase H2-E1: descr_ (std::string → char*)
- Access sites: ~55
- C replacement: char* (strdup to assign, free to destroy)
- Init: NULL (calloc-safe)
- Assign: `free(old); new = strdup(src);`
- Read: direct pointer (no .c_str() needed)
- Destroy: `free(descr_)`
- Update DeoptBase destructor: free(descr_) instead of ~string()
- Remove placement new for string from hir_c_init_deopt
- Estimate: 1 session

### Phase H2-E2: live_regs_ (std::vector<RegState> → PhxArray)
- Access sites: ~65
- C replacement: `struct { RegState* data; size_t count; size_t capacity; }`
- Init: {NULL, 0, 0} (calloc-safe)
- Push: realloc when count==capacity, data[count++] = item
- Iterate: `for(i=0; i<count; i++) data[i]`
- Destroy: `free(data)`
- PhxArray pattern already proven in LIR Phase B
- Remove placement new for vector from hir_c_init_deopt
- Estimate: 1-2 sessions

### Phase H2-E3: frame_state_ (std::unique_ptr<FrameState> → FrameState*)
- Access sites: TBD (grep for frame_state_ in DeoptBase callers)
- C replacement: raw FrameState* (malloc'd, explicit free)
- Init: NULL (calloc-safe)
- Assign: `ptr = malloc(sizeof(FrameState)); memcpy(ptr, src, sizeof(FrameState));`
- Destroy: `free(frame_state_)`
- Estimate: 1 session

### Intermediate Destroy Logic (Mixed C/C++ Cleanup)
During per-field migration, DeoptBase destructor handles mixed representations:
```cpp
// After H2-E1 (descr_ → char*), before H2-E2:
~DeoptBase() {
    if (descr_) free(descr_);           // E1: C cleanup
    live_regs_.~vector();                // Still C++ until E2
    frame_state_.reset();                // Still C++ until E3
}
// After H2-E2 (live_regs_ → PhxArray):
~DeoptBase() {
    if (descr_) free(descr_);           // E1: C cleanup
    free(live_regs_.data);              // E2: C cleanup
    frame_state_.reset();                // Still C++ until E3
}
// After H2-E3 (frame_state_ → raw ptr):
void hir_c_destroy_deopt(void *instr) {  // Pure C!
    HirDeoptLayout *d = (HirDeoptLayout *)instr;
    if (d->descr) free(d->descr);
    if (d->live_regs.data) free(d->live_regs.data);
    if (d->frame_state) free(d->frame_state);
}
```
Each field's cleanup matches its CURRENT representation. Mixed mode is safe
because field cleanups are independent (free() and ~vector() don't interact).

### After All Three Phases:
- hir_c_init_deopt becomes pure C (no #ifdef __cplusplus, no placement new)
- Instr::Destroy becomes convertible to C (hir_c_destroy_deopt)
- All DeoptBase factories are truly pure C end-to-end

## Key Invariants
- Each phase converts ONE field type across ALL DeoptBase instructions
- Mixed state is safe: unconverted fields use placement new, converted fields use C
- DeoptBase destructor updated per phase to handle the C replacement
- 980/980 x86_64 + 13/13 ARM64 pydebug gate on every commit

## Dependencies
- H2-E1/E2/E3 are independent of each other (can be done in any order)
- All three must complete before hir_c_init_deopt can drop #ifdef __cplusplus
- Instr::Destroy C conversion comes AFTER all three phases

## Estimate
- H2-E1 + E2 + E3: 3-4 sessions
- Destroy C conversion: 1 session
- Total: 4-5 sessions
