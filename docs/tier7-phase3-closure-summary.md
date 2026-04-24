# Tier 7 Phase 3 — Closure Summary

**Status:** PHASE 3 COMPLETE 2026-04-24. All 6 batches landed + pushed
to SonicField/cpython phoenix-asm-integration. Per-batch dual-arch +
per-bench floor + W45 §3.5 BUILD MODE all green.

**Phase 3 design choice (per pythia #103 + theologian 00:59:37Z +
supervisor 00:59:39Z):** Class B-kept is a TRANSITIONAL disposition,
not the final shape. Bridge surface paid as Phase 3 foundation cost;
Tier 8 pilot validates the migrate-arm by porting one Class B field
to pure C (per docs/tier8-class-b-cport-migrate-arm-spec.md).

---

## Batch ledger

| Batch | Commit | NET | Description |
|-------|--------|-----|-------------|
| 1 | c905827742 | +140L | PhxHirBuilderState foundation + parseExceptionTable port + 3 bridges |
| 2 | 1343895045 | +80L  | findExceptionHandler port + exception_table_ Class B-kept closure (read+write bridges) |
| 3 | b92d85e751 | -13L  | pending_b2_blocks_ DEAD-STATE delete (no writers post-W26 refactor) |
| 4 | b44a5143cc | +27L  | block_map_ Class B-kept closure (blocks_lookup_cpp + hir_c_api rewire) |
| 5 | 782d56d0c6 | +9L   | static_method_stack_ rename _pop_c → _pop_cpp (consistency) |
| 6 | 7971941afe | +14L  | temps_ alloc_stack rename (75-site sed) — all 5 Class B disposed |

**Cumulative NET: +257L** (foundation cost).

**Push count:** 18 → 24 today; full session 4 + 24 = 28 push events
across the multi-day workstream (inclusive of pre-Phase 3 Phase 1
burndown + W44/W45/§3.5 falsifier infra).

## §5 forcing-decision endpoint

All 5 Class B HirBuilder members disposed:

| Member | Disposition | Closure mechanism |
|--------|-------------|--------------------|
| `exception_table_` (std::vector) | CLOSED via Batch 2 | 4 bridges: push_cpp + size_cpp + entry_cpp + find_c (read+write) |
| `block_map_` (std::unordered_map ×2) | CLOSED via Batch 4 | 1 bridge: blocks_lookup_cpp + hir_c_api rewire |
| `pending_b2_blocks_` (std::vector) | DELETED via Batch 3 | dead-state, zero writers post-W26 refactor |
| `static_method_stack_` (jit::Stack) | CLOSED via Batch 5 | rename to _cpp convention |
| `temps_` (TempAllocator) | CLOSED via Batch 6 | rename + 73-site mechanical sed |

§5 keep-bias is a DESIGN CHOICE (architectural decision, not deferral).
PhxArray-equivalent C-port for std::vector / unordered_map / Stack /
TempAllocator is multi-session-per-container scope; bridging at the
opaque-pointer surface is the tractable Phase 3 endpoint.

## Foundation surface (carried to Tier 8)

9 extern surfaces shipped across B1–B6 (6 _cpp bridges + 2 _c
algorithmic + 1 init):

1. `hir_builder_state_init` (B1 ctor)
2. `hir_builder_state_parse_exception_table_c` (B1 algorithm)
3. `hir_builder_state_exception_table_push_cpp` (B1 write)
4. `hir_builder_state_exception_table_size_cpp` (B2 read)
5. `hir_builder_state_exception_table_entry_cpp` (B2 read)
6. `hir_builder_state_find_exception_handler_c` (B2 algorithm)
7. `hir_builder_state_block_map_blocks_lookup_cpp` (B4 read)
8. `hir_builder_state_static_method_stack_pop_cpp` (B5 read)
9. `hir_builder_state_temps_alloc_stack_cpp` (B6 alloc)

PhxHirBuilderState struct: 5 Class A members (code, preloader,
current_func, func, kwnames). Class B members not in struct (kept C++
behind opaque-access via the 7 _cpp bridges).

## Falsifier infrastructure landed in-flight

- **W44 caller-gate** (pre-Phase 3): DO-NOT-USE marker enforcement.
- **W45 §1–§2 bridge-sig falsifier** (pre-Phase 3): signature mutation +
  build-fail verifier. 15/15 fixtures cover Phase 1 #2–#7 + Phase 3
  Batches 1, 2, 4, 5, 6 bridges.
- **W45 §3.5 derivation-drift falsifier** (mid-Phase 3, bfc6321b77):
  4 fixtures (BEFORE_ASYNC_WITH + _Py_ID + ExceptionTableEntry.depth +
  block_map_blocks_lookup_cpp return-type). Integrated as
  gate_phoenix.sh Step 1g (eb3cdf3f54), BUILD MODE per push for
  builder*.{cpp,h,c} touch.
- **§3.5 5/5 backstop:** triggered post-Batch 4 land per
  supervisor 23:49:22Z; impl shipped same workstream.

## Per-bench floor (spec §6 #9 amendment)

4-criterion floor enforced per commit:

1. geo-mean ≥ 1.0x
2. geo-mean drop ≤ 5% same-session
3. no single bench drops > 20% same-session
4. no single bench < 0.5x absolute

Phase 3 series: 1.27x → 1.29x → 1.28x → 1.26x → 1.29x → 1.26x. Net
geo-mean change ≈ -1% across 6 batches; well within noise floor;
zero floor violations.

## Tier 8 pilot BLOCKED (push 28+, 2026-04-24)

Per `docs/tier8-class-b-cport-migrate-arm-spec.md` (theologian
01:01:50Z, supervisor 01:02:46Z ADOPTED):

- Pilot field: `exception_table_` (POD-equivalent per Phase 3 §2.2)
- Phase A execution attempted at 01:18:35Z+, HALTED at 01:23:55Z by
  external file-state revert (generalist 01:24:43Z observed-but-not-
  attributed)
- Resume gated on Alex disposition (01:25:09Z ping unanswered) +
  fixup PIR (01:25:09Z + 02:14:16Z scope assigned, no result by
  deadline)
- DEADLINE = push 28 OR session-end per supervisor 02:13:12Z +
  theologian 02:13:54Z; reached push 28 at 02:15:47Z

**Pythia #103 escape question RE-OPENED.** Phase 3 keep-bias is the
END STATE pending future migrate-arm validation. The 'transitional
foundation cost paid back' framing is HONEST-ASPIRATIONAL not
RESOLVED. Tier 8 pilot remains FILED but UNVALIDATED. ZERO-C++
terminal goal (MEMORY.md L70 + L104) remains gated on Tier 8 pilot
landing in a future session.

Phase 3 cumulative +257L is the END STATE for this session. Future
session must re-attempt Tier 8 Phase A under same spec OR honestly
amend Phase 3 closure framing to 'permanent scaffolding' if pilot is
permanently unfeasible.

## Cross-link

- Phase 3 spec: `docs/tier7-phase3-hirbuilder-state-extraction-spec.md`
  (theologian 22:59:11Z, supervisor 22:59:54Z + 23:00:15Z amendments)
- §5 amendment: keep-bias as design choice; Tier 8 migrate-arm pilot
  FILED but BLOCKED — see 'Tier 8 pilot BLOCKED' section above for
  status (replaces withdrawn "FINAL" framing per pythia #103)
- Tier 8 spec: `docs/tier8-class-b-cport-migrate-arm-spec.md`
  (theologian 01:01:50Z, supervisor 01:02:46Z ADOPTED)
- W44: `scripts/check_do_not_use_callers.sh`,
  `docs/w44-do-not-use-callers-gate.md`
- W45: `scripts/w45_bridge_drift_falsifier.sh`,
  `scripts/w45_section_3_5_derivation_drift.sh`,
  `docs/w45-bridge-signature-drift-falsifier.md` (§§1–2 + §2.5/§2.7)
- Pythia checkpoints: #89 (Phase 3 spec gate), #90/#91 (W45 trigger),
  #92 (cross-post storm), #93 (block_map writer-grep + per-bench
  floor + §3.5 trigger amendments), #94 (per-bench mechanized untested
  + §5 keep-bias risk), #103 (cathedral-scaffold six-month-regret +
  Tier 8 migrate-arm pilot demand)
- MEMORY.md terminal goal: ZERO C++ — Phase 3 is foundation toward,
  not satisfaction of
