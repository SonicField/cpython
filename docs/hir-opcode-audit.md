# HIR Opcode Audit — Phase H1a

Date: 2026-04-14
Source: Python/jit/hir/hir.h (4132 lines, 168 opcodes)

## Summary

- **82 SIMPLE** (49%): DEFINE_SIMPLE_INSTR, no custom fields
- **76 COMPLEX with custom fields** (45%): mostly POD (enums, ints, bools, Type, pointers)
- **10 COMPLEX without custom fields** (6%): full class definition but no extra data
- **HasOutput**: 129 (77%)
- **DeoptBase**: 89 (53%)
- **Variadic operands**: 17 (10%)

## C++ Features (non-POD)

Only 8 opcodes use C++ containers:
- `std::string`: LoadField, StoreField, BeginInlinedFunction
- `std::vector`: Phi (BasicBlock*), HintType (ProfiledTypes)
- `std::unique_ptr`: Snapshot, BeginInlinedFunction, FillTypeAttrCache
- `BorrowedRef`: BeginInlinedFunction, LoadGlobalCached

## Union Design Impact

- ~132 opcodes need 0-8 bytes of per-opcode data (trivial union entry or none)
- ~28 opcodes need 9-24 bytes (2-3 POD fields)
- ~8 opcodes need special handling (C++ containers → C equivalents)
- Union size: ~32 bytes (dominated by BeginInlinedFunction)
- Total HirInstr: 40 (base) + 32 (union) = 72 bytes

## Full Opcode Table

Format: OPCODE | OPS | OUT | DEOPT | CUSTOM_FIELDS | CPP

BeginInlinedFunction | 0 | N | N | func_:BorrowedRef, reifier_:BorrowedRef, caller_state_:unique_ptr<FrameState>, fullname_:string | BorrowedRef,string,unique_ptr
BinaryOp | 2 | Y | Y | op_:BinaryOpKind | POD
BitCast | 1 | Y | N | type_:Type | POD
Branch | 0 | N | N | edge_:Edge | POD
BuildInterpolation | 3 | Y | Y | conversion_:int | POD
CallCFunc | var | Y | N | func_:Func | POD
CallEx | 3 | Y | Y | flags_:CallFlags | POD
CallInd | var | Y | Y | name_:char*, ret_type_:Type | POD
CallIntrinsic | var | Y | N | index_:size_t | POD
CallMethod | var | Y | Y | flags_:CallFlags | POD
CallStatic | var | Y | N | addr_:void*, ret_type_:Type | POD
CallStaticRetVoid | var | N | N | addr_:void* | POD
Cast | 1 | Y | Y | pytype_:PyTypeObject*, optional_:bool, exact_:bool | POD
Compare | 2 | Y | Y | op_:CompareOp | POD
CompareBool | 2 | Y | Y | op_:CompareOp | POD
CondBranchCheckType | 1 | N | Y | type_:Type | POD
ConvertValue | 1 | Y | Y | converter_idx_:int | POD
DeoptPatchpoint | 0 | N | Y | patcher_:JumpPatcher* | POD
DoubleBinaryOp | 2 | Y | N | op_:BinaryOpKind | POD
EndInlinedFunction | 0 | N | N | begin_:BeginInlinedFunction*, inline_depth_:int | POD
FillTypeAttrCache | 1 | Y | Y | cache_id_:int | POD
FillTypeMethodCache | 1 | Y | Y | cache_id_:int | POD
FloatBinaryOp | 2 | Y | Y | op_:BinaryOpKind | POD
FloatCompare | 2 | Y | N | op_:CompareOp | POD
FormatValue | 2 | Y | Y | conversion_:int | POD
GetSecondOutput | 1 | Y | N | type_:Type | POD
GuardIs | 1 | Y | Y | target_:PyObject* | POD
GuardType | 1 | Y | Y | target_:Type | POD
HintType | var | N | N | types_:ProfiledTypes | std::vector
InPlaceOp | 2 | Y | Y | op_:InPlaceOpKind | POD
IndexUnbox | 1 | Y | N | exc_:PyObject* | POD
InitFrameCellVars | 1 | N | N | cells_:int | POD
IntBinaryOp | 2 | Y | N | op_:BinaryOpKind | POD
IntConvert | 1 | Y | N | type_:Type | POD
InvokeStaticFunction | var | Y | Y | func_:PyFunctionObject*, ret_type_:Type | POD
LoadArg | 0 | Y | N | arg_idx_:uint32_t, type_:Type | POD
LoadArrayItem | 3 | Y | N | offset_:ssize_t, type_:Type | POD
LoadAttr | 1 | Y | Y | already_optimized_:bool | POD
LoadAttrSpecial | 1 | Y | Y | id_:IDType*, failure_fmt_str_:char* | POD
LoadConst | 0 | Y | N | type_:Type | POD
LoadField | 1 | Y | N | name_:string, offset_:size_t, type_:Type, borrowed_:bool | std::string
LoadFunctionIndirect | 0 | Y | Y | funcptr_:PyObject**, descr_:PyObject* | POD
LoadGlobalCached | 0 | Y | N | code_:BorrowedRef, builtins_:BorrowedRef, globals_:BorrowedRef, name_idx_:int | BorrowedRef
LoadSpecial | 1 | Y | Y | special_idx_:int | POD
LoadSplitDictItem | 1 | Y | N | item_idx_:Py_ssize_t | POD
LoadTupleItem | 1 | Y | N | idx_:size_t | POD
LoadTypeAttrCacheEntryType | 0 | Y | N | cache_id_:int | POD
LoadTypeAttrCacheEntryValue | 0 | Y | N | cache_id_:int | POD
LoadTypeMethodCacheEntryType | 0 | Y | N | cache_id_:int | POD
LoadTypeMethodCacheEntryValue | 1 | Y | N | cache_id_:int | POD
LongBinaryOp | 2 | Y | Y | op_:BinaryOpKind | POD
LongCompare | 2 | Y | N | op_:CompareOp | POD
LongInPlaceOp | 2 | Y | Y | op_:InPlaceOpKind | POD
MakeCheckedDict | 0 | Y | Y | capacity_:size_t, type_:Type | POD
MakeCheckedList | var | Y | Y | type_:Type | POD
MakeDict | 0 | Y | Y | capacity_:size_t | POD
Phi | var | Y | N | basic_blocks_:vector<BasicBlock*> | std::vector
PrimitiveBox | 1 | Y | Y | type_:Type | POD
PrimitiveCompare | 2 | Y | N | op_:PrimitiveCompareOp | POD
PrimitiveUnaryOp | 1 | Y | N | op_:PrimitiveUnaryOpKind | POD
PrimitiveUnbox | 1 | Y | N | type_:Type | POD
RaiseAwaitableError | 1 | N | Y | is_aenter_:bool | POD
RaiseStatic | var | N | Y | fmt_:char*, exc_type_:PyObject* | POD
RefineType | 1 | Y | N | type_:Type | POD
Return | 1 | N | N | type_:Type | POD
SetFunctionAttr | 2 | N | N | field_:FunctionAttr | POD
StoreArrayItem | 4 | N | N | type_:Type | POD
StoreField | 3 | N | N | name_:string, offset_:size_t, type_:Type | std::string
TpAlloc | 0 | Y | Y | pytype_:PyTypeObject* | POD
UnaryOp | 1 | Y | Y | op_:UnaryOpKind | POD
UnicodeCompare | 2 | Y | N | op_:CompareOp | POD
UnpackExToTuple | 1 | Y | Y | before_:int, after_:int | POD
UpdatePrevInstr | 0 | N | N | line_no_:int, parent_:BeginInlinedFunction* | POD
UseType | 1 | N | N | type_:Type | POD
VectorCall | var | Y | Y | flags_:CallFlags | POD

## Simple Opcodes (no custom fields — 92 total)

Assign, AtQuiescentState, BatchDecref, BuildSlice, BuildString, BuildTemplate,
CIntToCBool, CheckErrOccurred, CheckExc, CheckField, CheckFreevar, CheckNeg,
CheckSequenceBounds, CheckVar, CondBranch, CondBranchIterNotDone,
CopyDictWithoutKeys, Decref, DeleteAttr, DeleteSubscr, Deopt, DictMerge,
DictSubscr, DictUpdate, EagerImportName, FormatWithSpec, GetAIter, GetANext,
GetIter, GetLength, GetTuple, Guard, ImportFrom, ImportName, Incref,
InitialYield, InvokeIterNext, IsInstance, IsNegativeAndErrOccurred, IsTruthy,
ListAppend, ListExtend, LoadAttrCached, LoadAttrSuper, LoadCellItem,
LoadCurrentFunc, LoadEvalBreaker, LoadFieldAddress, LoadFrame, LoadGlobal,
LoadMethod, LoadMethodCached, LoadMethodSuper, LoadModuleAttrCached,
LoadModuleMethodCached, LoadVarObjectSize, MakeCell, MakeFunction, MakeList,
MakeMakeSet, MakeTuple, MakeTupleFromList, MatchClass, MatchKeys,
MergeSetUnpack, PrimitiveBoxBool, Raise, RunPeriodicTasks, Send,
SetCellItem, SetCurrentAwaiter, SetDictItem, SetSetItem, SetUpdate,
Snapshot, StealCellItem, StoreAttr, StoreAttrCached, StoreSubscr,
SwapCellItem, UnicodeConcat, UnicodeRepeat, UnicodeSubscr, Unreachable,
WaitHandleLoadCoroOrResult, WaitHandleLoadWaiter, WaitHandleRelease,
XDecref, XIncref, YieldAndYieldFrom, YieldFrom,
YieldFromHandleStopAsyncIteration, YieldValue
