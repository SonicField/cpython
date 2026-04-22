"""Forcing-function tests for Phoenix JIT PartialConversion bridges.

Each test corresponds to one PartialConversion bridge — a C function
extracted from a C++ method body while the surrounding C++ method body
remains C++ pending a follow-on architectural workstream.

Each test follows the pre-condition + assertion pattern:

  PRE-CONDITION (skip path)
    Verify the architectural condition that justifies the partial state
    still holds (e.g. dependent C++ methods still C++). If true, the
    bridge IS legitimately partial — skip the assertion.

  ASSERTION (fail path)
    When the pre-condition no longer holds (e.g. all dependent methods
    converted to C), assert the bridge call has been removed from the
    parent method body — i.e. the bridge has been REABSORBED into a
    full C body.

This pattern catches the "calcification window" pythia #79 #4 named:
when the dependency converts but the partial-conversion artifact
fails to absorb. The test PASSES today (pre-condition holds), PASSES
post-reabsorption (no bridge call), and FAILS in the calcification
window — providing a forcing function rather than aspirational comment.

Origin: pythia #79 #4 [chat 2026-04-22 18:35Z], theologian design
[chat 2026-04-22 18:36Z], generalist implementation [same chat L2034
+#4 take].

Pattern docs for future PartialConversion authors:
  1. Add a test function `test_<method>_reabsorb_when_<condition>`.
  2. Encode pre-condition as text-grep on builder.cpp source.
  3. Encode assertion as bridge-symbol grep with explicit failure msg
     citing the source location of the artifact (so future readers
     find the comment + decision context fast).
"""

import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILDER_CPP = REPO_ROOT / "Python" / "jit" / "hir" / "builder.cpp"


class TestPhoenixPartialConversions(unittest.TestCase):
    """Per-bridge forcing function for Phoenix JIT PartialConversion artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.builder_src = BUILDER_CPP.read_text()

    def test_anycall_reabsorb_when_invokes_converted(self):
        """emitAnyCall await-tail bridge must reabsorb when all INVOKE_* convert.

        Bridge: hir_builder_emit_awaited_call_tail_c
        Source artifact: builder_emit_c.c (REABSORB-WHEN comment block)
        Pre-condition: at least one of emitInvokeFunction / Native / Method
          remains a bool-returning C++ method body in builder.cpp.
        Reabsorb-trigger: all 3 INVOKE_* converted → emitAnyCall opcode-switch
          can call them via C → emitAnyCall body itself can move to C →
          await-tail bridge no longer needed → inline back into full C body.
        """
        invokes_still_cpp = (
            "bool HIRBuilder::emitInvokeFunction(" in self.builder_src
            and "bool HIRBuilder::emitInvokeNative(" in self.builder_src
            and "bool HIRBuilder::emitInvokeMethod(" in self.builder_src
        )
        if invokes_still_cpp:
            self.skipTest(
                "INVOKE_* (Function/Native/Method) still C++ — "
                "PartialConversion bridge is legitimately partial."
            )
        bridge_called = (
            "hir_builder_emit_awaited_call_tail_c" in self.builder_src
        )
        self.assertFalse(
            bridge_called,
            "INVOKE_Function/Native/Method are all converted to C, but "
            "hir_builder_emit_awaited_call_tail_c bridge call remains in "
            "Python/jit/hir/builder.cpp emitAnyCall body. REABSORB now: "
            "merge the C body of hir_builder_emit_awaited_call_tail_c "
            "(builder_emit_c.c, see 'PARTIAL CONVERSION ARTIFACT' comment "
            "block) into emitAnyCall_c. Then delete the bridge function "
            "and its extern decl in builder.cpp."
        )


if __name__ == "__main__":
    unittest.main()
