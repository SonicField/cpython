r"""W-RE-PARSER-DEOPT-DETACH sentinel — re._parser JIT correctness
regression discovered via pythia #128 #2 perf seam (generalist
2026-04-24 22:05Z; supervisor-opened workstream 22:06:56Z).

Workload: re.compile() of patterns containing named groups, hammered
across the auto-compile threshold so re._parser:Tokenizer.__next + sibling
parser methods get JIT-compiled by cinderjit.auto().

Observed at HEAD a45aa5b69c (push 50, x86_64 release):
- After ~1000 iterations the JIT auto-compiles re._parser internals.
- Subsequent re.compile() calls either:
  (a) raise AttributeError: 'SubPattern' object has no attribute 'state'
      (re/_compiler.py:761 reading p.state.groupdict)
  (b) SIGSEGV (exit code 139)
- Without JIT (no JIT_ENABLE=1): clean PASS through 2000+ iterations.

Reduction: even a single pattern r'(?P<int>\d+)' is sufficient to
trigger the crash; diverse-pattern corpus also reproduces.

JIT trace messages preceding the failure (cinderjit.auto() context):
  Deopt backoff: Tokenizer.__next reached 1000 guard failures, suppressing
  detaching JIT from Tokenizer.__next
  Deopt backoff: Tokenizer.match reached 1000 guard failures, suppressing
  detaching JIT from Tokenizer.match
  Skipping compilation of re._parser:SubPattern.__init__
    (all STORE_ATTRs generic, no specialised LOAD_ATTRs)
  Deopt backoff: SubPattern.__getitem__ reached 1000 guard failures
  Deopt backoff: SubPattern.getwidth reached 1000 guard failures
  Skipping compilation of re._parser:State.__init__
    (all STORE_ATTRs generic, no specialised LOAD_ATTRs)

Two failure modes (AttributeError vs SIGSEGV) suggest UAF / corrupted
attribute storage post-deopt-detach rather than a missing __init__ run
(theologian 22:06:38Z structural read).

W-RE-PARSER-DEOPT-DETACH workstream open per supervisor 22:06:56Z;
investigation pending. cinderx_dev oracle deferred until devgpu004
SSH-2FA infra restored.

This test is currently SKIPPED — uncomment the workload to reproduce
the SIGSEGV / AttributeError. It exists as a sentinel: when
W-RE-PARSER-DEOPT-DETACH is fixed, un-skip and add as regression
gate.
"""

import unittest


class TestREParserJITCrash(unittest.TestCase):
    def setUp(self):
        try:
            import _cinderx  # noqa: F401
            import cinderjit
            self.cinderjit = cinderjit
            cinderjit.auto()
        except ImportError:
            self.skipTest("cinderjit not available")

    @unittest.skip(
        "W-RE-PARSER-DEOPT-DETACH sentinel — uncomment to reproduce "
        "AttributeError/SIGSEGV; un-skip when bug is fixed (sentinel "
        "preserved per W-A4-mid sentinel precedent)."
    )
    def test_named_group_compile_repeated(self):
        """Reduced repro: single named-group pattern, hammered past
        the auto-compile threshold so re._parser:Tokenizer.__next +
        SubPattern.__getitem__ + .getwidth all hit deopt-backoff +
        detach. Subsequent re.compile() either crashes or raises
        AttributeError on SubPattern.state."""
        import re
        for _ in range(2000):
            re.compile(r'(?P<int>\d+)')
            re.purge()


if __name__ == '__main__':
    unittest.main()
