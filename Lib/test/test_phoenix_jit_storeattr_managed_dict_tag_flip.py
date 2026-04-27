"""W-PYTORCH-CM-(ii) parked-bug regression test.

Reproduces the runtime StoreAttr managed-dict tag-flip corruption that
fires under ``Tools/benchmark_phoenix.py:bench_pytorch_cm`` after
``cinderjit.force_compile`` and a 50,000-iter warmup. Symptom is a SEGV
at PyDict_SetItem (NULL+0xAB deref through Py_TYPE(NULL)->tp_flags),
caused by an LSB-clear write at ``obj + 0x18`` flipping the PEP 697
managed-dict tag from IsValues (low bits 0b111) to IsDict (low bit 0).
The IsDict misinterpretation hands the slow path a non-pointer as the
dict, so the next slot write dereferences NULL.

Per supervisor 2026-04-27T07:13:18Z (cascading Alex 2026-04-27T07:12:25Z
"make sure this bug (ii) has a failing test ... fix it _after_ getting
the whole project to pure C"): the bug is parked behind the pure-C JIT
roadmap. Heavy-tier instrumentation designs for the writer hunt are on
disk at ``docs/w-pytorch-cm-tp-alloc-watchpoint-design.md`` and
``docs/w-pytorch-cm-allocate-counter-design.md``; full source-trace at
``docs/w-pytorch-cm-tooling-note.md`` and the consolidated parked-bug
entry at ``docs/known-bugs/bug-ii-storeattr-corruption.md``.

The test runs the canonical repro (``/tmp/repro_s3.py`` content,
preserved here verbatim) in a subprocess so a SEGV crashes the harness,
not the unittest runner. ``@unittest.expectedFailure`` swallows the
resulting ``AssertionError`` so the parked bug does not block CI; an
``unexpectedSuccess`` (subprocess returns 0 with ``S3 OK`` on stdout)
signals the bug has been fixed and the decorator should be removed.

Trigger sensitivity caveat (per ``docs/w-pytorch-cm-tooling-note.md``
D-1777190733): the bug is timing-sensitive. A Python ``__enter__``
wrapper around the workload was already shown to perturb JIT timing
enough to suppress the trigger. The test therefore execs the repro
verbatim through ``sys.executable -c`` rather than wrapping it in
unittest scaffolding.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

try:
    import _cinderx  # noqa: F401
    import cinderjit  # noqa: F401
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "Tools"


# Canonical repro from /tmp/repro_s3.py (228 bytes, 7 LOC). Preserved
# byte-equivalent — any edit risks perturbing the timing-sensitive
# trigger.
HARNESS_SOURCE = textwrap.dedent(
    """\
    import sys; sys.path.insert(0, 'Tools')
    import _cinderx, cinderjit
    from benchmark_phoenix import bench_pytorch_cm
    bench_pytorch_cm(5000)  # warmup
    cinderjit.force_compile(bench_pytorch_cm)
    bench_pytorch_cm(50000)
    print("S3 OK")
    """
)


@unittest.skipUnless(HAS_JIT, "requires cinderjit")
@unittest.skipUnless(
    (TOOLS_DIR / "benchmark_phoenix.py").exists(),
    "Tools/benchmark_phoenix.py not present",
)
class TestStoreAttrManagedDictTagFlip(unittest.TestCase):
    """Parked-bug oracle for W-PYTORCH-CM-(ii)."""

    @unittest.expectedFailure
    def test_pytorch_cm_no_segv_after_force_compile(self):
        """Subprocess runs the canonical repro; expects clean exit + 'S3 OK'.

        Currently parked: subprocess SEGVs (returncode != 0) and the
        AssertionError is swallowed by @expectedFailure. When the LSB-clear
        writer is identified and fixed (see docs/known-bugs/
        bug-ii-storeattr-corruption.md), this test will pass and surface
        as 'unexpectedSuccess' — at which point remove the decorator.
        """
        proc = subprocess.run(
            [sys.executable, "-c", HARNESS_SOURCE],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(
                "bench_pytorch_cm(50000) post-force_compile crashed "
                "(rc={rc}); see docs/known-bugs/bug-ii-storeattr-"
                "corruption.md\nstdout:\n{out}\nstderr (last 40 lines):\n{err}"
            ).format(
                rc=proc.returncode,
                out=proc.stdout,
                err="\n".join(proc.stderr.splitlines()[-40:]),
            ),
        )
        self.assertIn(
            "S3 OK",
            proc.stdout,
            "harness completed without SEGV but did not print 'S3 OK'",
        )


if __name__ == "__main__":
    unittest.main()
