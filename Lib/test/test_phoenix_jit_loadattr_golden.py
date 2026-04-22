"""Phoenix JIT LoadAttr Golden HIR Test.

Regression oracle for the three LOAD_ATTR specializations:

  - LOAD_ATTR_SLOT          (class with __slots__)
  - LOAD_ATTR_INSTANCE_VALUE (regular dict-backed attribute)
  - LOAD_ATTR_MODULE        (module-level attribute)

The wiring gate's force_compile harness exercises NONE of these paths (verified
by testkeeper 2026-04-21), so HIR divergence introduced when the C emit methods
replace their C++ counterparts cannot be caught behaviourally without an
explicit oracle. This test fills that gap by:

  1. Running a self-contained subprocess that warms up an attr-heavy function
     past the adaptive interpreter's specialization threshold, asserts that
     dis() reports the three specialized opcodes, then force-compiles the
     function with PHOENIX_GOLDEN_CAPTURE=1.
  2. Capturing the GOLDEN_HIR_FINAL and GOLDEN_HIR_COMPILE blocks from stderr.
  3. Comparing the captured bytes byte-for-byte against
     docs/golden/loadattr_hir.txt.

To regenerate the golden file (e.g. after an intentional HIR change):

    PHOENIX_REGEN_GOLDEN=1 ./python -m test test_phoenix_jit_loadattr_golden

Run with: ./python -m test test_phoenix_jit_loadattr_golden
"""

import os
import re
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

# Mode-aware golden selection. Pydebug builds populate sys with extra
# attributes (sys.gettotalrefcount, etc.), incrementing sys.__dict__'s
# keys-version counter that LOAD_ATTR_MODULE inlines as a CInt64 constant.
# The +1 delta on that single embedded constant is intrinsic to the
# build mode, NOT a JIT codegen regression. Per W21 root-cause investigation
# 2026-04-22 17:54Z (testkeeper).
_PYDEBUG = hasattr(sys, "gettotalrefcount")
GOLDEN_NAME = "loadattr_hir.pydebug.txt" if _PYDEBUG else "loadattr_hir.txt"
GOLDEN_PATH = REPO_ROOT / "docs" / "golden" / GOLDEN_NAME


HARNESS_SOURCE = textwrap.dedent(
    """\
    import dis
    import io
    import sys

    # _cinderx must be imported first; importing it registers the cinderjit
    # module entry in sys.modules. Without this, ``import cinderjit`` fails.
    import _cinderx  # noqa: F401
    import cinderjit


    class Pt:
        __slots__ = ("x", "y")

        def __init__(self, x, y):
            self.x = x
            self.y = y


    class Inst:
        def __init__(self, v):
            self.v = v


    def attr_probe(p, i):
        a = p.x          # LOAD_ATTR_SLOT after warmup
        b = i.v          # LOAD_ATTR_INSTANCE_VALUE after warmup
        c = sys.maxsize  # LOAD_ATTR_MODULE after warmup
        return a + b + c


    def main():
        p = Pt(1, 2)
        i = Inst(3)

        # Warmup so the adaptive interpreter specializes each LOAD_ATTR site.
        # ADAPTIVE_WARMUP_VALUE is 1 in CPython 3.12; 64 iterations is well past
        # the threshold for all three sites.
        for _ in range(64):
            attr_probe(p, i)

        # adaptive=True is REQUIRED — without it, dis prints the un-adapted
        # bytecode and the LOAD_ATTR_SLOT/MODULE/INSTANCE_VALUE specializations
        # are invisible even when the adaptive interpreter has applied them.
        buf = io.StringIO()
        dis.dis(attr_probe, file=buf, adaptive=True)
        disasm = buf.getvalue()

        for needed in (
            "LOAD_ATTR_SLOT",
            "LOAD_ATTR_INSTANCE_VALUE",
            "LOAD_ATTR_MODULE",
        ):
            if needed not in disasm:
                sys.stderr.write(
                    "SPECIALIZATION_MISSING {}\\n{}\\n".format(needed, disasm)
                )
                sys.exit(2)

        cinderjit.force_compile(attr_probe)
        if not cinderjit.is_jit_compiled(attr_probe):
            sys.stderr.write("FORCE_COMPILE_FAILED\\n")
            sys.exit(3)

        # Sanity: the compiled function must still produce the expected value.
        expected = 1 + 3 + sys.maxsize
        got = attr_probe(p, i)
        if got != expected:
            sys.stderr.write(
                "RESULT_MISMATCH expected={} got={}\\n".format(expected, got)
            )
            sys.exit(4)

        sys.stdout.write("OK\\n")


    if __name__ == "__main__":
        main()
    """
)


# Embedded pointer addresses (e.g. `GuardIs<0x7fd75bd8e7a0>`) are
# process-dependent — they vary between runs because they identify Python
# objects allocated by the host process. Canonicalize them so the golden file
# stays stable.
_PTR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")

# LOAD_ATTR_MODULE embeds the module dict's `dk_version` (a monotonic
# counter bumped on every dict-key mutation) into the HIR as
# `LoadConst<CUInt32[N]>`. N depends on how many dict mutations happened
# during interpreter startup, which is architecture- and import-order
# dependent (x86_64 saw 37, ARM64 saw 46). Canonicalize the embedded
# integer on `CUInt32[N]` annotations so the golden stays cross-arch
# stable. Other integer-specialized constant types (CUInt64[N], CInt64[N])
# encode stable shape information (slot index, key hash) and are NOT
# canonicalized — divergence in those values is a real regression signal.
_DK_VERSION_RE = re.compile(r"CUInt32\[\d+\]")


def _canonicalize(text: str) -> str:
    text = _PTR_RE.sub("0xPTR", text)
    text = _DK_VERSION_RE.sub("CUInt32[N]", text)
    return text


def _extract_blocks(stderr_text: str) -> str:
    """Concatenate the HIR_FINAL + HIR_COMPILE blocks for ``attr_probe``.

    Output format matches docs/golden/loadattr_hir.txt:

        GOLDEN_HIR_FINAL <fullname>
        <body>
        END_GOLDEN_HIR_FINAL
        GOLDEN_HIR_COMPILE <fullname>
        <body>
        END_GOLDEN_HIR_COMPILE
    """
    pattern = re.compile(
        r"^GOLDEN_HIR_(?P<kind>FINAL|COMPILE) (?P<name>[^\n]+)\n"
        r"(?P<body>.*?)\n"
        r"END_GOLDEN_HIR_(?P=kind)\n",
        re.MULTILINE | re.DOTALL,
    )

    chunks = []
    for match in pattern.finditer(stderr_text):
        if "attr_probe" not in match.group("name"):
            continue
        chunks.append(
            "GOLDEN_HIR_{kind} {name}\n{body}\nEND_GOLDEN_HIR_{kind}".format(
                kind=match.group("kind"),
                name=match.group("name"),
                body=match.group("body"),
            )
        )
    return "\n".join(chunks) + "\n" if chunks else ""


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestLoadAttrGolden(unittest.TestCase):
    """Golden-HIR oracle for the three LOAD_ATTR specializations."""

    def _run_harness(self) -> str:
        env = dict(os.environ)
        env["PHOENIX_GOLDEN_CAPTURE"] = "1"
        # Force unbuffered stderr so the GOLDEN_HIR_* blocks land before exit.
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.run(
            [sys.executable, "-c", HARNESS_SOURCE],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(
                "harness exited {rc}\nstdout:\n{out}\nstderr (last 80 lines):\n{err}"
            ).format(
                rc=proc.returncode,
                out=proc.stdout,
                err="\n".join(proc.stderr.splitlines()[-80:]),
            ),
        )
        self.assertIn("OK", proc.stdout, "harness did not report OK")
        return proc.stderr

    def test_loadattr_hir_matches_golden(self):
        stderr_text = self._run_harness()
        captured = _canonicalize(_extract_blocks(stderr_text))
        self.assertTrue(
            captured,
            "no GOLDEN_HIR_* blocks for attr_probe found in harness stderr",
        )

        if os.environ.get("PHOENIX_REGEN_GOLDEN") == "1":
            GOLDEN_PATH.write_text(captured)
            self.skipTest(
                "regenerated {}; rerun without PHOENIX_REGEN_GOLDEN".format(
                    GOLDEN_PATH
                )
            )

        self.assertTrue(
            GOLDEN_PATH.exists(),
            "golden file missing: {}".format(GOLDEN_PATH),
        )
        expected = _canonicalize(GOLDEN_PATH.read_text())

        if captured != expected:
            # Surface a precise diff so divergence is debuggable.
            import difflib

            diff = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    captured.splitlines(keepends=True),
                    fromfile=str(GOLDEN_PATH),
                    tofile="captured",
                    n=3,
                )
            )
            self.fail(
                "LoadAttr HIR diverged from golden — a LOAD_ATTR specialization "
                "emit method changed observable HIR. Inspect the diff and, if "
                "intentional, regenerate via PHOENIX_REGEN_GOLDEN=1.\n\n"
                + diff[:8000]
            )


if __name__ == "__main__":
    unittest.main()
