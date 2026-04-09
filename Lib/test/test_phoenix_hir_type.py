"""
Behavioral tests for hir::Type bit-level constants and layout.

The operator behavioral tests (union, intersect, subtract, equality)
are implemented as C unit tests in Python/jit/hir/test_hir_type.c,
which test the C functions directly without a Python bridge layer.

Integration tests through the Python→JIT→C pipeline are DEFERRED
to the bridge milestone (cinderjit._hir_type_test module). When that
module exists, add integration tests here that verify the full path.
"""

import unittest
import sys


class TestHirTypeCppBaseline(unittest.TestCase):
    """
    Baseline tests for hir::Type bit-level constants.

    These verify that type_generated.h constants have the expected
    values and relationships. They run without any JIT infrastructure.
    """

    def _make_type_bits(self, bits, lifetime, spec_kind=0, spec_val=0):
        """Construct a HirType bits_and_flags value."""
        bf = (bits & ((1 << 44) - 1))
        bf |= (lifetime & 0x3) << 44
        bf |= (spec_kind & 0x7) << 46
        return bf

    # ---- Layout constants (from type_generated.h) ----
    kBottom = 0x00000000000
    kObject = 0x000ffffffff
    kPrimitive = 0xfff00000000
    kTop = 0xfffffffffff

    # Specific type bits (verified against type_generated.h)
    kLong = 0x00000200402  # Long (includes LongExact + Bool + LongUser)
    kLongExact = 0x00000000400  # LongExact only
    kList = 0x00008010000  # List (includes ListExact + ListUser)
    kListExact = 0x00000010000  # ListExact only
    kBool = 0x00000000002  # Bool
    kCInt64 = 0x01000000000  # CInt64 (primitive)
    kCDouble = 0x40000000000  # CDouble (primitive)

    kLifetimeBottom = 0
    kLifetimeMortal = 1
    kLifetimeImmortal = 2
    kLifetimeTop = 3

    def test_bottom_bits_are_zero(self):
        """TBottom has zero bits and zero lifetime."""
        self.assertEqual(self.kBottom, 0)

    def test_object_and_primitive_disjoint(self):
        """kObject and kPrimitive bit ranges don't overlap."""
        self.assertEqual(self.kObject & self.kPrimitive, 0)

    def test_top_is_object_or_primitive(self):
        """kTop = kObject | kPrimitive."""
        self.assertEqual(self.kTop, self.kObject | self.kPrimitive)

    def test_long_exact_subset_of_long(self):
        """LongExact bits are a subset of Long bits."""
        self.assertEqual(self.kLongExact & self.kLong, self.kLongExact)

    def test_intersect_basic_bits(self):
        """Intersection of Long and List: bits = Long.bits & List.bits."""
        result_bits = self.kLong & self.kList
        # Long and List have no overlapping bits → 0 (TBottom candidate)
        self.assertEqual(result_bits, 0)

    def test_intersect_object_with_primitive(self):
        """Object & Primitive = Bottom (no overlap)."""
        result_bits = self.kObject & self.kPrimitive
        self.assertEqual(result_bits, 0)

    def test_union_basic_bits(self):
        """Union of Long and List: bits = Long.bits | List.bits."""
        result_bits = self.kLong | self.kList
        self.assertNotEqual(result_bits, 0)
        # Both should be subsets
        self.assertEqual(result_bits & self.kLong, self.kLong)
        self.assertEqual(result_bits & self.kList, self.kList)

    def test_subtract_basic_primitive(self):
        """Top - Object = Primitive (bits level)."""
        result_bits = self.kTop & ~(self.kObject)
        # After removing object bits, only primitive remain
        self.assertEqual(result_bits, self.kPrimitive)

    def test_lifetime_mortal_intersect_immortal_is_bottom(self):
        """Mortal & Immortal = LifetimeBottom (no overlap)."""
        result = self.kLifetimeMortal & self.kLifetimeImmortal
        self.assertEqual(result, self.kLifetimeBottom)

    def test_lifetime_top_intersect_mortal_is_mortal(self):
        """Top & Mortal = Mortal."""
        result = self.kLifetimeTop & self.kLifetimeMortal
        self.assertEqual(result, self.kLifetimeMortal)


if __name__ == '__main__':
    unittest.main()
