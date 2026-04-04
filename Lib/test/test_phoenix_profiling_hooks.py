"""Phoenix JIT Profiling Hooks Test.

CinderX's design: when sys.setprofile() or sys.settrace() is active, the JIT
deoptimizes ALL compiled code back to the interpreter. Profiling hooks are
then handled by the interpreter, not by JIT code. When profiling stops, code
can be re-JIT'd.

This test verifies:
1. Profiling events fire correctly (via deopt-to-interpreter path)
2. JIT correctly deoptimizes when profiling is activated
3. Correct results through the deopt path
4. Event parity between pure-interpreter and JIT-then-deopt paths

Run with: ./python -m test test_phoenix_profiling_hooks
"""

import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


# --- Target functions for JIT compilation ---

def simple_add(a, b):
    return a + b

def nested_call(x):
    return simple_add(x, x + 1)

def with_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total

def with_exception(x):
    try:
        if x < 0:
            raise ValueError("negative")
        return x * 2
    except ValueError:
        return -1

def recursive_fib(n):
    if n <= 1:
        return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)


@unittest.skipUnless(HAS_JIT, "requires cinderjit")
class TestProfileDeoptPath(unittest.TestCase):
    """Test that profiling causes JIT deopt and events fire via interpreter."""

    def _collect_profile_events(self, func, *args, filter_names=None):
        """Run func under sys.setprofile and return collected events."""
        events = []

        def profiler(frame, event, arg):
            name = frame.f_code.co_name
            if filter_names is None or name in filter_names:
                events.append((event, name, frame.f_lineno))
            return profiler

        sys.setprofile(profiler)
        try:
            result = func(*args)
        finally:
            sys.setprofile(None)
        return result, events

    def test_simple_call_return_via_deopt(self):
        """JIT-compiled function deopts under profiling; events fire."""
        cinderjit.force_compile(simple_add)
        result, events = self._collect_profile_events(
            simple_add, 3, 4, filter_names={'simple_add'})
        self.assertEqual(result, 7)

        event_types = [e[0] for e in events]
        self.assertIn('call', event_types,
                      f"Missing 'call' event. Events: {events}")
        self.assertIn('return', event_types,
                      f"Missing 'return' event. Events: {events}")

    def test_nested_calls_via_deopt(self):
        """Nested JIT-compiled calls deopt correctly under profiling."""
        cinderjit.force_compile(simple_add)
        cinderjit.force_compile(nested_call)
        result, events = self._collect_profile_events(
            nested_call, 5, filter_names={'nested_call', 'simple_add'})
        self.assertEqual(result, 11)

        func_names = [e[1] for e in events]
        self.assertIn('nested_call', func_names,
                      f"Missing events for nested_call. Events: {events}")
        self.assertIn('simple_add', func_names,
                      f"Missing events for simple_add. Events: {events}")

    def test_exception_path_via_deopt(self):
        """Exception handling works correctly through deopt path."""
        cinderjit.force_compile(with_exception)

        # Normal path
        result, events = self._collect_profile_events(
            with_exception, 5, filter_names={'with_exception'})
        self.assertEqual(result, 10)
        event_types = [e[0] for e in events]
        self.assertIn('call', event_types)
        self.assertIn('return', event_types)

        # Exception path (caught internally)
        result, events = self._collect_profile_events(
            with_exception, -1, filter_names={'with_exception'})
        self.assertEqual(result, -1)
        event_types = [e[0] for e in events]
        self.assertIn('call', event_types)
        self.assertIn('return', event_types)

    def test_recursive_via_deopt(self):
        """Recursive JIT-compiled calls deopt and fire correct event counts."""
        cinderjit.force_compile(recursive_fib)
        result, events = self._collect_profile_events(
            recursive_fib, 6, filter_names={'recursive_fib'})
        self.assertEqual(result, 8)

        fib_calls = [e for e in events if e[0] == 'call']
        fib_returns = [e for e in events if e[0] == 'return']
        # fib(6) makes 25 calls total
        self.assertEqual(len(fib_calls), 25,
                         f"Expected 25 calls, got {len(fib_calls)}. "
                         f"Events: {events}")
        self.assertEqual(len(fib_calls), len(fib_returns),
                         "Mismatched call/return count")

    def test_correct_frame_objects(self):
        """Frame objects have correct code/filename through deopt path."""
        cinderjit.force_compile(simple_add)
        frames_seen = []

        def profiler(frame, event, arg):
            if frame.f_code.co_name == 'simple_add' and event == 'call':
                frames_seen.append({
                    'co_name': frame.f_code.co_name,
                    'co_filename': frame.f_code.co_filename,
                })
            return profiler

        sys.setprofile(profiler)
        try:
            simple_add(1, 2)
        finally:
            sys.setprofile(None)

        self.assertEqual(len(frames_seen), 1,
                         f"Expected 1 call frame. Got: {frames_seen}")
        self.assertEqual(frames_seen[0]['co_name'], 'simple_add')
        self.assertEqual(frames_seen[0]['co_filename'], __file__)


@unittest.skipUnless(HAS_JIT, "requires cinderjit")
class TestTraceDeoptPath(unittest.TestCase):
    """Test sys.settrace() causes JIT deopt and trace events fire."""

    def _collect_trace_events(self, func, *args, filter_names=None):
        """Run func under sys.settrace and return collected events."""
        events = []

        def tracer(frame, event, arg):
            name = frame.f_code.co_name
            if filter_names is None or name in filter_names:
                events.append((event, name, frame.f_lineno))
            return tracer

        sys.settrace(tracer)
        try:
            result = func(*args)
        finally:
            sys.settrace(None)
        return result, events

    def test_call_return_via_deopt(self):
        """Trace call/return events fire through deopt path."""
        cinderjit.force_compile(simple_add)
        result, events = self._collect_trace_events(
            simple_add, 10, 20, filter_names={'simple_add'})
        self.assertEqual(result, 30)

        event_types = [e[0] for e in events]
        self.assertIn('call', event_types,
                      f"Missing 'call' trace event. Events: {events}")
        self.assertIn('return', event_types,
                      f"Missing 'return' trace event. Events: {events}")

    def test_line_events_via_deopt(self):
        """Line trace events fire for loop body through deopt path."""
        cinderjit.force_compile(with_loop)
        result, events = self._collect_trace_events(
            with_loop, 3, filter_names={'with_loop'})
        self.assertEqual(result, 3)

        line_events = [e for e in events if e[0] == 'line']
        self.assertTrue(len(line_events) > 0,
                        f"No line events for with_loop. Events: {events}")

    def test_exception_trace_via_deopt(self):
        """Exception trace events fire through deopt path."""
        cinderjit.force_compile(with_exception)
        result, events = self._collect_trace_events(
            with_exception, -1, filter_names={'with_exception'})
        self.assertEqual(result, -1)

        exception_events = [e for e in events if e[0] == 'exception']
        self.assertTrue(len(exception_events) > 0,
                        f"No exception trace events. Events: {events}")


@unittest.skipUnless(HAS_JIT, "requires cinderjit")
class TestProfileInterpreterParity(unittest.TestCase):
    """Verify JIT deopt-path profile events match pure interpreter events."""

    def test_event_parity(self):
        """Deopt-path events match interpreter events exactly."""
        target_names = {'simple_add', 'nested_call'}

        # Collect events from interpreter (before compilation)
        interp_events = []

        def interp_profiler(frame, event, arg):
            if frame.f_code.co_name in target_names:
                interp_events.append((event, frame.f_code.co_name))
            return interp_profiler

        sys.setprofile(interp_profiler)
        try:
            nested_call(3)
        finally:
            sys.setprofile(None)

        # Compile, then collect events (JIT should deopt under profiling)
        cinderjit.force_compile(simple_add)
        cinderjit.force_compile(nested_call)

        jit_events = []

        def jit_profiler(frame, event, arg):
            if frame.f_code.co_name in target_names:
                jit_events.append((event, frame.f_code.co_name))
            return jit_profiler

        sys.setprofile(jit_profiler)
        try:
            nested_call(3)
        finally:
            sys.setprofile(None)

        self.assertEqual(
            interp_events, jit_events,
            f"Profile events differ.\n"
            f"Interpreter: {interp_events}\n"
            f"JIT (deopt): {jit_events}"
        )

    def test_correctness_through_deopt(self):
        """Function results are correct through deopt path."""
        cinderjit.force_compile(simple_add)
        cinderjit.force_compile(nested_call)
        cinderjit.force_compile(with_loop)
        cinderjit.force_compile(with_exception)
        cinderjit.force_compile(recursive_fib)

        results = {}

        def profiler(frame, event, arg):
            return profiler

        sys.setprofile(profiler)
        try:
            results['add'] = simple_add(100, 200)
            results['nested'] = nested_call(10)
            results['loop'] = with_loop(100)
            results['exc_normal'] = with_exception(5)
            results['exc_catch'] = with_exception(-1)
            results['fib'] = recursive_fib(10)
        finally:
            sys.setprofile(None)

        self.assertEqual(results['add'], 300)
        self.assertEqual(results['nested'], 21)
        self.assertEqual(results['loop'], 4950)
        self.assertEqual(results['exc_normal'], 10)
        self.assertEqual(results['exc_catch'], -1)
        self.assertEqual(results['fib'], 55)


if __name__ == '__main__':
    unittest.main()
