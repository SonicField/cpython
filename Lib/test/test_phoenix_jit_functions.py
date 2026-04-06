"""Tests for Phoenix JIT compilation of function calls, closures, classes, and related operations.

Every test:
1. Defines a Python function exercising specific opcode(s)
2. Runs it under the interpreter first to get the expected result
3. Force-compiles via cinderjit.force_compile()
4. Runs under JIT and asserts the result matches

Run with: ./python -m test test_phoenix_jit_functions
"""

import functools
import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


# Global variable used by LOAD_GLOBAL / STORE_GLOBAL tests
_GLOBAL_COUNTER = 0
_GLOBAL_VALUE = 42


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestJitFunctions(unittest.TestCase):
    """Comprehensive tests for JIT compilation of function/class patterns."""

    def _jit_test(self, func, *args, inner_funcs=None):
        """Run func under interpreter, force-compile, run under JIT, compare."""
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(cinderjit.is_jit_compiled(func),
                        f"{func.__name__} did not compile")
        if inner_funcs:
            for f in inner_funcs:
                try:
                    cinderjit.force_compile(f)
                except Exception:
                    pass
        jit_result = func(*args)
        self.assertEqual(interp_result, jit_result,
                         f"{func.__name__}: interpreter={interp_result!r}, JIT={jit_result!r}")
        return jit_result

    # ---------------------------------------------------------------
    # 1. CALL -- simple function calls
    # ---------------------------------------------------------------

    def test_call_no_args(self):
        def f():
            return 99
        self._jit_test(f)

    def test_call_positional_args(self):
        def add(a, b, c):
            return a + b + c
        def f():
            return add(1, 2, 3)
        self._jit_test(f)

    def test_call_with_kwargs(self):
        def greet(name, greeting="hello"):
            return f"{greeting} {name}"
        def f():
            return greet("world", greeting="hi")
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 2. CALL_FUNCTION_EX -- *args, **kwargs unpacking in calls
    # ---------------------------------------------------------------

    def test_call_star_args_unpack(self):
        def add3(a, b, c):
            return a + b + c
        def f():
            args = (10, 20, 30)
            return add3(*args)
        self._jit_test(f)

    def test_call_star_kwargs_unpack(self):
        def make_point(x=0, y=0, z=0):
            return (x, y, z)
        def f():
            kw = {"x": 1, "y": 2, "z": 3}
            return make_point(**kw)
        self._jit_test(f)

    def test_call_star_args_and_kwargs_unpack(self):
        def target(a, b, c=0, d=0):
            return a + b + c + d
        def f():
            args = (1, 2)
            kwargs = {"c": 3, "d": 4}
            return target(*args, **kwargs)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 3. Keyword arguments -- positional + keyword mixed
    # ---------------------------------------------------------------

    def test_mixed_positional_and_keyword(self):
        def compute(a, b, op="add"):
            if op == "add":
                return a + b
            elif op == "mul":
                return a * b
            return 0
        def f():
            r1 = compute(3, 4)
            r2 = compute(3, 4, op="mul")
            return (r1, r2)
        self._jit_test(f)

    def test_keyword_only_args(self):
        def kw_only(*, x, y):
            return x * y
        def f():
            return kw_only(x=5, y=7)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 4. Default arguments -- mutable and immutable defaults
    # ---------------------------------------------------------------

    def test_immutable_defaults(self):
        def with_default(a, b=10, c=20):
            return a + b + c
        def f():
            return (with_default(1), with_default(1, 2), with_default(1, 2, 3))
        self._jit_test(f)

    def test_mutable_default_list(self):
        def accumulate(val, lst=None):
            if lst is None:
                lst = []
            lst.append(val)
            return lst
        def f():
            r1 = accumulate(1)
            r2 = accumulate(2)
            r3 = accumulate(3, [10, 20])
            return (r1, r2, r3)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 5. *args and **kwargs -- variadic functions
    # ---------------------------------------------------------------

    def test_variadic_args(self):
        def var_sum(*args):
            return sum(args)
        def f():
            return var_sum(1, 2, 3, 4, 5)
        self._jit_test(f)

    def test_variadic_kwargs(self):
        def kw_join(**kwargs):
            return ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        def f():
            return kw_join(a=1, b=2, c=3)
        self._jit_test(f)

    def test_variadic_args_and_kwargs(self):
        def both(*args, **kwargs):
            return (args, tuple(sorted(kwargs.items())))
        def f():
            return both(1, 2, x=3, y=4)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 6. MAKE_FUNCTION -- nested function definitions
    # ---------------------------------------------------------------

    def test_nested_function_def(self):
        def f():
            def inner(x):
                return x * 2
            return inner(21)
        self._jit_test(f)

    def test_nested_function_with_defaults(self):
        def f():
            def inner(x, multiplier=3):
                return x * multiplier
            return (inner(5), inner(5, multiplier=10))
        self._jit_test(f)

    def test_nested_function_with_annotations(self):
        def f():
            def inner(x: int, y: int) -> int:
                return x + y
            return inner(3, 4)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 7-9. LOAD_CLOSURE / MAKE_CELL / LOAD_DEREF / STORE_DEREF / nested closures
    # ---------------------------------------------------------------

    def test_simple_closure(self):
        def f():
            x = 10
            def inner():
                return x + 5
            return inner()
        self._jit_test(f)

    def test_closure_captures_loop_var(self):
        def f():
            funcs = []
            for i in range(5):
                def capture(val=i):
                    return val * val
                funcs.append(capture)
            return [fn() for fn in funcs]
        self._jit_test(f)

    def test_closure_store_deref(self):
        def f():
            count = 0
            def increment():
                nonlocal count
                count += 1
                return count
            r1 = increment()
            r2 = increment()
            r3 = increment()
            return (r1, r2, r3, count)
        self._jit_test(f)

    def test_closure_two_levels(self):
        def f():
            x = 1
            def mid():
                y = 2
                def inner():
                    return x + y
                return inner()
            return mid()
        self._jit_test(f)

    def test_closure_three_levels(self):
        def f():
            a = 10
            def level1():
                b = 20
                def level2():
                    c = 30
                    def level3():
                        return a + b + c
                    return level3()
                return level2()
            return level1()
        self._jit_test(f)

    def test_closure_mutation_across_levels(self):
        def f():
            result = []
            x = 0
            def outer():
                nonlocal x
                x += 1
                def inner():
                    nonlocal x
                    x += 10
                    return x
                return inner()
            result.append(outer())
            result.append(outer())
            result.append(x)
            return result
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 10. Lambda expressions
    # ---------------------------------------------------------------

    def test_lambda_simple(self):
        def f():
            double = lambda x: x * 2
            return double(21)
        self._jit_test(f)

    def test_lambda_with_closure(self):
        def f():
            base = 100
            add_base = lambda x: x + base
            return add_base(42)
        self._jit_test(f)

    def test_lambda_in_list(self):
        def f():
            ops = [lambda x: x + 1, lambda x: x * 2, lambda x: x ** 2]
            return [op(5) for op in ops]
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 11-12. LOAD_GLOBAL / STORE_GLOBAL
    # ---------------------------------------------------------------

    def test_load_global(self):
        def f():
            return _GLOBAL_VALUE
        self._jit_test(f)

    def test_store_global(self):
        def f():
            global _GLOBAL_COUNTER
            _GLOBAL_COUNTER = 999
            return _GLOBAL_COUNTER
        interp_result = f()
        cinderjit.force_compile(f)
        self.assertTrue(cinderjit.is_jit_compiled(f))
        global _GLOBAL_COUNTER
        _GLOBAL_COUNTER = 0  # reset before JIT run
        jit_result = f()
        self.assertEqual(interp_result, jit_result)
        self.assertEqual(_GLOBAL_COUNTER, 999)

    # ---------------------------------------------------------------
    # 13. LOAD_FAST / STORE_FAST -- local variable access
    # ---------------------------------------------------------------

    def test_load_store_fast(self):
        def f():
            a = 1
            b = 2
            c = a + b
            d = c * a
            e = d - b
            return e
        self._jit_test(f)

    def test_many_locals(self):
        def f():
            a, b, c, d, e = 1, 2, 3, 4, 5
            g, h, i, j, k = 6, 7, 8, 9, 10
            return a + b + c + d + e + g + h + i + j + k
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 14. DELETE_FAST -- del local_var
    # ---------------------------------------------------------------

    def test_delete_fast(self):
        def f():
            x = 42
            y = x
            del x
            try:
                _ = x
                return "should not reach"
            except UnboundLocalError:
                return ("deleted", y)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 15. Class instantiation
    # ---------------------------------------------------------------

    def test_class_simple_instantiation(self):
        def f():
            class Point:
                pass
            p = Point()
            return type(p).__name__
        self._jit_test(f)

    def test_class_with_init(self):
        def f():
            class Point:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            p = Point(3, 4)
            return (p.x, p.y)
        self._jit_test(f)

    def test_class_with_new(self):
        def f():
            class Singleton:
                _instance = None
                def __new__(cls):
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                    return cls._instance
            a = Singleton()
            b = Singleton()
            return a is b
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 16. Method calls -- instance, class, static
    # ---------------------------------------------------------------

    def test_instance_method(self):
        def f():
            class Calc:
                def __init__(self, base):
                    self.base = base
                def add(self, x):
                    return self.base + x
            c = Calc(100)
            return c.add(23)
        self._jit_test(f)

    def test_classmethod(self):
        def f():
            class Counter:
                count = 0
                @classmethod
                def increment(cls):
                    cls.count += 1
                    return cls.count
            Counter.increment()
            Counter.increment()
            return Counter.increment()
        self._jit_test(f)

    def test_staticmethod(self):
        def f():
            class Math:
                @staticmethod
                def square(x):
                    return x * x
            return Math.square(7)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 17-19. LOAD_ATTR / STORE_ATTR / DELETE_ATTR
    # ---------------------------------------------------------------

    def test_load_attr(self):
        def f():
            class Obj:
                def __init__(self):
                    self.value = 42
            return Obj().value
        self._jit_test(f)

    def test_store_attr(self):
        def f():
            class Obj:
                pass
            o = Obj()
            o.x = 10
            o.y = 20
            return (o.x, o.y)
        self._jit_test(f)

    def test_delete_attr(self):
        def f():
            class Obj:
                pass
            o = Obj()
            o.x = 42
            val = o.x
            del o.x
            try:
                _ = o.x
                return "should not reach"
            except AttributeError:
                return ("deleted", val)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 20. Property access -- @property getter/setter
    # ---------------------------------------------------------------

    def test_property_getter(self):
        def f():
            class Circle:
                def __init__(self, r):
                    self._r = r
                @property
                def area(self):
                    return 3.14159 * self._r * self._r
            c = Circle(5)
            return round(c.area, 2)
        self._jit_test(f)

    def test_property_setter(self):
        def f():
            class Temp:
                def __init__(self, c):
                    self._celsius = c
                @property
                def fahrenheit(self):
                    return self._celsius * 9 / 5 + 32
                @fahrenheit.setter
                def fahrenheit(self, val):
                    self._celsius = (val - 32) * 5 / 9
            t = Temp(100)
            before = t.fahrenheit
            t.fahrenheit = 32
            after = t._celsius
            return (round(before, 1), round(after, 1))
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 21. Inheritance -- single, multiple, super()
    # ---------------------------------------------------------------

    def test_single_inheritance(self):
        def f():
            class Animal:
                def speak(self):
                    return "..."
            class Dog(Animal):
                def speak(self):
                    return "woof"
            return Dog().speak()
        self._jit_test(f)

    def test_multiple_inheritance(self):
        def f():
            class A:
                def who(self):
                    return "A"
            class B(A):
                def who(self):
                    return "B+" + super().who()
            class C(A):
                def who(self):
                    return "C+" + super().who()
            class D(B, C):
                def who(self):
                    return "D+" + super().who()
            return D().who()
        self._jit_test(f)

    def test_super_call(self):
        def f():
            class Base:
                def __init__(self):
                    self.base_val = 10
            class Child(Base):
                def __init__(self):
                    super().__init__()
                    self.child_val = 20
            c = Child()
            return (c.base_val, c.child_val)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 22. __dunder__ methods
    # ---------------------------------------------------------------

    def test_dunder_len(self):
        def f():
            class Bag:
                def __init__(self, items):
                    self._items = items
                def __len__(self):
                    return len(self._items)
            return len(Bag([1, 2, 3, 4, 5]))
        self._jit_test(f)

    def test_dunder_getitem_setitem(self):
        def f():
            class MyList:
                def __init__(self):
                    self._data = {}
                def __getitem__(self, key):
                    return self._data.get(key, -1)
                def __setitem__(self, key, val):
                    self._data[key] = val
            m = MyList()
            m[0] = 99
            m[1] = 100
            return (m[0], m[1], m[2])
        self._jit_test(f)

    def test_dunder_contains(self):
        def f():
            class EvenSet:
                def __contains__(self, item):
                    return item % 2 == 0
            s = EvenSet()
            return (2 in s, 3 in s, 4 in s, 5 in s)
        self._jit_test(f)

    def test_dunder_iter_next(self):
        def f():
            class CountUp:
                def __init__(self, limit):
                    self.limit = limit
                    self.current = 0
                def __iter__(self):
                    return self
                def __next__(self):
                    if self.current >= self.limit:
                        raise StopIteration
                    val = self.current
                    self.current += 1
                    return val
            return list(CountUp(5))
        self._jit_test(f)

    def test_dunder_add_mul(self):
        def f():
            class Vec:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                def __add__(self, other):
                    return Vec(self.x + other.x, self.y + other.y)
                def __mul__(self, scalar):
                    return Vec(self.x * scalar, self.y * scalar)
                def as_tuple(self):
                    return (self.x, self.y)
            v1 = Vec(1, 2)
            v2 = Vec(3, 4)
            v3 = (v1 + v2) * 2
            return v3.as_tuple()
        self._jit_test(f)

    def test_dunder_repr_str(self):
        def f():
            class Named:
                def __init__(self, name):
                    self.name = name
                def __repr__(self):
                    return f"Named({self.name!r})"
                def __str__(self):
                    return self.name
            n = Named("test")
            return (repr(n), str(n))
        self._jit_test(f)

    def test_dunder_eq_hash(self):
        def f():
            class Pt:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                def __eq__(self, other):
                    return self.x == other.x and self.y == other.y
                def __hash__(self):
                    return hash((self.x, self.y))
            p1 = Pt(1, 2)
            p2 = Pt(1, 2)
            p3 = Pt(3, 4)
            return (p1 == p2, p1 == p3, hash(p1) == hash(p2))
        self._jit_test(f)

    def test_dunder_bool(self):
        def f():
            class Truthy:
                def __init__(self, val):
                    self.val = val
                def __bool__(self):
                    return self.val > 0
            return (bool(Truthy(1)), bool(Truthy(0)), bool(Truthy(-1)))
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 23. isinstance / issubclass checks
    # ---------------------------------------------------------------

    def test_isinstance(self):
        def f():
            class A: pass
            class B(A): pass
            class C: pass
            b = B()
            return (
                isinstance(b, B),
                isinstance(b, A),
                isinstance(b, C),
                isinstance(b, (A, C)),
                isinstance(42, int),
                isinstance("hi", str),
            )
        self._jit_test(f)

    def test_issubclass(self):
        def f():
            class A: pass
            class B(A): pass
            class C: pass
            return (
                issubclass(B, A),
                issubclass(B, B),
                issubclass(A, B),
                issubclass(B, (A, C)),
            )
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 24. hasattr / getattr / setattr / delattr builtins
    # ---------------------------------------------------------------

    def test_hasattr_getattr(self):
        def f():
            class Obj:
                x = 10
            o = Obj()
            o.y = 20
            return (
                hasattr(o, "x"),
                hasattr(o, "y"),
                hasattr(o, "z"),
                getattr(o, "x"),
                getattr(o, "y"),
                getattr(o, "z", -1),
            )
        self._jit_test(f)

    def test_setattr_delattr(self):
        def f():
            class Obj: pass
            o = Obj()
            setattr(o, "val", 42)
            r1 = o.val
            delattr(o, "val")
            r2 = hasattr(o, "val")
            return (r1, r2)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 25. Decorators -- function, class, stacked
    # ---------------------------------------------------------------

    def test_function_decorator(self):
        def f():
            def double_result(fn):
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs) * 2
                return wrapper
            @double_result
            def add(a, b):
                return a + b
            return add(3, 4)
        self._jit_test(f)

    def test_stacked_decorators(self):
        def f():
            def add_one(fn):
                def wrapper(*a, **kw):
                    return fn(*a, **kw) + 1
                return wrapper
            def double(fn):
                def wrapper(*a, **kw):
                    return fn(*a, **kw) * 2
                return wrapper
            @add_one
            @double
            def value(x):
                return x
            # value(5) -> double(5) -> 10, then add_one -> 11
            return value(5)
        self._jit_test(f)

    def test_class_decorator(self):
        def f():
            def add_greet(cls):
                cls.greet = lambda self: f"hello from {cls.__name__}"
                return cls
            @add_greet
            class Foo:
                pass
            return Foo().greet()
        self._jit_test(f)

    def test_decorator_with_args(self):
        def f():
            def multiply_by(n):
                def decorator(fn):
                    def wrapper(*args, **kwargs):
                        return fn(*args, **kwargs) * n
                    return wrapper
                return decorator
            @multiply_by(3)
            def get_val(x):
                return x + 1
            return get_val(10)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 26. Callable objects -- __call__ method
    # ---------------------------------------------------------------

    def test_callable_object(self):
        def f():
            class Adder:
                def __init__(self, base):
                    self.base = base
                def __call__(self, x):
                    return self.base + x
            add10 = Adder(10)
            return (add10(5), add10(20), add10(-3))
        self._jit_test(f)

    def test_callable_with_state(self):
        def f():
            class Counter:
                def __init__(self):
                    self.n = 0
                def __call__(self):
                    self.n += 1
                    return self.n
            c = Counter()
            return (c(), c(), c())
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 27. Partial application -- functools.partial
    # ---------------------------------------------------------------

    def test_functools_partial(self):
        def f():
            def power(base, exp):
                return base ** exp
            square = functools.partial(power, exp=2)
            cube = functools.partial(power, exp=3)
            return (square(5), cube(3))
        self._jit_test(f)

    def test_functools_partial_positional(self):
        def f():
            def add3(a, b, c):
                return a + b + c
            add_10 = functools.partial(add3, 10)
            return add_10(20, 30)
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 28. Map/filter/reduce with JIT-compiled callables
    # ---------------------------------------------------------------

    def test_map_with_function(self):
        def f():
            def square(x):
                return x * x
            return list(map(square, range(10)))
        self._jit_test(f)

    def test_filter_with_function(self):
        def f():
            def is_even(x):
                return x % 2 == 0
            return list(filter(is_even, range(20)))
        self._jit_test(f)

    def test_reduce_with_function(self):
        def f():
            def mul(a, b):
                return a * b
            return functools.reduce(mul, range(1, 8))
        self._jit_test(f)

    def test_map_filter_chained(self):
        def f():
            def sq(x):
                return x * x
            def big(x):
                return x > 20
            return list(filter(big, map(sq, range(10))))
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 29. Recursive functions
    # ---------------------------------------------------------------

    def test_recursive_fibonacci(self):
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
        def f():
            return [fib(i) for i in range(12)]
        self._jit_test(f, inner_funcs=[fib])

    def test_recursive_factorial(self):
        def fact(n):
            if n <= 1:
                return 1
            return n * fact(n - 1)
        def f():
            return [fact(i) for i in range(10)]
        self._jit_test(f, inner_funcs=[fact])

    def test_recursive_sum_list(self):
        def rsum(lst):
            if not lst:
                return 0
            return lst[0] + rsum(lst[1:])
        def f():
            return rsum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self._jit_test(f, inner_funcs=[rsum])

    # ---------------------------------------------------------------
    # 30. Mutual recursion
    # ---------------------------------------------------------------

    def test_mutual_recursion(self):
        def is_even(n):
            if n == 0:
                return True
            return is_odd(n - 1)
        def is_odd(n):
            if n == 0:
                return False
            return is_even(n - 1)
        def f():
            return [(i, is_even(i), is_odd(i)) for i in range(8)]
        self._jit_test(f, inner_funcs=[is_even, is_odd])

    # ---------------------------------------------------------------
    # 31. LOAD_METHOD / CALL_METHOD optimization path
    # ---------------------------------------------------------------

    def test_method_call_optimization(self):
        def f():
            s = "hello world"
            return (
                s.upper(),
                s.split(),
                s.replace("world", "jit"),
                s.startswith("hello"),
                s.endswith("world"),
            )
        self._jit_test(f)

    def test_list_method_calls(self):
        def f():
            lst = [3, 1, 4, 1, 5, 9, 2, 6]
            result = []
            result.append(lst.count(1))
            result.append(lst.index(5))
            s = sorted(lst)
            result.append(s)
            lst2 = list(lst)
            lst2.sort()
            result.append(lst2)
            return result
        self._jit_test(f)

    def test_dict_method_calls(self):
        def f():
            d = {"a": 1, "b": 2, "c": 3}
            return (
                d.get("a"),
                d.get("z", -1),
                sorted(d.keys()),
                sorted(d.values()),
                sorted(d.items()),
            )
        self._jit_test(f)

    # ---------------------------------------------------------------
    # 32. Builtin function calls
    # ---------------------------------------------------------------

    def test_builtin_len(self):
        def f():
            return (len([1, 2, 3]), len("hello"), len({1: 2, 3: 4}), len((1,)))
        self._jit_test(f)

    def test_builtin_range(self):
        def f():
            return (list(range(5)), list(range(2, 8)), list(range(0, 10, 3)))
        self._jit_test(f)

    def test_builtin_min_max_sum(self):
        def f():
            data = [3, 1, 4, 1, 5, 9, 2, 6]
            return (min(data), max(data), sum(data))
        self._jit_test(f)

    def test_builtin_sorted_reversed(self):
        def f():
            data = [3, 1, 4, 1, 5]
            return (sorted(data), sorted(data, reverse=True), list(reversed(data)))
        self._jit_test(f)

    def test_builtin_enumerate(self):
        def f():
            return list(enumerate(["a", "b", "c"], start=1))
        self._jit_test(f)

    def test_builtin_zip(self):
        def f():
            return list(zip([1, 2, 3], ["a", "b", "c"], [10, 20, 30]))
        self._jit_test(f)

    def test_builtin_map_filter(self):
        def f():
            return (list(map(str, [1, 2, 3])), list(filter(None, [0, 1, "", "a", [], [1]])))
        self._jit_test(f)

    def test_builtin_any_all(self):
        def f():
            return (
                any([False, False, True]),
                any([False, False]),
                all([True, True, True]),
                all([True, False, True]),
            )
        self._jit_test(f)

    def test_builtin_type_id_hash(self):
        def f():
            x = 42
            s = "hello"
            return (
                type(x).__name__,
                type(s).__name__,
                type(x) is int,
                isinstance(hash(x), int),
                isinstance(hash(s), int),
            )
        self._jit_test(f)

    def test_builtin_repr_str(self):
        def f():
            return (repr(42), repr("hi"), str(42), str(3.14))
        self._jit_test(f)

    def test_builtin_conversions(self):
        def f():
            return (
                int("42"),
                int(3.7),
                float("3.14"),
                float(42),
                bool(0),
                bool(1),
                bool(""),
                bool("x"),
                list((1, 2, 3)),
                tuple([1, 2, 3]),
                set([1, 2, 2, 3]),
                dict([("a", 1), ("b", 2)]),
            )
        self._jit_test(f)

    def test_builtin_abs_pow_round(self):
        def f():
            return (abs(-5), abs(5), pow(2, 10), round(3.14159, 2))
        self._jit_test(f)

    # ---------------------------------------------------------------
    # Additional coverage: edge cases and compound patterns
    # ---------------------------------------------------------------

    def test_generator_function(self):
        def f():
            def gen(n):
                for i in range(n):
                    yield i * i
            return list(gen(6))
        self._jit_test(f)

    def test_generator_with_send(self):
        def f():
            def accumulator():
                total = 0
                while True:
                    val = yield total
                    if val is None:
                        break
                    total += val
            g = accumulator()
            next(g)  # prime
            results = []
            for v in [10, 20, 30]:
                results.append(g.send(v))
            return results
        self._jit_test(f)

    def test_exception_handling_in_call(self):
        def f():
            def risky(x):
                if x == 0:
                    raise ValueError("zero")
                return 100 // x
            results = []
            for v in [5, 2, 0, 1]:
                try:
                    results.append(risky(v))
                except ValueError as e:
                    results.append(str(e))
            return results
        self._jit_test(f)

    def test_complex_class_hierarchy(self):
        def f():
            class Shape:
                def area(self):
                    return 0
            class Rectangle(Shape):
                def __init__(self, w, h):
                    self.w = w
                    self.h = h
                def area(self):
                    return self.w * self.h
            class Square(Rectangle):
                def __init__(self, side):
                    super().__init__(side, side)
            shapes = [Rectangle(3, 4), Square(5), Shape()]
            return [s.area() for s in shapes]
        self._jit_test(f)

    def test_closure_as_callback(self):
        def f():
            def make_multiplier(n):
                def multiply(x):
                    return x * n
                return multiply
            times3 = make_multiplier(3)
            times7 = make_multiplier(7)
            return (times3(10), times7(10), list(map(times3, [1, 2, 3])))
        self._jit_test(f)

    def test_nested_class_definition(self):
        def f():
            class Outer:
                class Inner:
                    value = 42
                def get_inner_value(self):
                    return self.Inner.value
            o = Outer()
            return (o.get_inner_value(), Outer.Inner.value)
        self._jit_test(f)

    def test_slots_class(self):
        def f():
            class Point:
                __slots__ = ("x", "y")
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            p = Point(3, 4)
            return (p.x, p.y)
        self._jit_test(f)

    def test_class_with_class_var_and_instance_var(self):
        def f():
            class Config:
                default = 10
                def __init__(self, val=None):
                    self.val = val if val is not None else Config.default
            c1 = Config()
            c2 = Config(99)
            return (c1.val, c2.val, Config.default)
        self._jit_test(f)

    def test_method_resolution_order(self):
        def f():
            class A:
                def who(self):
                    return "A"
            class B(A):
                pass
            class C(A):
                def who(self):
                    return "C"
            class D(B, C):
                pass
            return (D().who(), [c.__name__ for c in D.__mro__])
        self._jit_test(f)

    def test_unpacking_in_function_args(self):
        def f():
            def add(a, b, c, d):
                return a + b + c + d
            pair1 = (1, 2)
            pair2 = (3, 4)
            return add(*pair1, *pair2)
        self._jit_test(f)

    def test_dict_merge_in_call(self):
        def f():
            def conf(**kw):
                return sorted(kw.items())
            d1 = {"a": 1, "b": 2}
            d2 = {"c": 3, "d": 4}
            return conf(**d1, **d2)
        self._jit_test(f)

    def test_walrus_operator_in_function(self):
        def f():
            results = []
            data = [1, 5, 3, 8, 2, 9]
            for x in data:
                if (y := x * 2) > 6:
                    results.append(y)
            return results
        self._jit_test(f)

    def test_chained_comparison(self):
        def f():
            def check(x):
                return 1 < x < 10
            return [check(i) for i in range(12)]
        self._jit_test(f)

    def test_multiple_return_values(self):
        def f():
            def divmod_func(a, b):
                return a // b, a % b
            return divmod_func(17, 5)
        self._jit_test(f)

    def test_string_formatting(self):
        def f():
            name = "JIT"
            version = 3
            return (
                f"{name} v{version}",
                "{} v{}".format(name, version),
                "%s v%d" % (name, version),
            )
        self._jit_test(f)

    def test_list_dict_set_comprehensions(self):
        def f():
            lst = [x * x for x in range(10)]
            dct = {x: x * x for x in range(5)}
            st = {x % 3 for x in range(10)}
            gen_sum = sum(x for x in range(100))
            return (lst, dct, sorted(st), gen_sum)
        self._jit_test(f)

    def test_context_manager_protocol(self):
        def f():
            class CM:
                def __init__(self):
                    self.log = []
                def __enter__(self):
                    self.log.append("enter")
                    return self
                def __exit__(self, *args):
                    self.log.append("exit")
                    return False
            cm = CM()
            with cm as c:
                c.log.append("body")
            return cm.log
        self._jit_test(f)

    def test_nested_function_returning_function(self):
        def f():
            def make_adder(n):
                def adder(x):
                    return x + n
                return adder
            add5 = make_adder(5)
            add10 = make_adder(10)
            return (add5(1), add10(1), add5(add10(0)))
        self._jit_test(f)

    def test_closure_over_loop_with_lambda(self):
        def f():
            funcs = [lambda x, i=i: x + i for i in range(5)]
            return [fn(100) for fn in funcs]
        self._jit_test(f)

    def test_recursive_data_processing(self):
        def f():
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            return flatten([1, [2, 3], [4, [5, 6]], 7, [[8]]])
        self._jit_test(f)

    def test_builtin_sorted_with_key(self):
        def f():
            data = ["banana", "apple", "cherry", "date"]
            return (sorted(data), sorted(data, key=len), sorted(data, key=lambda s: s[-1]))
        self._jit_test(f)

    def test_complex_kwargs_dispatch(self):
        def f():
            def make(**kw):
                return kw
            r1 = make(a=1, b=2)
            r2 = make()
            r3 = make(x=10, y=20, z=30)
            return (sorted(r1.items()), sorted(r2.items()), sorted(r3.items()))
        self._jit_test(f)

    def test_method_bound_unbound(self):
        def f():
            class C:
                def method(self, x):
                    return x + 1
            obj = C()
            # Bound method call
            r1 = obj.method(10)
            # Unbound-style call
            r2 = C.method(obj, 20)
            # Method as first-class object
            m = obj.method
            r3 = m(30)
            return (r1, r2, r3)
        self._jit_test(f)

    def test_descriptor_protocol(self):
        def f():
            class Validator:
                def __init__(self, min_val, max_val):
                    self.min_val = min_val
                    self.max_val = max_val
                    self.name = None
                def __set_name__(self, owner, name):
                    self.name = name
                def __get__(self, obj, objtype=None):
                    if obj is None:
                        return self
                    return getattr(obj, f"_{self.name}", self.min_val)
                def __set__(self, obj, value):
                    if value < self.min_val:
                        value = self.min_val
                    elif value > self.max_val:
                        value = self.max_val
                    setattr(obj, f"_{self.name}", value)

            class Settings:
                volume = Validator(0, 100)
                brightness = Validator(0, 255)

            s = Settings()
            s.volume = 50
            s.brightness = 300  # clamped to 255
            return (s.volume, s.brightness)
        self._jit_test(f)


if __name__ == "__main__":
    unittest.main()
