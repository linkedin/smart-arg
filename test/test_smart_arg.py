"""Tests for smart-arg-suite."""
import os
import subprocess
import sys
from typing import List, NamedTuple, Tuple, Optional, Dict, Set

import pytest

from linkedin.smart_arg import arg_suite, custom_arg_suite, LateInit, SmartArgError, TypeHandler, PrimitiveHandlerAddon
from smart_arg_mock import main, MockArgTup


@arg_suite
class MyTupBasic(NamedTuple):
    """
    MyTup docstring goes to description
    """
    a_int: int  # a is int
    a_float: float  # a is float
    a_bool: bool
    a_str: str
    _a_str = {'choices': ['hello', 'bonjour', 'hola']}  # will overwrite the a_str argument choices
    b_list_int: List[int]
    b_set_str: Set[str]
    d_tuple_3: Tuple[int, float, bool]
    d_tuple_2: Tuple[str, int]
    e_dict_str_int: Dict[str, int]
    e_dict_int_bool: Dict[int, bool]
    c_optional_float: Optional[float] = None
    with_default: int = 10


my_tup_basic = MyTupBasic(
    a_int=32,
    a_float=0.3,
    a_bool=True,
    a_str='hello',
    b_list_int=[1, 2, 3],
    b_set_str={'set1', 'set2'},
    c_optional_float=None,
    d_tuple_2=('tuple', 12),
    d_tuple_3=(10, 0.5, False),
    e_dict_str_int={'size': 32, 'area': 90},
    e_dict_int_bool={10: True, 20: False, 30: True})


def test_basic_parse_to_arg():
    with pytest.raises(TypeError, match="missing 10 required keyword arguments"):
        MyTupBasic()
    arg_cmd = 'MyTupBasic+ --a_int 32 --a_float 0.3 --a_bool True --a_str hello --b_list_int 1 2 3 --b_set_str set1 set2 ' + \
              '--d_tuple_3 10 0.5 False --d_tuple_2 tuple 12 --e_dict_str_int size:32 area:90 --e_dict_int_bool 10:True 20:False 30:True MyTupBasic-'
    parsed_arg = MyTupBasic(arg_cmd.split())
    assert parsed_arg == my_tup_basic
    parsed_arg_from_factory: MyTupBasic = MyTupBasic.__from_argv__(arg_cmd.split())
    assert parsed_arg == parsed_arg_from_factory
    serialized_cmd_line = my_tup_basic.__to_argv__(separator=None)
    assert set(serialized_cmd_line) == set(arg_cmd.split())
    my_parser = MyTupBasic.__arg_suite__._parser
    assert my_parser._option_string_actions['--a_int'].help == '(int, required)  a is int'
    assert my_parser._option_string_actions['--a_float'].help == '(float, required)  a is float'
    assert my_parser._option_string_actions['--a_str'].choices == ['hello', 'bonjour', 'hola']


class Muted:
    def write(self, *_):
        pass


from contextlib import redirect_stderr


def test_parse_error():
    with redirect_stderr(Muted()):
        pytest.raises(SystemExit, MyTupBasic, [])


def test_optional():
    @arg_suite
    class MyTup(NamedTuple):
        ints: Optional[List[int]] = None

    with redirect_stderr(Muted()):
        pytest.raises(SystemExit, MyTup, ['--ints', 'None'])
    # assert MyTup(['--ints', 'None']).ints is None
    assert MyTup(['--ints', '1', '2']).ints == [1, 2]
    assert MyTup(['--ints']).ints == []

    class InvalidOptional(NamedTuple):
        no: Optional[int]

    pytest.raises(SmartArgError, arg_suite, InvalidOptional)


def test_post_process():
    @arg_suite
    class MyTup(NamedTuple):
        a_int: Optional[int] = LateInit  # if a_int is not in the argument, post_process will initialize it

    pytest.raises(SmartArgError, MyTup)
    pytest.raises(SmartArgError, MyTup, a_int='not a int')
    pytest.raises(SmartArgError, MyTup, [])
    MyTup.__late_init__ = lambda s: None
    pytest.raises(SmartArgError, MyTup, [])

    def post_process(self):
        return self._replace(a_int=10) if self.a_int is LateInit else self

    MyTup.__late_init__ = post_process
    assert MyTup([]).a_int == 10
    assert MyTup().a_int == 10
    assert MyTup(a_int=0).a_int == 0
    del MyTup.__late_init__
    assert MyTup(a_int=0).a_int == 0


def test_validate():
    @arg_suite
    class MyTup(NamedTuple):
        a_int: int = LateInit  # if a_int is not in the argument, post_process will initialize it

    validated = False

    def validate(s):
        nonlocal validated
        validated = True
        raise AttributeError()

    MyTup.__validate__ = validate
    pytest.raises(AttributeError, MyTup, ['--a_int', '1'])
    assert validated, "`validate` might not be executed."

    @arg_suite
    class MyTuple(NamedTuple):
        abc: str

        def __validate__(self):
            if self.abc != 'abc':
                raise AttributeError()

        @staticmethod
        def format():
            return "hello"

        @classmethod
        def format2(cls):
            return "hello2"

    # no exception for callable methods
    tup = MyTuple(['--abc', 'abc'])
    assert tup.abc == 'abc'
    assert tup.format() == 'hello'
    assert MyTuple.format2() == 'hello2'


def test_validate_fields():
    class DanglingParamOverride(NamedTuple):
        _a_str = "abc"

    class MyNonType(NamedTuple):
        a_str = "abc"

    class MyNonTypeTuple(NamedTuple):
        a_tuple: Tuple

    class TrailingUnderscore(NamedTuple):
        a_: int

    pytest.raises(SmartArgError, arg_suite, DanglingParamOverride)
    pytest.raises(SmartArgError, arg_suite, MyNonType)
    pytest.raises(SmartArgError, arg_suite, MyNonTypeTuple)
    pytest.raises(SmartArgError, arg_suite, TrailingUnderscore)


def test_argv():
    import unittest.mock
    expected_arg = MockArgTup(
        a_bool=False,
        b_list_int=[1, 2, 3],
        d_tuple_2=('hello', 2),
        e_dict_str_int={'size': 32, 'area': 90}
    )
    cmd_line = expected_arg.__to_argv__()
    with unittest.mock.patch('sys.argv', ['mock'] + cmd_line):
        parsed = main(sys.argv[1:])
        parsed_factory = MockArgTup.__from_argv__()

        assert parsed == parsed_factory
        assert parsed == expected_arg


def test_primitive_addon():
    class IntHandlerAddon(PrimitiveHandlerAddon):
        @staticmethod
        def build_primitive(arg_type, kwargs):
            if arg_type is int:
                kwargs.type = lambda s: int(s) ** 2

        handled_types = [int]

    class IntTypeHandler(TypeHandler):
        def _build_common(self, kwargs, field_meta):
            kwargs.help = '(int, squared)'

        handled_types = [int]

    @custom_arg_suite(primitive_handler_addons=[IntHandlerAddon], type_handlers=[IntTypeHandler])
    class MyTuple(NamedTuple):
        a_int: int

    tup = MyTuple(['--a_int', '3'])
    assert tup.a_int == 9
    my_parser = MyTuple.__arg_suite__._parser
    assert my_parser._option_string_actions['--a_int'].help == '(int, squared)'


def test_unsupported_types():
    class MyTuple(NamedTuple):
        a: List[List[int]]

    pytest.raises(SmartArgError, arg_suite, MyTuple)


def test_nested():
    @arg_suite
    class MyTup(NamedTuple):
        def __late_init__(self):
            return self._replace(b_int=10) if self.b_int is LateInit else self

        b_int: int = LateInit  # if a_int is not in the argument, __late_init__ will initialize it

    @arg_suite
    class Nested(NamedTuple):
        a_int: int
        nested: MyTupBasic = my_tup_basic  # nested
        another_int: int = 0
        another_nested: MyTup = MyTup()

    nested = Nested(a_int=0)
    assert nested.another_nested.b_int == 10
    argv = nested.__to_argv__()
    assert Nested(argv) == nested
    assert Nested(argv[0:2]) == nested
    pytest.raises(SmartArgError, Nested, ['--a_int', '0', '--nested', 'raise'])

    class NotDecoratedWithPost(NamedTuple):
        def __validate__(self): pass

    class Nested(NamedTuple):
        nested: NotDecoratedWithPost

    pytest.raises(SmartArgError, arg_suite, Nested)
    del NotDecoratedWithPost.__validate__
    arg_suite(Nested)  # should not raise


def test_cli_execution():
    demo = f'{sys.executable if sys.executable else "python"} {os.path.join(os.path.dirname(__file__), "smart_arg_demo.py")}'
    args = '--nn 200 300 --a_tuple s 5 --encoder fastText --h_param y:1 n:0 --nested.n_nested.n_' \
           'name "nested name" --embedding_dim 100 --lr 0.001'
    arg_line = demo + ' ' + args
    kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'shell': True}
    completed_process = subprocess.run(arg_line, **kwargs)
    assert completed_process.returncode == 0

    completed_process = subprocess.run(arg_line + ' --adp False', **kwargs)
    assert completed_process.returncode == 168

    completed_process = subprocess.run(arg_line + ' --nested "OH NO!"', **kwargs)
    assert completed_process.returncode == 1

    completed_process = subprocess.run(f'{demo} MyTup+ {args} MyTup- --nested "OH NO!"', **kwargs)
    assert completed_process.returncode == 0
