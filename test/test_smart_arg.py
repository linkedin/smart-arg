"""Tests for smart-arg-suite."""
import os
import subprocess
import sys
from dataclasses import dataclass, FrozenInstanceError
from math import sqrt
from types import SimpleNamespace
from typing import List, NamedTuple, Tuple, Optional, Dict, Set
from contextlib import redirect_stderr

import pytest

from smart_arg import arg_suite, custom_arg_suite, LateInit, SmartArgError, TypeHandler, PrimitiveHandlerAddon
from smart_arg_mock import main, MockArgTup


@arg_suite
@dataclass(frozen=True)
class MyTupBasic:
    """
    MyTup docstring goes to description
    """
    a_int: int  # a is int
    a_float: float  # a is float
    a_bool: bool
    a_str: str
    __a_str = {'choices': ['hello', 'bonjour', 'hola']}  # will overwrite the a_str argument choices
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
    with pytest.raises(TypeError, match="missing 10"):
        MyTupBasic()
    arg_cmd = 'MyTupBasic+ --a_int 32 --a_float 0.3 --a_bool True --a_str hello --b_list_int 1 2 3 --b_set_str set1 set2 ' + \
              '--d_tuple_3 10 0.5 False --d_tuple_2 tuple 12 --e_dict_str_int size:32 area:90 --e_dict_int_bool 10:True 20:False 30:True MyTupBasic-'
    parsed_arg_from_factory: MyTupBasic = MyTupBasic.__from_argv__(arg_cmd.split())
    assert my_tup_basic == parsed_arg_from_factory
    serialized_cmd_line = my_tup_basic.__to_argv__(separator=None)
    assert set(serialized_cmd_line) == set(arg_cmd.split())
    my_parser = MyTupBasic.__arg_suite__._parser
    assert my_parser._option_string_actions['--a_int'].help == '(int, required)  a is int'
    assert my_parser._option_string_actions['--a_float'].help == '(float, required)  a is float'
    assert my_parser._option_string_actions['--a_str'].choices == ['hello', 'bonjour', 'hola']

    parsed_arg = MyTupBasic(arg_cmd.split())
    assert parsed_arg == my_tup_basic


muted = redirect_stderr(SimpleNamespace(write=lambda *_: None))


def test_parse_error():
    with muted: pytest.raises(SystemExit, MyTupBasic.__from_argv__, [])


def test_optional():
    @arg_suite
    class MyTup(NamedTuple):
        ints: Optional[List[int]] = None

    with muted: pytest.raises(SystemExit, MyTup.__from_argv__, ['--ints', 'None'])
    assert MyTup.__from_argv__([]).ints is None
    assert MyTup.__from_argv__(['--ints', '1', '2']).ints == [1, 2]
    assert MyTup.__from_argv__(['--ints']).ints == []

    class InvalidOptional(NamedTuple):
        no: Optional[int]

    pytest.raises(SmartArgError, arg_suite, InvalidOptional)


def test_post_process():
    @arg_suite
    class MyTup(NamedTuple):
        a_int: Optional[int] = LateInit  # if a_int is not in the argument, post_process will initialize it

    pytest.raises(SmartArgError, MyTup)
    pytest.raises(SmartArgError, MyTup, a_int='not a int')
    pytest.raises(SmartArgError, MyTup.__from_argv__, [])

    @arg_suite
    class MyTup(NamedTuple):
        a_int: Optional[int] = LateInit  # if a_int is not in the argument, post_process will initialize it

        def __post_init__(self):
            self.a_int = 10 if self.a_int is LateInit else self.a_int

    assert MyTup.__from_argv__([]).a_int == 10
    assert MyTup().a_int == 10
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

    MyTup.__post_init__ = validate
    pytest.raises(AttributeError, MyTup.__from_argv__, ['--a_int', '1'])
    assert validated, "`validate` might not be executed."

    @arg_suite
    class MyTuple(NamedTuple):
        abc: str

        def __post_init__(self):
            if self.abc != 'abc':
                raise AttributeError()
            return self

        @staticmethod
        def format():
            return "hello"

        @classmethod
        def format2(cls):
            return "hello2"

        format3 = lambda: "hello3"

    # no exception for callable methods
    tup = MyTuple.__from_argv__(['--abc', 'abc'])
    assert tup.abc == 'abc'
    assert tup.format() == 'hello'
    assert MyTuple.format2() == 'hello2'
    assert MyTuple.format3() == 'hello3'


def test_validate_fields():
    class DanglingParamOverride(NamedTuple):
        __a_str = "abc"

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

        @staticmethod
        def str(arg) -> str:
            return str(int(sqrt(arg)))

        handled_types = [int]

    class IntTypeHandler(TypeHandler):
        def _build_common(self, kwargs, field_meta):
            kwargs.help = '(int, squared)'

        handled_types = [int]

    @custom_arg_suite(primitive_handler_addons=[IntHandlerAddon], type_handlers=[IntTypeHandler])
    class MyTuple(NamedTuple):
        a_int: int

    argv = ['--a_int', '3']
    tup = MyTuple.__from_argv__(argv)
    assert tup.a_int == 9
    assert tup.__to_argv__() == argv
    my_parser = MyTuple.__arg_suite__._parser
    assert my_parser._option_string_actions['--a_int'].help == '(int, squared)'


def test_unsupported_types():
    class MyTuple(NamedTuple):
        a: List[List[int]]

    pytest.raises(SmartArgError, arg_suite, MyTuple)


def test_nested():
    @arg_suite
    class MyTup(NamedTuple):
        def __post_init__(self):
            self.b_int = 10 if self.b_int is LateInit else self.b_int

        b_int: int = LateInit  # if a_int is not in the argument, __post_init__ will initialize it

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
        def __post_init__(self): pass

    class Nested(NamedTuple):
        nested: NotDecoratedWithPost

    # Nested class is not allowed to have __post_init__ if not decorated
    pytest.raises(SmartArgError, arg_suite, Nested)
    del NotDecoratedWithPost.__post_init__
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


def test_dataclass():
    @dataclass
    class GdmixParams:
        ACTIONS = ("action_inference", "action_train")
        action: str = ACTIONS[1]  # Train or inference.
        __action = {"choices": ACTIONS}
        STAGES = ("fixed_effect", "random_effect")
        stage: str = STAGES[0]  # Fixed or random effect.
        __stage = {"choices": STAGES}
        MODEL_TYPES = ("logistic_regression", "detext")
        model_type: str = MODEL_TYPES[0]  # The model type to train, e.g, logistic regression, detext, etc.
        __model_type = {"choices": MODEL_TYPES}

        # Input / output files or directories
        training_output_dir: Optional[str] = None  # Training output directory.
        validation_output_dir: Optional[str] = None  # Validation output directory.

        # Driver arguments for random effect training
        partition_list_file: Optional[str] = None  # File containing a list of all the partition ids, for random effect only

        def __post_init__(self):
            assert self.action in self.ACTIONS, "Action must be either train or inference"
            assert self.stage in self.STAGES, "Stage must be either fixed_effect or random_effect"
            assert self.model_type in self.MODEL_TYPES, "Model type must be either logistic_regression or detext"

    @dataclass
    class SchemaParams:
        # Schema names
        sample_id: str  # Sample id column name in the input file.
        sample_weight: Optional[str] = None  # Sample weight column name in the input file.
        label: Optional[str] = None  # Label column name in the train/validation file.
        prediction_score: Optional[str] = None  # Prediction score column name in the generated result file.
        prediction_score_per_coordinate: str = "predictionScorePerCoordinate"  # ColumnName of the prediction score without the offset.

    @arg_suite
    @dataclass
    class Params(GdmixParams, SchemaParams):
        """GDMix Driver"""

        def __post_init__(self):
            super().__post_init__()
            assert (self.action == self.ACTIONS[1] and self.label) or (self.action == self.ACTIONS[0] and self.prediction_score)
            self.prediction_score = self.prediction_score

    argv = ['--sample_id', 'uid', '--sample_weight', 'weight', '--feature_bags', 'global', '--train_data_path',
            'resources/train', '--validation_data_path',
            'resources/validate', '--model_output_dir', 'dummy_model_output_dir', '--metadata_file',
            'resources/fe_lbfgs/metadata/tensor_metadata.json', '--feature_file',
            'test/resources/fe_lbfgs/featureList/global']
    with muted: pytest.raises(SystemExit, Params.__from_argv__, argv)
    pytest.raises(AssertionError, Params.__from_argv__, argv, error_on_unknown=False)
    args: Params = Params.__from_argv__(argv + ['--label', 'bluh'], error_on_unknown=False)
    pytest.raises(AssertionError, Params, sample_id='uid', action='no_such_action')
    pytest.raises(FrozenInstanceError, args.__post_init__)  # mutation not allowed after init
    object.__delattr__(args, '__frozen__')
    args.__post_init__()  # mutation allowed after '__frozen__` mark removed
    assert args == Params(sample_id='uid',  sample_weight='weight', label='bluh')
    assert args.__to_argv__() == ['--sample_id', 'uid', '--sample_weight', 'weight', '--label', 'bluh']

    @arg_suite
    @dataclass
    class NoPostInit:
        def mutate(self):
            self.frozen = False
        frozen: bool = True

    pytest.raises(FrozenInstanceError, NoPostInit().mutate)  # mutation not allowed after init
