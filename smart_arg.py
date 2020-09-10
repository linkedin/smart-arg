"""Smart Argument Suite

This module is an argument serialization and deserialization library that:

    - handles two-way conversions: a typed immutable argument object (A `NamedTuple` or `dataclass` instance) <==> a command-line argv
    - enables IDE type hints and code auto-completion by using `NamedTuple` or `dataclass`
    - promotes type-safety of cli-ready arguments


The following is a simple usage example::

    # Define the argument class:
    @arg_suite
    class MyArg(NamedTuple):
        '''MyArg is smart and the docstring goes to description'''

        nn: List[int]  # Comments go to ArgumentParser argument help
        # Each named (without "_" prefix) and fully typed attribute will be turned into an ArgumentParser.add_argument
        # DISCOURAGED: This only overrides one-way conversion (deserialization of cli) behavior and introduces inconsistencies.
        _nn = {'choices': (200, 300)}  # (Advanced feature) Optional user supplement/override for `ArgumentParser.add_argument("--nn", **(kwargs|_nn))`

        a_tuple: Tuple[str, int]  # Arguments without defaults are treated as "required = True"
        h_param: Dict[str, int]  # Also supports List, Set

        ###### arguments without defaults go first as required by NamedTuple ######

        l_r: float = 0.01 # Arguments with defaults are treated as "required = False"
        n: Optional[int] = None  # Optional can only default to `None`


    # Create the corresponding argument class instance from the command line argument list:
    # by using
    arg: ArgClass = ArgClass.__from_argv__(sys.argv[1:])  # with factory method, need the manual type hint `:ArgClass` to help IDE
    # or the monkey-patched constructor of the annotated argument class
    arg = ArgClass(sys.argv[1:])  # with monkey-patched constructor with one positional argument, no manual hint needed

    # Create a NamedTuple argument instance and generate its command line counterpart:
    # the monkey-patched constructor only take named arguments for directly creating the NamedTuple
    arg = ArgClass(nn=[200, 300], a_tuple=('t', 0), h_param={'batch_size': 1000})
    # generate the corresponding command line argument list
    arg.__to_argv__()


The module contains the following public classes:

- `arg_suite` -- The main entry point for Smart Argument Suite. As the
    example above shows, this decorator will attach an `ArgSuite` instance to
    the argument `NamedTuple` or `dataclass`.

- `PrimitiveHandlerAddon` -- The base class to deal some basic operation on primitive types,
    and users can implement their owns to change the behavior.

- `TypeHandler` -- The base class to handle types, users can extend to expand the
    support or change existing behaviors.

All other classes and methods in this module are considered implementation details."""

__version__ = '0.2.*'
__all__ = (
    'arg_suite',
    'custom_arg_suite',
    'LateInit',
    'SmartArgError',
    'TypeHandler',
    'PrimitiveHandlerAddon',
)

import logging
import os
import sys
import tokenize
import warnings
from argparse import Action, ArgumentParser
from types import SimpleNamespace as KwargsType
from typing import Any, Callable, Dict, Generic, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar, Union
from enum import EnumMeta, Enum

ArgType = TypeVar('ArgType', bound=NamedTuple)  # NamedTuple is not a real class bound, but setting `bound` to NamedTuple makes mypy happier
NoneType = None.__class__
FieldMeta = NamedTuple('FieldMeta', (('comment', str), ('default', Any), ('type', Type), ('optional', bool)))

logger = logging.getLogger(__name__)
SMART_ARG_LOG_LEVEL = 'SMART_ARG_LOG_LEVEL'
if SMART_ARG_LOG_LEVEL in os.environ:
    logger.addHandler(logging.StreamHandler())
    log_level = os.environ[SMART_ARG_LOG_LEVEL].upper()
    logger.setLevel(log_level)
    logger.info(f"Detected environment var `{SMART_ARG_LOG_LEVEL}, set log level to '{log_level}' and log to stream.")

if sys.version_info >= (3, 7):
    # Python == 3.7.x. Defining the back-ported get_origin, get_args
    # 3.7 `List.__origin__ == list`
    get_origin, get_args = lambda tp: getattr(tp, '__origin__', None), lambda tp: getattr(tp, '__args__', ())
elif sys.version_info >= (3, 6):
    # Python == 3.6.x. Defining the back-ported get_origin, get_args
    # 3.6 `List.__origin__ == List`, `Optional` does not have `__dict__`
    get_origin, get_args = lambda tp: getattr(tp, '__extra__', ()) or getattr(tp, '__origin__', None), lambda tp: getattr(tp, '__args__', ()) or []
else:
    try:
        warnings.warn(f"Unsupported and untested python version {sys.version_info} < 3.6. "
                      f"The package may or may not work properly. "
                      f"Try 'from typing_inspect import get_origin, get_args' now.")
        from typing_inspect import get_args, get_origin
    except ImportError:
        warnings.warn(f"`from typing_inspect import get_origin, get_args` failed for "
                      f"unsupported python version {sys.version_info} < 3.6. It might work with 'pip install typing_inspect'.")
        raise

_black_hole = lambda *args, **kwargs: None
_mro_all = lambda arg_class, get_dict: {k: v for b in (*arg_class.__mro__[-1:0:-1], arg_class) if b.__module__ != 'builtins' for k, v in get_dict(b).items()}
# note: Fields without type annotation won't be regarded as a property/entry _fields.
_annotations = lambda arg_class: _mro_all(arg_class, lambda b: getattr(b, '__annotations__', {}))
try:
    from dataclasses import MISSING, asdict, is_dataclass
except ImportError:
    logger.warning("Importing dataclasses failed. You might need to 'pip install dataclasses' on python 3.6.")
    class MISSING: pass  # type: ignore # noqa: E701
    is_dataclass = _black_hole  # type: ignore # Always return None


class LateInit:
    """special singleton/class to mark late initialized fields"""


class SmartArgError(Exception):  # TODO Extend to better represent different types of errors.
    """Base exception for smart-arg."""


def _raise_if(message: str, condition: Any = True):
    if condition:
        raise SmartArgError(message)


def _first_handles(with_handles, arg_type, default: Optional[bool] = None):
    for h in with_handles:
        if h.handles(arg_type):
            return h
    return _raise_if(f"{arg_type!r} is not supported.") if default is None else default


class PrimitiveHandlerAddon:
    """Primitive handler addon defines some basic operations on primitive types. Only `staticmethod`s are expected.
    Users can extend/modify the primitive handling by inheriting this class."""
    @staticmethod
    def build_type(arg_type: Type) -> Union[Type, Callable[[str], Any]]:
        return (lambda s: True if s == 'True' else False if s == 'False' else _raise_if(f"Invalid bool: {s!r}")) if arg_type is bool else \
            (lambda s: getattr(arg_type, s, None) or _raise_if(f"Invalid enum {s!r} for {arg_type!r}")) if type(arg_type) is EnumMeta else \
            arg_type

    @staticmethod
    def build_str(arg: Any) -> str:
        """Define to serialize the `arg` to a string.

        :param arg: The argument
        :type arg: Any type that is supported by this class
        :return: The string serialization of `arg`"""
        return str(arg.name) if isinstance(arg, Enum) else str(arg)

    @staticmethod
    def build_metavar(arg_type: Type) -> str:
        """Define the hint string in argument help message for `arg_type`."""
        return '{True, False}' if arg_type == bool else \
            f"{{{', '.join(str(c) for c in arg_type._member_names_)}}}" if type(arg_type) == EnumMeta else \
            arg_type.__name__

    @staticmethod
    def build_choices(arg_type) -> Optional[Iterable[Any]]:
        """Enumerate `arg_type` if possible, or return `None`."""
        return (True, False) if arg_type == bool else \
            arg_type if type(arg_type) == EnumMeta else \
            None

    @staticmethod
    def handles(t: Type) -> bool:
        return t in (str, int, float, bool) or type(t) is EnumMeta


class TypeHandler:
    """Base type handler"""
    def __init__(self, primitive_addons: Sequence[Type[PrimitiveHandlerAddon]]):
        self.primitive_addons = primitive_addons

    def _build_common(self, kwargs: KwargsType, field_meta: FieldMeta) -> None:
        """Build `help`, `default` and `required` for keyword arguments for ArgumentParser.add_argument

        :param kwargs: the keyword argument KwargsType object
        :param field_meta: the meta information extracted from the NamedTuple class"""
        # Build help message
        arg_type = field_meta.type
        help_builder = ['(', 'Optional[' if field_meta.optional else '', self._type_to_str(arg_type), ']' if field_meta.optional else '']
        # Get default if specified and set required if no default
        if field_meta.default is MISSING:
            kwargs.required = True
            help_builder.append('; required')
        else:
            # Only add default to the help message for informational purpose. The default is set when creating the argument class instance.
            help_builder.append(', default: ')
            help_builder.append(str(field_meta.default))

        help_builder.append(') ')
        # Add from source code comment
        help_builder.append(field_meta.comment)
        kwargs.help = ''.join(help_builder)

    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        """Build `nargs`, `type` and `metavar` for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (primitive types)"""

    def gen_kwargs(self, field_meta: FieldMeta) -> KwargsType:
        """Build keyword argument object KwargsType

        :param field_meta: argument metadata information
        :return: keyword argument object"""
        kwargs = KwargsType()
        self._build_common(kwargs, field_meta)
        self._build_other(kwargs, field_meta.type)
        return kwargs

    def gen_cli_arg(self, action: Action, arg: Any) -> Iterable[str]:
        """Generate command line for argument

        :param action: action object stored for the argument
        :param arg: value of the argument
        :return: iterable command line str"""
        yield action.option_strings[0]
        args = (arg,) if isinstance(arg, str) or not isinstance(arg, Iterable) else arg
        yield from (_first_handles(self.primitive_addons, type(arg)).build_str(arg) for arg in args)

    def _type_to_str(self, t: Union[type, Type]) -> str:
        """Convert type to string for ArgumentParser help message

        :param t: type of the argument, i.e. float, Dict[str, int], Set[int], List[str] etc.
        :return: string representation of the argument type"""
        return f'{getattr(t, "_name", "") or t.__name__}[{", ".join(a.__name__ for a in get_args(t))}]'

    def handles(self, t: Type) -> bool:
        raise NotImplementedError


class PrimitiveHandler(TypeHandler):
    def handles(self, t: Type) -> bool:
        return _first_handles(self.primitive_addons, t, False)

    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        addon = _first_handles(self.primitive_addons, arg_type)
        kwargs.type = addon.build_type(arg_type)
        kwargs.metavar = addon.build_metavar(arg_type)
        kwargs.choices = addon.build_choices(arg_type)

    def _type_to_str(self, t: Union[type, Type]) -> str:
        return t.__name__


class TupleHandler(TypeHandler):
    class __BuildType:
        def __init__(self, types, p_addons):
            self.counter, self.types, self.p_addons = 0, types, p_addons

        def __call__(self, s):
            if self.counter == len(self.types):
                self.counter = 0
            t = self.types[self.counter]
            self.counter += 1
            return _first_handles(self.p_addons, t).build_type(t)(s)

    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        # get the tuple element types
        types = get_args(arg_type)
        kwargs.nargs = len(types)
        kwargs.metavar = tuple(_first_handles(self.primitive_addons, t).build_metavar(t) for t in types)
        kwargs.type = TupleHandler.__BuildType(types, self.primitive_addons)

    def handles(self, t: Type) -> bool:
        return get_origin(t) == tuple and get_args(t)  # type: ignore


class CollectionHandler(TypeHandler):
    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        kwargs.nargs = '*'
        unboxed_type = get_args(arg_type)[0]
        addon = _first_handles(self.primitive_addons, unboxed_type)
        kwargs.metavar = addon.build_metavar(unboxed_type)
        kwargs.type = addon.build_type(unboxed_type)

    def handles(self, t: Type) -> bool:
        args = get_args(t)
        return len(args) == 1 and get_origin(t) in (list, set) and _first_handles(self.primitive_addons, args[0], False)


class DictHandler(TypeHandler):
    def _build_other(self, kwargs, arg_type) -> None:
        kwargs.nargs = '*'
        arg_types = get_args(arg_type)
        addon_method = lambda type, method: getattr(_first_handles(self.primitive_addons, type), method)(type)  # Find the addon for a type and a method
        kv_apply = lambda method: (addon_method(arg_types[0], method), addon_method(arg_types[1], method))  # Apply the same method on a k/v pair of types
        k_type, v_type = kv_apply('build_type')

        def dict_type(s: str):
            k, v = s.split(":")
            return k_type(k), v_type(v)
        kwargs.type = dict_type
        k, v = kv_apply('build_metavar')
        kwargs.metavar = f'{k}:{v}'

    def gen_cli_arg(self, action: Action, arg):
        yield action.option_strings[0]
        arg_to_str = lambda arg_v: _first_handles(self.primitive_addons, type(arg_v)).build_str(arg_v)
        yield from (f'{arg_to_str(k)}:{arg_to_str(v)}' for k, v in arg.items())

    def handles(self, t: Type) -> bool:
        args, addons = get_args(t), self.primitive_addons
        return len(args) == 2 and get_origin(t) == dict and _first_handles(addons, args[0], False) and _first_handles(addons, args[1], False)


class _namedtuple:  # TODO expand lambdas to static methods or use a better holder representation
    """A NamedTuple proxy, helper function holder for NamedTuple support"""
    @staticmethod
    def new_instance(arg_class, kwargs):
        """:return A new instance of `arg_class`: call original __new__ -> __post_init__ -> post_validation"""
        new_instance = arg_class.__original_new__(arg_class, **kwargs)
        post_init = getattr(arg_class, '__post_init__', None)
        if post_init:
            fake_namedtuple = KwargsType(**new_instance._asdict())
            post_init(fake_namedtuple)  # make the faked NamedTuple mutable in post_init only while initialization
            new_instance = arg_class.__original_new__(arg_class, **vars(fake_namedtuple))
        arg_class.__arg_suite__.post_validation(new_instance)
        return new_instance

    @staticmethod
    def proxy(t: Type):
        """:return This proxy class if `t` is a `NamedTuple` or `None`"""
        b, f, f_t = getattr(t, '__bases__', []), getattr(t, '_fields', []), getattr(t, '__annotations__', {})
        return _namedtuple if (len(b) == 1 and b[0] == tuple and isinstance(f, tuple) and isinstance(f_t, dict)
                               and all(type(n) == str for n in f) and all(type(n) == str for n, _ in f_t.items())) else None
    asdict = lambda args: args._asdict()
    field_default = lambda arg_class, raw_arg_name: arg_class._field_defaults.get(raw_arg_name, MISSING)
    patch = _black_hole


class _dataclasses:
    @staticmethod
    def patch(cls):
        """Patch the argument dataclass so that `post_validation` is called and it's immutable if not `frozen` after initialization"""
        def throw_if_frozen(self, fun, name, *args, **kwargs):
            _raise_if(f"cannot assign to/delete field {name!r}", getattr(self, '__frozen__', False))
            getattr(object, fun)(self, name, *args, **kwargs)

        def init(self, *args, **kwargs):
            if args and hasattr(self, '__frozen__'):
                logger.debug(f"Assuming {self} is from patch new with __from_argv__, already initialized, skipping init.")
                return
            self.__original_init__(*args, **kwargs)
            object.__setattr__(self, '__frozen__', True)
            self.__class__.__arg_suite__.post_validation(self)
        cls.__init__, cls.__original_init__ = init, cls.__init__
        cls.__setattr__ = lambda self, name, value: throw_if_frozen(self, '__setattr__', name, value)
        cls.__delattr__ = lambda self, name, : throw_if_frozen(self, '__delattr__', name)
    proxy = lambda t: _dataclasses if is_dataclass(t) else None
    asdict = lambda args: asdict(args)
    field_default = lambda arg_class, raw_arg_name: arg_class.__dataclass_fields__[raw_arg_name].default
    new_instance = lambda arg_class, _: arg_class.__original_new__(arg_class)


def _get_type_proxy(arg_class):
    return _namedtuple.proxy(arg_class) or _dataclasses.proxy(arg_class)


class ArgSuite(Generic[ArgType]):
    """Generates the corresponding `ArgumentParser` and handles the two-way conversions."""
    @staticmethod
    def new_arg(arg_class, *args, **kwargs):
        """Monkey-Patched argument class __new__.
        If any positional arguments exist, it would assume that the user is trying to parse a sequence of strings.
        It would also assume there is only one positional argument, and raise an `SmartArgError` otherwise.
        If no positional arguments exist, it would call the argument class instance creator with all keyword arguments.

        :param arg_class: Decorated class
        :param args: Optional positional argument, to be parsed to the arg_class type.
               `args[0]`: a optional marker to mark the sub-sequence of `argv` to be parsed by the parser. ``None`` will
                be interpreted as ``sys.argv[1:]``
               `args[1]`: default to `None`, indicating using the default separator for the argument class

        :type `(Optional[Sequence[str]], Optional[str])`
        :param kwargs: Optional keyword arguments, to be passed to the argument class specific instance creator."""
        logger.debug(f"Patched new for {arg_class} is called with {args} and {kwargs}.")
        if args:
            warnings.warn(f"Calling the patched constructor of {arg_class} with argv is deprecated, please use {arg_class}.__from_argv__ instead.")
            # TODO Exception handling with helpful error message
            _raise_if(f"Calling '{arg_class}(positional {args}, keyword {kwargs})' is not allowed:\n"
                      f"Only accept positional arguments to parse to the '{arg_class}'\nkeyword arguments can only be used to create an instance directly.",
                      kwargs or len(args) > 2 or len(args) == 2 and args[1].__class__ not in (NoneType, str)
                      or not (args[0] is None or isinstance(args[0], Sequence) and all(a.__class__ is str for a in args[0])))

            return arg_class.__from_argv__(args[0])
        else:
            return _get_type_proxy(arg_class).new_instance(arg_class, kwargs)

    def __init__(self, type_handlers: Sequence[TypeHandler], arg_class):
        type_proxy = _get_type_proxy(arg_class)
        _raise_if(f"Unsupported argument class {arg_class}.", not type_proxy)
        self.handlers = type_handlers
        self.handler_actions: Dict[str, Union[Tuple[TypeHandler, Action], Tuple[None, Any]]] = {}
        type_proxy.patch(arg_class)
        #  A big assumption here is that the argument classes never override __new__
        if not hasattr(arg_class, '__original_new__'):
            arg_class.__original_new__ = arg_class.__new__
        arg_class.__new__ = ArgSuite.new_arg
        arg_class.__to_argv__ = lambda arg_self, separator='': self.to_cmd_argv(arg_self, separator)  # arg_class instance level method
        arg_class.__from_argv__ = self.parse_to_arg
        arg_class.__arg_suite__ = self
        self._arg_class = arg_class
        self._parser = ArgumentParser(description=self._arg_class.__doc__, argument_default=MISSING, fromfile_prefix_chars='@')  # type: ignore
        self._parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()  # type: ignore
        self._gen_arguments_from_class(self._arg_class, '', True, [], type_proxy)

    def _validate_fields(self, arg_class: Type) -> None:
        """Validate fields in `arg_class`.

        :raise: SmartArgError if the decorated argument class has non-typed field with defaults and such field
                does not startswith "_" to overwrite the existing argument field property."""
        arg_fields = _annotations(arg_class).keys()
        invalid_fields = tuple(filter(lambda s: s.endswith('_'), arg_fields))
        _raise_if(f"'{arg_class}': found invalid (ending with '_') fields : {invalid_fields}.", invalid_fields)
        private_prefix = f'_{arg_class.__name__}__'
        l_prefix = len(private_prefix)
        # skip callable methods and typed fields
        for f in [f for f in (vars(arg_class).keys()) if not callable(getattr(arg_class, f)) and f not in arg_fields]:
            is_private = f.startswith(private_prefix)
            _raise_if(f"'{arg_class}': there is no field '{f[l_prefix:]}' for '{f}' to override.", is_private and f[l_prefix:] not in arg_fields)
            _raise_if(f"'{arg_class}': found invalid (untyped) field '{f}'.", not (is_private or f.startswith('_')))

    def _gen_arguments_from_class(self, arg_class, prefix: str, parent_required, arg_classes: List, type_proxy) -> None:
        """Add argument to the self._parser for each field in the self._arg_class
        :raise: SmartArgError if cannot find corresponding handler for the argument type"""
        suite: ArgSuite = getattr(arg_class, '__arg_suite__', None)
        _raise_if(f"Recursively nested argument class '{arg_class}' is not supported.", arg_class in arg_classes)
        _raise_if(f"Nested argument class '{arg_class}' with '__post_init__' expected to be decorated.",
                  not (suite and suite._arg_class is arg_class) and hasattr(arg_class, '__post_init__'))
        self._validate_fields(arg_class)
        comments = _mro_all(arg_class, self.get_comments)
        for raw_arg_name, arg_type in _annotations(arg_class).items():
            arg_name = f'{prefix}{raw_arg_name}'
            try:
                default = type_proxy.field_default(arg_class, raw_arg_name)
                type_origin, type_args, optional = get_origin(arg_type), get_args(arg_type), False
                if type_origin == Union and len(type_args) == 2 and type_args[1] == NoneType:  # `Optional` support
                    arg_type, optional = type_args[0], True  # Unwrap `Optional` and validate
                    _raise_if(f"Optional field: {arg_name!r}={default!r} must default to `None` or `LateInit`", default not in (None, LateInit))
                sub_type_proxy = _get_type_proxy(arg_type)
                if sub_type_proxy:
                    required = parent_required and default is MISSING
                    arg_classes.append(arg_class)
                    self._gen_arguments_from_class(arg_type, f'{arg_name}.', required, arg_classes, sub_type_proxy)
                    arg_classes.pop()
                    kwargs = KwargsType(required=False,
                                        metavar='Help message only. Do NOT attempt to specify, or an exception will be raised.',
                                        type=lambda _: _raise_if(f"Nested argument {arg_name!r} can not be directly specified"),
                                        help=f"""This is a placeholder for the nested argument '{arg_name}'.
                                             Its parent is {'' if parent_required else 'not'} required.
                                             {"It's required" if default is MISSING else f"Not required with default: {default}"},
                                             if the parent is being parsed.""")
                    self._parser.add_argument(f'--{arg_name}', **vars(kwargs))
                    self.handler_actions[arg_name] = None, default
                else:
                    handler = _first_handles(self.handlers, arg_type)
                    field_meta = FieldMeta(comment=comments.get(raw_arg_name, ''), default=default, type=arg_type, optional=optional)
                    kwargs = handler.gen_kwargs(field_meta)
                    # apply user override to the argument object
                    kwargs.__dict__.update(getattr(arg_class, f'_{arg_class.__name__}__{raw_arg_name}', {}))
                    if not parent_required and hasattr(kwargs, 'required'):
                        del kwargs.required
                    kwargs.default = MISSING  # Marker for fields absent from parsing
                    logger.debug(f"Adding kwargs {kwargs}")
                    self.handler_actions[arg_name] = handler, self._parser.add_argument(f'--{arg_name}', **vars(kwargs))
            except BaseException as b_e:
                logger.critical(f"Failed creating argument parser for {arg_name!r}:{arg_type!r} with exception {b_e}.")
                raise

    @staticmethod
    def strip_argv(separator: str, argv: Optional[Sequence[str]] = None) -> Sequence[str]:
        """Strip any elements outside of `{separator}+` and `{separator}-` of `argv`.
        :param argv: Input argument list, treated as `sys.argv[1:]` if `None`
        :return: Stripped `argv`"""
        if argv is None:
            argv = sys.argv[1:]
        l_s, r_s = separator + '+', separator + '-'
        lc, rc = argv.count(l_s), argv.count(r_s)
        if lc == rc:
            if lc == 0:
                return argv
            elif lc == 1:
                b, e = argv.index(l_s), argv.index(r_s)
                if e > b:
                    return argv[b + 1: e]
        raise SmartArgError(f"Expecting up to 1 pair of separator '{l_s}' and '{r_s}' in {argv}")

    def parse_to_arg(self, argv: Optional[Sequence[str]] = None, separator: Optional[str] = '', *, error_on_unknown: bool = True) -> ArgType:
        """Parse the command line to decorated ArgType

        :param separator: Optional marker to mark the sub-sequence of `argv` ['{separator}+' to '{separator}-'] to parse
        :param error_on_unknown: When `True`, raise if there is any unknown argument in the marked sub-sequence.
        :param argv: the command line list
        :return: parsed decorated object from command line"""
        argv = self.strip_argv(separator or self._arg_class.__name__, argv)
        ns, unknown = self._parser.parse_known_args(argv)
        error_on_unknown and unknown and self._parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        logger.info(f"{argv} is parsed to {ns}")
        # arg_dict = vars(ns)  # {name: value for name, value in vars(ns).items() if value is not NO_DEFAULT}
        arg_dict = {name: value for name, value in vars(ns).items() if value is not MISSING}

        def to_arg(arg_class: Type[ArgType], prefix) -> ArgType:
            nest_arg = {}
            for raw_arg_name, arg_type in _annotations(arg_class).items():
                arg_name = f'{prefix}{raw_arg_name}'
                arg_type_flag, default = self.handler_actions[arg_name]
                if arg_type_flag:
                    value = arg_dict.get(arg_name, MISSING)
                    type_origin = get_origin(arg_type)
                    type_args = get_args(arg_type)
                    type_to_new = get_origin(type_args[0]) if type_origin == Union and len(type_args) == 2 and type_args[1] == NoneType else type_origin
                    # argparse reading variable length arguments are all lists, need to apply the origin type for the conversion to correct type.
                    value = type_to_new(value) if value is not None and isinstance(value, List) else value
                else:
                    is_nested = any((name for name in arg_dict.keys() if name.startswith(arg_name)))
                    value = to_arg(arg_type, f'{arg_name}.') if is_nested else default
                if value is not MISSING:
                    nest_arg[raw_arg_name] = value
            return arg_class(**nest_arg)
        return to_arg(self._arg_class, '')

    def post_validation(self, arg: ArgType, prefix: str = '') -> None:
        """This is called after __post_init__ to validate the fields."""
        arg_class = arg.__class__
        for name, t in _annotations(arg_class).items():
            attr = getattr(arg, name)
            _raise_if(f"Field '{name}' is still not initialized after post processing for {arg_class}", attr is LateInit)
            attr_class = attr.__class__
            type_origin, type_args = get_origin(t), get_args(t)
            if _get_type_proxy(t) or type_origin == Union and len(type_args) == 2 and type_args[1] == NoneType and _get_type_proxy(type_args[0]):
                _raise_if(f"Field {name}' value of {attr}:{attr_class} is not of the expected type '{t}' for {arg_class}", attr_class not in (t, *type_args))
                self.post_validation(attr, f'{prefix}{name}.')
            else:
                arg_type = get_origin(t) or t
                if arg_type == get_origin(Optional[Any]):
                    arg_type = get_args(t)
                try:
                    conforming = isinstance(attr, arg_type)  # e.g. list and List
                except TypeError:  # best effort to check the instance type
                    logger.warning(f"Unable to check if {attr} is of type {t} for field {name} of argument class {arg_class}")
                    conforming = True
                _raise_if(f"Field {name} has value of {attr} of type {attr_class} which is not of the expected type '{t}' for {arg_class}", not conforming)

    def _gen_cmd_argv(self, args: ArgType, prefix) -> Iterable[str]:
        proxy = _get_type_proxy(args.__class__)
        for name, arg in proxy.asdict(args).items():
            default = proxy.field_default(args.__class__, name)
            if arg != default:
                handler, action = self.handler_actions[f'{prefix}{name}']
                yield from handler.gen_cli_arg(action, arg) if handler else self._gen_cmd_argv(arg, f'{prefix}{name}.')

    @staticmethod
    def get_comments(arg_cls: Type[ArgType]) -> Dict[str, str]:
        """Get in-line comments for the input class fields. Only single line of trailing comment is supported.

        :param arg_cls: the input class
        :return: a dictionary with key of class field name and value of in-line comment"""
        def line_tokenizer(lines):
            def source():
                line_tokenizer.index, line = next(lines)
                return line
            source()  # skip the `class` line and set source.index
            last_index = line_tokenizer.index
            for t in tokenize.generate_tokens(source):
                if last_index != line_tokenizer.index:
                    yield '\n'  # New line indicator
                yield t
                last_index = line_tokenizer.index
        comments = {}
        field_column = field = last_token = None
        import inspect
        for token in line_tokenizer(lines=(enumerate(inspect.getsourcelines(arg_cls)[0]))):
            if token == '\n':
                field = None  # New line
            elif token.type == tokenize.NAME:
                if not field_column:
                    field_column = token.start[1]
            elif token.exact_type == tokenize.COLON:
                if last_token.start[1] == field_column:  # type: ignore
                    field = last_token.string  # type: ignore # All fields are required to have type annotation so last_token is not None
            elif token.type == tokenize.COMMENT and field:
                # TODO nicer way to deal with with long comments or support multiple lines
                comments[field] = (token.string + ' ')[1:token.string.lower().find('# noqa:')]  # TODO consider move processing out
            last_token = token
        return comments

    def to_cmd_argv(self, arg: ArgType, separator: Optional[str] = '') -> Sequence[str]:
        """Generate the command line arguments

        :param separator: separator marker, empty str to disable, None to default to class name
        :param arg: the annotated argument class object
        :return: command line sequence"""
        argv = self._gen_cmd_argv(arg, '')
        if separator is None:
            separator = self._arg_class.__name__
        return (separator + '+', *argv, separator + '-') if separator else tuple(argv)


def custom_arg_suite(type_handlers: Sequence[Type[TypeHandler]] = (), primitive_handler_addons: Sequence[Type[PrimitiveHandlerAddon]] = ()):
    """Generate a decorator to easily convert back and forth from command-line to `NamedTuple` or `dataclass`.

    The decorator monkey patches the constructor, so that the IDE would infer the type of
    the deserialized `arg` for code auto completion and type check::

        arg: ArgClass = ArgClass.__from_argv__(my_argv)  # Factory method, need the manual type hint `:ArgClass`
        arg = ArgClass(my_argv)  # with monkey-patched constructor, no manual hint needed

    For the ArgClass fields without types but with default values: only private fields starts with "__" to overwrite the existed
    argument parameters are allowed; others will throw SmartArgError.

    For the handlers, later ones win when they can handle the same type.

    Usage::

        @arg_suite  # `arg_suite` is a shorthand for `custom_arg_suite()`
        class MyTup(NamedTuple):
            field_one: str
            _field_one = {"choices": ["one", "two"]}  # advanced usage: overwrite the `field_one` parameter
            field_two: List[int]
            ...

    :param primitive_handler_addons: the primitive types handling in addition to the provided primitive type basic operations
    :param type_handlers: the types handling in addition to the provided types handling.
    :return: the argument class decorator"""
    class Decorator:
        def __init__(self) -> None:
            addons = (PrimitiveHandlerAddon, *primitive_handler_addons)
            handler_types: Sequence[Type] = (PrimitiveHandler, CollectionHandler, DictHandler, TupleHandler, *type_handlers)
            primitives = tuple(reversed(addons))
            self.handlers = tuple(handler(primitives) for handler in reversed(handler_types))

        def __call__(self, cls):
            ArgSuite(self.handlers, cls)
            return cls
    return Decorator()


arg_suite = custom_arg_suite()  # Default argument class decorator to expose smart arg functionalities.
