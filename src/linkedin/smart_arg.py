"""Command-Line Smart Arguments Parsing Suite

This module is an command-line argument library that:

    - generates argparse.ArgumentParser based a NamedTuple classed argument
    - handles two-way conversions: a typed argument object (A `NamedTuple`) <-> a command-line argv
    - enables IDE type hints and code auto-completion by using `NamedTuple`
    - promotes type-safety of command-line arguments


The following is a simple usage example::

    # Define the argument class:
    @arg_suite
    class MyArg(NamedTuple):
        '''MyArg docstring goes to description
        Each named (without "_" prefix) and fully typed attribute will be turned into an ArgumentParser.add_argument
        '''

        nn: List[int]  # Comments go to ArgumentParser argument help
        _nn = {'choices': (200, 300)}  # (Advanced feature) Optional user supplement/override for `ArgumentParser.add_argument("--nn", **(kwargs|_nn))`

        a_tuple: Tuple[str, int]  # Arguments without defaults are treated as "required = True"
        h_param: Dict[str, int]  # Also supports List, Set

        ###### arguments without defaults go first as required by NamedTuple ######

        l_r: float = 0.01 # Arguments with defaults are treated as "required = False"
        n: Optional[int] = None


    # Create the corresponding argument class instance from the command line argument list:
    # by using
    parsed_arg: ArgClass = ArgClass.from_argv(sys.argv[1:])  # with factory method, need the manual type hint `:ArgClass` to help IDE
    # or the monkey-patched constructor of the annotated NameTuple argument class
    parsed_arg = ArgClass(sys.argv[1:])  # with monkey-patched constructor with one positional argument, no manual hint needed

    # Create a NamedTuple argument instance and generate its command line counterpart:
    # the monkey-patched constructor only take named arguments for directly creating the NamedTuple
    arg = ArgClass(nn=[200, 300], a_tuple=('t', 0), h_param={'batch_size': 1000})
    # generate the corresponding command line argument list
    arg.to_argv()


The module contains the following public classes:

- arg_suite -- The main entry point for command-line argument suite. As the
    example above shows, this decorator will attach an ArgSuite instance to
    the argument NamedTuple "subclass".

- ArgSuite -- The main class that generates the corresponding ArgumentParser
    and handles the two-way conversions.

- PrimitiveHandlerAddon -- The base class to handle primitive types, and users can
    implement their owns to change the behavior.

- TypeHandler -- The base class to handle non-primitive types, users can extend or
    change existing behaviors.

All other classes and methods in this module are considered implementation details.
"""

__all__ = (
    'arg_suite',
    'custom_arg_suite',
    'ArgSuite', 'LateInit',
    'SmartArgError',
    'TypeHandler',
    'PrimitiveHandlerAddon')

import inspect
import logging
import os
import re
import sys
import tokenize
import warnings
from argparse import Action, ArgumentParser
from types import SimpleNamespace as KwargsType  # type for kwargs for `ArgumentParser.add_argument`, currently just an alias
from typing import (Any, Collection, Dict, Generic, Iterable, List, NamedTuple,
                    Optional, Sequence, Set, Tuple, Type, TypeVar, Union)


class DefaultMarker:
    """special singleton/class to mark default."""
    pass


class LateInit:
    """special singleton/class to mark post-parse-init'ed fields"""
    pass


ArgType = TypeVar('ArgType', bound=NamedTuple)  # NamedTuple is not a real class bound, but setting `bound` to NamedTuple makes mypy happier
FieldMeta = NamedTuple('FieldMeta', [('comment', str), ('default', Any), ('type', Type)])

PRIMITIVES = {str, int, float, bool}
RESTORE_OPTIONAL = re.compile(f'Union\\[({"|".join((p.__name__ for p in PRIMITIVES))}), NoneType\\]')

logger = logging.getLogger(__name__)
LEVEL_KEY = 'SMART_ARG_LOG_LEVEL'
if LEVEL_KEY in os.environ:
    logger.addHandler(logging.StreamHandler())
    log_level = os.environ[LEVEL_KEY].upper()
    logger.setLevel(log_level)
    logger.info(f"Detected environment var `{LEVEL_KEY}, set log level to '{log_level}' and log to stream.")

if sys.version_info >= (3, 8):
    logger.info(f"Python {sys.version_info} >= 3.8. from typing import get_origin, get_args ")
    from typing import get_args, get_origin
elif sys.version_info >= (3, 7):
    logger.info(f"Python {sys.version_info} == 3.7.x. Defining get_origin, get_args ")
    get_origin, get_args = lambda tp: vars(tp).get('__origin__'), lambda tp: vars(tp).get('__args__') if vars(tp).get('__args__') else ()  # noqa: E731
else:
    try:
        logger.warning(f"Unsupported python version {sys.version_info} < 3.7. Try 'from typing_inspect import get_origin, get_args' now.")
        from typing_inspect import get_args, get_origin
    except ImportError as e:
        warnings.warn(f"`from typing_inspect import get_origin, get_args` failed for "
                      f"unsupported python version {sys.version_info} < 3.7. It might work with 'pip install typing_inspect'.")
        raise e


class SmartArgError(Exception):  # TODO Extend to better represent different types of errors.
    pass


class PrimitiveHandlerAddon:
    """
    Primitive handler addon to do some special handling on the primitive types.
    Users want to modify the primitive handling can inherit this class.
    """
    @staticmethod
    def build_primitive(arg_type: Type, kwargs: KwargsType) -> None:
        kwargs.type = arg_type
        if arg_type is bool:
            kwargs.type = PrimitiveHandlerAddon.build_type(arg_type)
            kwargs.choices = [True, False]

    @staticmethod
    def build_type(arg_type: Type):
        return (lambda s: True if s == 'True' else False if s == 'False' else s) if arg_type is bool else arg_type
    handled_types: Collection[Type] = PRIMITIVES


class TypeHandler:
    """
    Base Non-Primitive type handler
    """
    def __init__(self, primitive_addons: Dict[Type, Type[PrimitiveHandlerAddon]]):
        self.primitive_addons = primitive_addons

    def _build_common(self, kwargs: KwargsType, field_meta: FieldMeta) -> None:
        """
        Build "help", "default" and "required" for keyword arguments for
        ArgumentParser.add_argument

        :param kwargs: the keyword argument KwargsType object
        :param field_meta: the meta information extracted from the NamedTuple class
        """
        # Build help message
        arg_type = field_meta.type
        help_builder = ['(', self.type_to_str(arg_type)]
        # Get default if specified and set required if no default
        if field_meta.default is DefaultMarker:
            # TODO handle positional if we decided to support it later
            kwargs.required = True
            help_builder.append(', required')
        else:
            # Only add default to the help message for informational purpose. The default is set when creating the NamedTuple.
            # kwargs.default = field_meta.default  # left for reference purpose
            help_builder.append(', default: ')
            help_builder.append(str(field_meta.default))

        help_builder.append(')')
        # Add from source code comment
        help_builder.append(' ')
        help_builder.append(field_meta.comment)
        kwargs.help = ''.join(help_builder)

    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        """
        Build other information for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (primitive types)
        """
        # primitives metavar is type itself
        kwargs.metavar = self.type_to_str(arg_type)
        kwargs.type = arg_type

    def gen_kwargs(self, field_meta: FieldMeta) -> KwargsType:
        """
        Build keyword argument object KwargsType

        :param field_meta: argument metadata information
        :return: keyword argument object
        """
        kwargs = KwargsType()
        self._build_common(kwargs, field_meta)
        self._build_other(kwargs, field_meta.type)
        addon = self.primitive_addons.get(kwargs.type)
        addon and addon.build_primitive(kwargs.type, kwargs)
        return kwargs

    def gen_cmdln_arg(self, action: Action, arg: Any) -> Iterable[str]:
        """
        Generate command line for argument

        :param action: action object stored for the argument
        :param arg: value of the argument
        :return: iterable command line str
        """
        yield action.option_strings[0]
        if not isinstance(arg, str) and isinstance(arg, Iterable):
            yield from (str(a) for a in arg)
        else:
            yield str(arg)

    @staticmethod
    def type_to_str(t: Union[type, Type]) -> str:
        """
        Convert type to string
        Note: Optional field shows as Union when getting type, i.e. Optional[int] -> Union[int, NoneType].
        So the method is expected to return the restored type which is Optional[int].

        :param t: type of the argument, i.e. float, typing.Dict[str, int], typing.Set[int], typing.List[str] etc.
        :return: string representation of the argument type
        """
        return t.__name__ if type(t) == type else RESTORE_OPTIONAL.sub('Optional[\\1]', str(t).replace('typing.', ''))
    handled_types: Collection[Type] = PRIMITIVES
    handled_original_types: Collection[Type] = ()


class TupleHandler(TypeHandler):
    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        """
        Build other information for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (Tuple type)
        """
        super()._build_other(kwargs, arg_type)
        # get the tuple element types
        types = get_args(arg_type)
        if not types:
            raise SmartArgError(f'Invalid Tuple type: {arg_type}')
        kwargs.nargs = len(types)
        kwargs.metavar = tuple(self.type_to_str(t) for t in types)
        p_addons = self.primitive_addons

        class BuildType:
            def __init__(self):
                self.counter = 0

            def __call__(self, s):
                self.counter == len(types) and self.__init__()
                t = types[self.counter]
                self.counter += 1
                return p_addons[t].build_type(t)(s)
        kwargs.type = BuildType()
    handled_types = ()
    # tuple needs special handling because the variable length and types of the tuple element
    handled_original_types = (tuple,)


class OptionalHandler(TypeHandler):
    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        """
        Build other information for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (Optional type)
        """
        super()._build_other(kwargs, arg_type)
        self.arg_types: Tuple[Type, ...] = get_args(arg_type)  # type: ignore
        # get element type
        self.unboxed_type = self.arg_types[0]
        kwargs.metavar = self.type_to_str(self.unboxed_type)
        kwargs.type = lambda s: None if s == 'None' else self.unboxed_type(s)
    handled_types = {Optional[p] for p in PRIMITIVES}  # type: ignore


class CollectionHandler(OptionalHandler):
    def _build_other(self, kwargs: KwargsType, arg_type: Type) -> None:
        """
        Build other information for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (List/Set type)
        """
        super()._build_other(kwargs, arg_type)
        kwargs.nargs = '*'  # Consider using comment to specify the number of size of the collection
        kwargs.type = self.unboxed_type
    handled_types = {Box[p] for Box in [List, Set] for p in PRIMITIVES}  # type: ignore


class DictHandler(CollectionHandler):
    def _build_other(self, kwargs, arg_type) -> None:
        """
        Build other information for the keyword argument KwargsType object

        :param kwargs: the keyword argument KwargsType object
        :param arg_type: the type of the argument extracted from NamedTuple (Dict type)
        """
        super()._build_other(kwargs, arg_type)
        arg_types = self.arg_types

        def type_fun(s: str, types=arg_types):
            k, v = s.split(":")
            type_builder = lambda t: self.primitive_addons[t].build_type(t)  # noqa: E731
            return type_builder(types[0])(k), type_builder(types[1])(v)

        kwargs.type = type_fun
        kwargs.metavar = f'{self.type_to_str(arg_types[0])}:{self.type_to_str(arg_types[1])}'

    def gen_cmdln_arg(self, action: Action, arg):
        yield action.option_strings[0]
        yield from (f'{k}:{v}' for k, v in arg.items())
    handled_types = {Dict[k, v] for k in PRIMITIVES for v in PRIMITIVES}  # type: ignore


def custom_arg_suite(separator: str = None, type_handlers: List[Type[TypeHandler]] = None, primitive_handler_addons: List[Type[PrimitiveHandlerAddon]] = None):
    """
    Generate a decorator on NamedTuple class to easily convert back and forth from commandline to NamedTuple class.

    The decorator monkey patches the constructor, so that the IDE would infer the type of
    parsed_arg for code auto completion and type check::

        parsed_arg: ArgClass = ArgClass.from_argv(my_argv)  # Factory method, need the manual type hint `:ArgClass`
        parsed_arg = ArgClass(my_argv)  # with monkey-patched constructor, no manual hint needed

    For NamedTuple fields without types but with default values: only fields starts with "_" to overwrite the existed
    argument parameters are allowed; others will throw SmartArgError.

    Usage::

        @arg_suite  # `arg_suite` is a shorthand for `custom_arg_suite()`
        class MyTup(NamedTuple):
            field_one: str
            _field_one = {"choices": ["one", "two"]}  # advanced usage: overwrite the `field_one` parameter
            field_two: List[int]
            ...

    :param separator: a optional marker to mark the sub-sequence of `argv` to be parsed by the parser.
    :type separator: Optional[str], default to `None`, indicating using the default separator for the argument class
    :param primitive_handler_addons: the primitive types handling in addition to the provided primitive types handling
    :param type_handlers: the non-primitive types handling in addition to the provided types handling.
    :return: the argument class (NamedTuple) decorator
    """
    def arg_decorator(cls: Type[ArgType]) -> Type[ArgType]:
        def strip_argv(separator: str, argv: Optional[Sequence[str]]) -> Sequence[str]:
            """Strip any elements outside of a pair of `separator` out of `argv`."""
            if argv is None:
                argv = sys.argv[1:]
            l_s = separator + '+'
            r_s = separator + '-'
            lc = argv.count(l_s)
            rc = argv.count(r_s)
            if lc == rc:
                if lc == 0:
                    return argv
                elif lc == 1:
                    b = argv.index(l_s)
                    e = argv.index(r_s)
                    if e > b:
                        return argv[b + 1: e]
            raise SmartArgError(f"Expecting up to 1 pair of separator '{l_s}' and '{r_s}' in {argv}")

        addons = [PrimitiveHandlerAddon]
        if primitive_handler_addons:
            addons += primitive_handler_addons
        handler_types = [TypeHandler, OptionalHandler, CollectionHandler, DictHandler, TupleHandler]
        if type_handlers:
            handler_types += type_handlers

        primitives = {t: addon for addon in addons for t in addon.handled_types}
        handler_list = [handler(primitives) for handler in handler_types]

        # special handling for tuple
        fallback_handlers = {t: handler for handler in handler_list for t in handler.handled_original_types}
        handlers = {t: handler for handler in handler_list for t in handler.handled_types}
        suite = ArgSuite(cls, handlers, fallback_handlers)
        __new = cls.__new__

        def __new__wrapper(arg_class, *args, **kwargs):
            """
            Monkey-Patched NamedTuple constructor __new__.
            If any positional arguments exist, it would assume that the user is trying to parse a sequence of strings. It would also assume there is only one
            positional argument, and raise an SmartArgError otherwise.
            If no positional arguments exist, it would call the original NamedTuple constructor with all keyword arguments.

            :param arg_class:
            :type arg_class:
            :param argv: Optional positional argument, a sequence of strings to be parsed to the arg_class type
            :type argv: Sequence[str]
            :param kwargs: Optional keyword arguments, to be passed to the original NamedTuple constructor.
            """
            if args:
                # TODO Exception handling with helpful error message
                if len(args) != 1 or not isinstance(args[0], Sequence) or any(a.__class__ is not str for a in args[0]):
                    raise SmartArgError(f"Calling '{arg_class}({args}, {kwargs})' is not allowed:\n"
                                        f"Only accept one positional argument `Sequence[str]` to create the NamedTuple '{arg_class}' by parsing, got '{args}'\n"
                                        f"Use keyword arguments only to create NamedTuple object directly, got '{kwargs}'.")
                return suite.parse_to_arg(strip_argv(separator if separator else cls.__name__, args[0]))
            else:
                try:
                    return __new(arg_class, **kwargs)
                except TypeError as err:
                    # Change the message for missing arguments
                    err.args = (err.args[0].replace('positional argument', 'keyword argument'),)
                    logger.error("Creating NamedTuple failed. Might be missing required keyword arguments")
                    raise err
        # monkey patch the class
        setattr(cls, '__new__', __new__wrapper)
        setattr(cls, 'to_argv', lambda self: to_cmd_argv(self))
        setattr(cls, 'from_argv', lambda argv=None: suite.parse_to_arg(argv))
        setattr(cls, 'arg_suite', suite)
        return cls
    return arg_decorator


arg_suite = custom_arg_suite()
"""
Default argument class decorator to expose smart arg functionalities.
"""


class ArgSuite(Generic[ArgType]):
    @staticmethod
    def is_arg_type(t: Type[ArgType]):
        """
        A static method to infer whether the input class is a NamedTuple at best effort

        :param t: input class
        :return: True if the input class is a NamedTuple class otherwise False
        """
        b, f, f_t = getattr(t, '__bases__', []), getattr(t, '_fields', []), getattr(t, '_field_types', {})
        return (len(b) == 1 and b[0] == tuple and isinstance(f, tuple) and isinstance(f_t, dict)
                and all(type(n) == str for n in f) and all(type(n) == str for n, _ in f_t.items()))

    def __init__(self, arg_class: Type[ArgType], handlers: Dict[Type, TypeHandler], fallback_handlers: Dict[type, TypeHandler]) -> None:
        """
        constructor of the ArgSuite

        :param arg_class: the name of the decorated class (NamedTuple)
        :param handlers: type handling dictionary with key is the handling type and value of the type handler.
        :param fallback_handlers: the fallback type handling dictionary
        """
        assert self.is_arg_type(arg_class)
        self._arg_class = arg_class
        self._validate_fields()
        self._parser = ArgumentParser(description=self._arg_class.__doc__, argument_default=DefaultMarker, fromfile_prefix_chars='@')  # type: ignore
        setattr(self._parser, 'convert_arg_line_to_args', lambda arg_line: arg_line.split())
        self.handlers = handlers
        self.fallback_handlers = fallback_handlers
        self.handler_actions: Dict[str, Union[Tuple[TypeHandler, Action], Tuple[None, Any]]] = {}
        self._arg_classes: List[ArgType] = []
        self._gen_arguments_from_class(self._arg_class, '', True)

    def _validate_fields(self) -> None:
        """
        validate fields in the decorated class.

        :raise: SmartArgError if the decorated NamedTuple has non-typed field with defaults and such field
                does not startswith "_" to overwrite the existing argument field property.
        """
        # empty namedtuple to extract all the fields.
        class EmptyTup(NamedTuple):
            pass

        # note: NamedTuple fields without type won't be regarded as a property/entry _fields.
        arg_fields = self._arg_class._fields
        invalidate_fields = list(filter(lambda s: s.endswith('_'), arg_fields))
        if invalidate_fields:
            raise SmartArgError(f"Do not support arguments ending with '_': {invalidate_fields}.")
        arg_all_fields = vars(self._arg_class).keys() - vars(EmptyTup).keys()
        # skip callable methods and typed fields
        arg_non_callable_fields = [f for f in arg_all_fields if not callable(getattr(self._arg_class, f)) and f not in arg_fields]
        for f in arg_non_callable_fields:
            if f.startswith('_'):
                if f[1:] not in arg_fields:
                    raise SmartArgError(f"Parameter override '{f}'`s company field '{f[1:]}' is missing.")
                else:
                    logger.info(f"Found '{f[1:]}' companion field '{f}'")
            else:
                raise SmartArgError(f"Field '{f}' is not typed.")

    def _gen_arguments_from_class(self, arg_class, prefix: str, parent_required) -> None:
        """
        Add argument to the self._parser for each field in the self._arg_class
        :raise: SmartArgError if cannot find corresponding handler for the argument type
        """
        if arg_class in self._arg_classes:
            raise SmartArgError(f"Self nested argument class '{arg_class}' is not supported.")
        else:
            self._arg_classes.append(arg_class)
            comments = self.get_comments(arg_class)
            for raw_arg_name, arg_type in arg_class._field_types.items():
                arg_name = f'{prefix}{raw_arg_name}'
                default = arg_class._field_defaults.get(raw_arg_name, DefaultMarker)  # TODO move to meta
                if self.is_arg_type(arg_type):
                    required = parent_required and default is DefaultMarker
                    self._gen_arguments_from_class(arg_type, f'{arg_name}.', required)

                    class ShouldNotSpecify:
                        def __init__(self, name):
                            self.name = name

                        def __call__(self, _):
                            raise SmartArgError(f"Nested argument '{self.name}' can not be directly specified")
                    kwargs = KwargsType(required=False,
                                        metavar='Help message only. Do NOT attempt to specify, or an exception will be raised.',
                                        type=ShouldNotSpecify(arg_name),
                                        help=f"""This is a placeholder for the nested argument '{arg_name}'.
                                             Its parent is {'' if parent_required else 'not'} required.
                                             {"It's required" if default is DefaultMarker else f"Not required with default: {default}"},
                                             if the parent is being parsed.""")
                    # if default is not NO_DEFAULT:
                    #     kwargs['default'] = default
                    self._parser.add_argument(f'--{arg_name}', **vars(kwargs))
                    self.handler_actions[arg_name] = None, default
                else:
                    # fallback to special handling for the types which can not be fully enumerated, e.g. tuple
                    handler = self.handlers.get(arg_type) or self.fallback_handlers.get(get_origin(arg_type))  # type: ignore
                    if not handler:
                        raise SmartArgError(f"Unsupported type: {arg_type} for '{arg_name}'")
                    field_meta = FieldMeta(comment=comments.get(raw_arg_name, ''), default=default, type=arg_type)
                    kwargs = handler.gen_kwargs(field_meta)
                    # apply user overwrite to the argument object
                    kwargs.__dict__.update(**vars(arg_class).get(f'_{raw_arg_name}', {}))
                    if hasattr(kwargs, 'choices'):
                        logger.info(f"'{arg_name}': 'choices' {kwargs.choices} specified, removing 'metavar'.")
                        del kwargs.metavar  # 'metavar' would override 'choices' which makes the help message less helpful
                    if not parent_required and hasattr(kwargs, 'required'):
                        del kwargs.required
                    kwargs.default = DefaultMarker  # Marker for fields absent from parsing
                    self.handler_actions[arg_name] = handler, self._parser.add_argument(f'--{arg_name}', **vars(kwargs))

    def parse_to_arg(self, argv: Sequence[str], *, error_on_unknown: bool = True) -> ArgType:
        """
        Parse the command line to decorated ArgType

        :param error_on_unknown:
        :param argv: the command line list
        :return: parsed decorated object from command line
        """
        ns, unknown = self._parser.parse_known_args(argv)
        error_on_unknown and unknown and self._parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        logger.info(f"{argv} is parsed to {ns}")
        # arg_dict = vars(ns)  # {name: value for name, value in vars(ns).items() if value is not NO_DEFAULT}
        arg_dict = {name: value for name, value in vars(ns).items() if value is not DefaultMarker}

        def to_arg(arg_class: ArgType, prefix) -> ArgType:
            nest_arg = {}
            for raw_arg_name, arg_type in arg_class._field_types.items():
                arg_name = f'{prefix}{raw_arg_name}'
                arg_type_flag, default = self.handler_actions[arg_name]
                if arg_type_flag:
                    value = arg_dict.get(arg_name, DefaultMarker)
                    # ns reading variable length arguments are all lists, need to apply the origin type
                    # for the conversion to correct type.
                    value = get_origin(arg_class._field_types[raw_arg_name])(value) if isinstance(value, List) else value  # type: ignore
                else:
                    is_nested = any((name for name in arg_dict.keys() if name.startswith(arg_name)))
                    value = to_arg(arg_type, f'{arg_name}.') if is_nested else default  # type: ignore
                if value is not DefaultMarker:
                    nest_arg[raw_arg_name] = value
            return arg_class(**nest_arg)  # type: ignore
        parse_arg = to_arg(self._arg_class, '')  # type: ignore

        def call_if_defined(obj, fun: str):
            try:
                return getattr(obj, fun)()
            except AttributeError as err:
                if err.args != (f"'{obj.__class__.__name__}' object has no attribute '{fun}'",):
                    raise err
                return obj
        parse_arg = call_if_defined(parse_arg, '__post_process__')
        if parse_arg.__class__ is not self._arg_class:
            raise SmartArgError(f"Expect post_process to return a '{self._arg_class}', but got '{parse_arg.__class__}'.")
        for f in self._arg_class._fields:
            if getattr(parse_arg, f) is LateInit:
                raise SmartArgError(f"Field '{f} is still not initialized after post processing")
        call_if_defined(parse_arg, '__validate__')
        return parse_arg

    def _gen_cmd_argv(self, args, prefix) -> Iterable[str]:
        for name, arg in args.items():
            handler, action = self.handler_actions[f'{prefix}{name}']
            yield from handler.gen_cmdln_arg(action, arg) if handler else self._gen_cmd_argv(arg._asdict(), f'{prefix}{name}.')

    @staticmethod
    def get_comments(arg_cls: Type[ArgType]) -> Dict[str, str]:
        """
        Get in-line comments for the input class fields. Only single line of trailing comment is supported.

        :param arg_cls: the input class
        :return: a dictionary with key of class field name and value of in-line comment
        """
        class Source:
            def __call__(self, lines=enumerate(inspect.getsourcelines(arg_cls)[0])):
                self.index, line = next(lines)
                return line

        def line_tokenizer():
            source = Source()
            source()  # skip the `class` line and set source.index
            last_index = source.index
            for t in tokenize.generate_tokens(source):
                if last_index != source.index:
                    yield '\n'  # New line indicator
                yield t
                last_index = source.index
        comments = {}
        field_column = field = last_token = None
        for token in line_tokenizer():
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
                comments[field] = (token.string+' ')[1:token.string.lower().find('# noqa:')]  # TODO consider move processing out
            last_token = token
        return comments


def to_cmd_argv(args: ArgType) -> Sequence[str]:
    """
    Generate the command line arguments

    :param args: the annotated argument class object
    :return: command line sequence
    """
    return list(getattr(args, 'arg_suite')._gen_cmd_argv(args._asdict(), ''))
