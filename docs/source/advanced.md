## More Features
### Post processing & validation

A user can define a method in the argument class to do post processing after the argument is created.
For example, when a field's default value depends on some other input fields, one could use a default
placeholder `LateInit`, and a `__post_init__` function to define the actual value. 

The `__post_init__` function can also be used to validate the argument after instantiation:

```python
from typing import NamedTuple

from smart_arg import arg_suite, LateInit


@arg_suite
class MyArg(NamedTuple):
    network: str
    _network = {'choices': ['cnn', 'mlp']}
    num_layers: int = LateInit

    def __post_init__(self) -> 'MyArg':
        assert self.network in self._network['choices'], f'Invalid network {self.network}'
        if self.num_layers is LateInit:
            if self.network == 'cnn':
                num_layers = 3
            elif self.network == 'mlp':
                num_layers = 5
            self.num_layers=num_layers
        else: 
            assert self.num_layers >= 0, f"number_layers: {self.num_layers} can not be negative"

```
Notes:
* If any fields are assigned a default placeholder `LateInit`, a `__post_init__` is expected
to be defined, replace any `LateInit` with actual values, or it will fail the internal validation and trigger a `SmartArgError`.
* Field mutations are only allowed in the `__post_init__` as part of the class instance construction. No mutations are
allowed after construction, including manually calling `__post_init__`.
* `__post_init__` of a `NamedTuple` works similar to that of a `dataclass`.

## Supported Argument Classes
### [`NameTuple`](https://docs.python.org/3.7/library/typing.html?highlight=namedtuple#typing.NamedTuple)
* Strong immutability
* Part of the Python distribution  
* No inheritance
 
### [`dataclass`](https://docs.python.org/3.7/library/dataclasses.html)
* Weak immutability with native `@dataclass(frozen=True)` or `smart-arg` patched (when not `frozen`) 
* `pip install dataclasses` is needed for Python 3.6
* Inheritance support
* Native `__post_init__` support
## Advanced Usages

By default, `smart-arg` supports the following types as fields of an argument class:
* primitives: `int`, `float`, `bool`, `str`, `enum.Enum`
* `Tuple`: elements of the tuple are expected to be primitives
* `List`/`Set`: `List[int]`, `List[float]`, `List[bool]`, `List[str]`, `List[enum.Enum]`, `Set[int]`, `Set[float]`, `Set[bool]`, `Set[str]`, `Set[enum.Enum]`
* `Dict`: `Dict[int, int]`, `Dict[int, float]`, `Dict[int, bool]`, `Dict[int, str]`, `Dict[float, int]`, `Dict[float, float]`, 
`Dict[float, bool]`, `Dict[float, str]`, `Dict[bool, int]`, `Dict[bool, float]`, `Dict[bool, bool]`, `Dict[bool, str]`, 
`Dict[str, int]`, `Dict[str, float]`, `Dict[str, bool]`, `Dict[str, str]`, `Dict[enum.Enum, int/float/bool/str]`, `Dict[int/float/bool/str, enum.Enum]`
* `Optional[AnyOtherSupportedType]`: Beware that any optional field is required to **default to `None`**.

### override argument fields
A user can change the parsing behavior of certain field of an argument class.
One can only do this when the field's type is already supported by `smart-arg`.  


This is done by defining a private companion field starts with "``__``" (double underscores) to overwrite the keyed arguments
to [ArgumentParser.add_argument](https://docs.python.org/3/library/argparse.html#the-add-argument-method) with a dictionary.

<font color='red'>ALERT:</font> this can lead to **inconsistent behaviors** when one also generates the command-line
representation of an argument class instance, since it can only modify the deserialization 
behavior from the command-line representation.
```python
from typing import NamedTuple, List

from smart_arg import arg_suite


@arg_suite
class MyTup(NamedTuple):
    a_list: List[int]
    __a_list = {'choices': [200, 300], 'nargs': '+'}
```

### override or extend the support of primitive and other types
A user can use this provided functionality to change serialization and deserialization behavior for supported types and add support for additional types.
* User can overwrite the primitive types handling by defining additional `PrimitiveHandlerAddon`. The basic primitive handler
is defined in source code `PrimitiveHandlerAddon`. A user can pass in the customized handlers to the decorator.
* Same to type handler by providing additional `TypeHandler` and pass in the decorator argument. `TypeHandler` is to deal with complex types
other than primitive ones such as `List`, `Set`, `Dict`, `Tuple`, etc.

```python
from math import sqrt
from typing import NamedTuple, Any, Type

from smart_arg import PrimitiveHandlerAddon, TypeHandler, custom_arg_suite


# overwrite int primitive type handling by squaring it
class IntHandlerAddon(PrimitiveHandlerAddon):
    @staticmethod
    def build_type(arg_type) -> Any:
        return lambda s: int(s) ** 2

    @staticmethod
    def build_str(arg) -> str:
        return str(int(sqrt(arg)))

    @staticmethod
    def handles(t: Type) -> bool:
        return t == int

class IntTypeHandler(TypeHandler):
    def _build_common(self, kwargs, field_meta) -> None:
        super()._build_common(kwargs, field_meta)
        kwargs.help = '(int, squared)'

    def _build_other(self, kwargs, arg_type) -> None:
        kwargs.type = self.primitive_addons[0].build_type(arg_type)

    def handles(self, t: Type) -> bool:
        return t == int


@custom_arg_suite(primitive_handler_addons=[IntHandlerAddon], type_handlers=[IntTypeHandler])
class MyTuple(NamedTuple):
    a_int: int

```