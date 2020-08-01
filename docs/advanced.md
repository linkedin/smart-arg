## More Features
### Post processing

A user can define a method in the argument class to do post processing after the argument is created.
For example, when a field's default value depends on some other input fields, one could use a default
placeholder `LateInit`, and a `__late_init__` function to define the actual value:
```python
from typing import NamedTuple

from linkedin.smart_arg import arg_suite, LateInit


@arg_suite
class MyArg(NamedTuple):
    network: str
    _network = {'choices': ['cnn', 'mlp']}
    num_layers: int = LateInit

    def __late_init__(self) -> 'MyArg':
        if self.num_layers is LateInit:
            if self.network == 'cnn':
                num_layers = 3
            elif self.network == 'mlp':
                num_layers = 5
            else:
                raise AssertionError('Not reachable')
            return self._replace(num_layers=num_layers)
        else:
            return self
```
Note that if any fields are assigned a default placeholder `LateInit`, a `__late_init__` is expected
to be defined, replace any `LateInit` with actual values and return an argument(`NamedTuple`) instance, 
or it will raise a `SmartArgError`.

### Validation

An optional `__validate__` function can be defined to validate the argument after instantiation or `__late_init__`:
```python
from typing import NamedTuple

from linkedin.smart_arg import arg_suite


@arg_suite
class MyArg(NamedTuple):
    num_layers: int
    
    def __validate__(self):
        assert self.num_layers >= 0, f"number_layers: {self.num_layers} can not be negative"
```

## Advanced Usages

By default, `smart-arg` supports the following types as fields of a `NamedClass` argument class:
* primitives: `int`, `float`, `bool`, `str`
* `Tuple`: elements of the tuple are expected to be primitives
* `List`/`Set`: `List[int]`, `List[float]`, `List[bool]`, `List[str]`, `Set[int]`, `Set[float]`, `Set[bool]`, `Set[str]`
* `Dict`: `Dict[int, int]`, `Dict[int, float]`, `Dict[int, bool]`, `Dict[int, str]`, `Dict[float, int]`, `Dict[float, float]`, 
`Dict[float, bool]`, `Dict[float, str]`, `Dict[bool, int]`, `Dict[bool, float]`, `Dict[bool, bool]`, `Dict[bool, str]`, 
`Dict[str, int]`, `Dict[str, float]`, `Dict[str, bool]`, `Dict[str, str]`
* `Optional[AnyOtherSupportedType]`: Beware that any optional field is required to **default to `None`**.

### override argument fields
A user can change the parsing behavior of certain field of an argument class.
One can only do this when the field's type is already supported by `smart-arg`.  


This is done by defining a companion field starts with "``_``" to overwrite the keyed arguments
to [ArgumentParser.add_argument](https://docs.python.org/3/library/argparse.html#the-add-argument-method) with a dictionary.

<font color='red'>ALERT:</font> this can lead to **inconsistent behaviors** when one also generates the command-line
representation of an argument class instance, since it can only modify the deserialization 
behavior from the command-line representation.
```python
from typing import NamedTuple, List

from linkedin.smart_arg import arg_suite


@arg_suite
class MyTup(NamedTuple):
    a_list: List[int]
    _a_list = {'choices': [200, 300], 'nargs': '+'}
```

### override or extend the support of primitive and other types
A user can use this provided functionality to change serialization and deserialization behavior for supported types and add support for additional types.
* User can overwrite the primitive types handling by defining additional `PrimitiveHandlerAddon`. The basic primitive handler
is defined in source code `PrimitiveHandlerAddon`. A user can pass in the customized handlers to the decorator.
* Same to type handler by providing additional `TypeHandler` and pass in the decorator argument. `TypeHandler` is to deal with complex types
other than primitive ones such as `List`, `Set`, `Dict`, `Tuple`, etc.

```python
from typing import NamedTuple

from linkedin.smart_arg import PrimitiveHandlerAddon, TypeHandler, custom_arg_suite


# overwrite int primitive type handling by squaring it
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

```