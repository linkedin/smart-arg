# Smart Arguments Suite `(smart-arg`)

Smart Arguments Suite (`smart-arg`) is a slim and handy Python library that help one work safely and conveniently with
command line arguments. It will generate a 
[argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argumentparser-objects) based on 
the associated [`NameTuple`](https://docs.python.org/3.7/library/typing.html?highlight=namedtuple#typing.NamedTuple) 
class fields with additional help information and can convert back and forth between
immutable `NamedTuple` objects and user command-line interface. 

`smart-arg` promotes type-safety with command-line arguments, enables IDEs' code autocompletion and type hints 
functionalities, and helps one produce correct code around command line arguments.

## How to use

### install
```shell script
pip3 install smart-arg
```
### develop
```shell script
# Uncomment the next two lines to set up and activate a virtual environment as needed
# python3 -m venv .venv
# . .venv/bin/activate

python3 setup.py develop
# pip install pytest-flake8  # as needed
pytest --flake8
```
### quick start
Users can define the `NamedTuple` class with smart-arg decorator `@arg_suite` in the python script expected to digest the command-line arguments.
```python
import sys
from typing import NamedTuple, List, Tuple, Set, Dict, Optional
from linkedin.smart_arg import arg_suite

@arg_suite
class MyArg(NamedTuple):
    """
    MyArg is smart! (docstring goes to description)
    """
    nn: List[int]  # Comments go to argparse help
    a_tuple: Tuple[str, int]  # a random tuple argument
    encoder: str  # Text encoder type
    h_param: Dict[str, int]  # Hyperparameters
    batch_size: Optional[int] = None
    adp: bool = True  # bool is a bit tricky
    embedding_dim: int = 100  # Size of embedding vector
    lr: float = 1e-3  # Learning rate


def main(argv):
    parsed_arg = MyArg(argv)
    # parsed_arg can be used in later script
    # ...
    print(f"My network has {len(parsed_arg.nn)} layers with size of {parsed_arg.nn}.")


if __name__ == '__main__':
    main(sys.argv[1:])

```
### supported types defined in `NamedTuple` class by default
* primitives: `int`, `float`, `bool`, `str`
* `Optional`: `Optional[int]`, `Optional[float]`, `Optional[bool]`, `Optional[str]`
* `Tuple`: elements of the tuple are expected to be primitives
* `List`/`Set`: `List[int]`, `List[float]`, `List[bool]`, `List[str]`, `Set[int]`, `Set[float]`, `Set[bool]`, `Set[str]`
* `Dict`: `Dict[int, int]`, `Dict[int, float]`, `Dict[int, bool]`, `Dict[int, str]`, `Dict[float, int]`, `Dict[float, float]`, 
`Dict[float, bool]`, `Dict[float, str]`, `Dict[bool, int]`, `Dict[bool, float]`, `Dict[bool, bool]`, `Dict[bool, str]`, 
`Dict[str, int]`, `Dict[str, float]`, `Dict[str, bool]`, `Dict[str, str]`

### post_process

A user can define a method in the argument class to do post processing after the parsing is done.
For example, when a field's default value depends on some other input fields, one could use a default
placeholder `LateInit`, and a `__post_process__` function to define the actual value:
```python
@arg_suite
class MyArg(NamedType):
    network: str
    _network = {'choices': ['cnn', 'mlp']}
    num_layers: int = LateInit
    
    def __post_process__(self) -> 'MyArg':
        if self.num_layers is LateInit:
            if self.network == 'cnn':
                num_layers = 3
            elif self.network == 'mlp':
                num_layers = 5
            else:
                raise AssertionError('Not reachable')
        return self._replace(num_layers = num_layers)
```
Note that if any fields are assigned a default placeholder `LateInit`, a `__post_process__` is expected
to be defined, replace any `LateInit` with actual values and return an argument(NamedTuple) instance, 
or it will raise a `SmartArgError`.

### validate

An optional `__validate__` function can be defined to validated the parsed argument:
```python
@arg_suite
class MyArg(NamedType):
    num_layers: int
    
    def __validate__(self):
        assert self.num_layers >= 0, f"number_layers: {self.num_layers} can not be negative"
```
## promoted practices:
* Focus on defining the arguments diligently, and let the `smart-arg` to handle the serialization/deserialization from/to command line consistently.  
* Stick to the default parsing behavior, try to avoid using [_kwArg](#overwrite-argument-fields) to overwrite `type`
* If possible, always work directly with argument `NamedTuple` object, even if you only need the command line counterpart.

## Advanced usage
### overwrite argument fields
User can define a companion field starts with "_" to overwrite the keyed arguments
to [ArgumentParser.add_argument](https://docs.python.org/3/library/argparse.html#the-add-argument-method) with a dictionary.

This enable users to overwrite **individual field** parsing behavior with types that are already supported by the default 
or the additional handlers.
```python
@arg_suite
class MyTup(NamedTuple):
    a_list: List[int]
    _a_list = {'choices': [200, 300], 'nargs': '+'}
```

### overwrite primitive handling and type handling
User can use this provided functionality to **batch overwrite** parsing behavior for types and also add support for other types.
* User can overwrite the primitive types handling by defining additional `PrimitiveHandlerAddon`. The basic primitive handler
is defined in source code `PrimitiveHandlerAddon`. User can pass-in the customized handler in the decorator.
* Same to type handler by providing additional `TypeHandler` and pass-in the decorator argument. `TypeHandler` is to deal with boxed types
other than primitive ones such as List, Set, Dict, Tuple, etc.

```python
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

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE](LICENSE) file for details