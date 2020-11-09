# Introduction to Smart Argument Suite (`smart-arg`)

Smart Argument Suite (`smart-arg`) is a slim and handy Python library that helps one work safely and conveniently 
with the arguments that are represented by an immutable argument class 
([`NamedTuple`](https://docs.python.org/3.7/library/typing.html?highlight=namedtuple#typing.NamedTuple) or 
[`dataclass`](https://docs.python.org/3.7/library/dataclasses.html#dataclasses.dataclass) out-of-box),
and passed through command-line interfaces.

`smart-arg` promotes arguments type-safety, enables IDEs' code autocompletion and type hints 
functionalities, and helps one produce correct code.

![](smart-arg-demo.gif)

## Quick start

The [`smart-arg`](https://pypi.org/project/smart-arg/) package is available through `pip`.
```shell
pip3 install smart-arg
```

Users can define their argument -- a `NamedTuple` or `dataclass` class with `smart-arg` decorator `@arg_suite` in their Python scripts 
and pass it through the command-ine interface.

```python
import sys
from typing import NamedTuple, List, Tuple, Dict, Optional
from smart_arg import arg_suite


# Define the argument 
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


def cli_interfaced_job_scheduler():
    # Create the argument instance
    my_arg = MyArg(nn=[3], a_tuple=("str", 1), encoder='lstm', h_param={}, adp=False)  # The patched argument class requires keyword arguments to instantiate the class

    # Serialize the argument to command-line representation
    argv = my_arg.__to_argv__()
    cli = 'my_job.py ' + ' '.join(argv)
    # Schedule the job with command line `cli`
    print(f"Executing job:\n{cli}")
    # Executing job:
    # my_job.py --nn 3 --a_tuple str 1 --encoder lstm --h_param --batch_size None --adp False --embedding_dim 100 --lr 0.001


# my_job.py
# Deserialize the command-line representation of the argument back to an instance 
my_arg: MyArg = MyArg.__from_argv__(sys.argv[1:])  # Equivalent to `MyArg(None)`, one positional arg required to indicate the arg is a command-line representation.
print(my_arg)
# MyArg(nn=[3], a_tuple=('str', 1), encoder='lstm', h_param={}, batch_size=None, adp=False, embedding_dim=100, lr=0.001)

# `my_arg` can be used in later script with a typed manner, which help of IDEs (type hints and auto completion)
# ...
print(f"My network has {len(my_arg.nn)} layers with sizes of {my_arg.nn}.")
# My network has 1 layers with sizes of [3].

```

```shell-session
> python my_job.py -h
usage: my_job.py [-h] --nn [int [int ...]] --a_tuple str int --encoder str
                 --h_param [str:int [str:int ...]] [--batch_size int]
                 [--adp {True,False}] [--embedding_dim int] [--lr float]

MyArg is smart! (docstring goes to description)

optional arguments:
  -h, --help            show this help message and exit
  --nn [int [int ...]]  (List[int], required) Comments go to argparse help
  --a_tuple str int     (Tuple[str, int], required) a random tuple argument
  --encoder str         (str, required) Text encoder type
  --h_param [str:int [str:int ...]]
                        (Dict[str, int], required) Hyperparameters
  --batch_size int      (Optional[int], default: None)
  --adp {True,False}    (bool, default: True) bool is a bit tricky
  --embedding_dim int   (int, default: 100) Size of embedding vector
  --lr float            (float, default: 0.001) Learning rate

```
## Promoted practices
* Focus on defining the arguments diligently, and let the `smart-arg` 
  (backed by [argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argumentparser-objects)) 
  work its magic around command-line interface. 
* Always work directly with argument class instances when possible, even if you only need to generate the command-line representation.
* Stick to the default behavior and the basic features, think twice before using any of the [advanced features](TODO-linked-to-readthedocs).


## More detail
For more features and implementation detail, please refer to the [documentation](TODO-linked-to-readthedocs).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE.md](LICENSE.md) file for details
