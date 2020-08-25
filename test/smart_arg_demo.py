import sys
from typing import (List, NamedTuple, Tuple, Optional, Dict)

from smart_arg import arg_suite


class NestedArg(NamedTuple):
    """
    NestedArg docstring
    """
    class NNested(NamedTuple):
        n_name: str
    n_nested: NNested = NNested(n_name='nested name')


@arg_suite
class MyTup(NamedTuple):
    """
    MyTup docstring goes to description
    """
    # nested_arg: NestedArg

    nn: List[int]  # Comments go to argparse help
    __nn = {'choices': (200, 300)}
    a_tuple: Tuple[str, int]
    encoder: str  # Text encoder type
    # empty_tuple: Tuple = 0
    h_param: Dict[str, int]  # Hyperparameters
    nested: NestedArg  # nested args
    n: Optional[int] = None
    adp: bool = True  # bool is a bit tricky for now
    embedding_dim: int = 100  # Size of embedding vector
    lr: float = 1e-3  # Learning rate

    def __validate__(self):
        for f in self:
            print(f)
        assert self.n

    def __late_init__(self) -> 'MyTup':
        processed = self
        if processed.n is None:
            processed = processed._replace(n=processed.nn[0])
        return processed


expected_arg = MyTup(nn=[200, 300], encoder="fastText", a_tuple=("s", 5), h_param={"y": 1, "n": 0}, nested=NestedArg())
parsed = MyTup.__from_argv__(sys.argv[1:])
if parsed == expected_arg:
    print(f"Matched: '{parsed}'")
else:
    err_code = 168
    print(f"Error({err_code})"
          f"Expected: {expected_arg}\n"
          f"Parsed: {parsed}")
    exit(err_code)
