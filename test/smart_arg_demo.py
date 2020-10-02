import sys
from enum import Enum, auto
from typing import List, NamedTuple, Tuple, Optional, Dict

from smart_arg import arg_suite, LateInit


class Encoder(Enum):
    FASTTEXT = auto()
    WORD2VEC = auto()


class NestedArg(NamedTuple):
    class NNested(NamedTuple):
        n_name: str
    n_nested: NNested = NNested(n_name='nested name')


@arg_suite
class MyTup(NamedTuple):
    """MyTup docstring goes to description"""
    nn: List[int]  # Comments go to argparse help
    a_tuple: Tuple[str, int]
    h_param: Dict[str, int]  # Hyperparameters
    encoder: Encoder = Encoder.FASTTEXT  # Word encoder type
    nested: Optional[NestedArg] = None  # nested args
    n: Optional[int] = LateInit
    adp: bool = True  # bool is a bit tricky for now
    embedding_dim: int = 100  # Size of embedding vector
    lr: float = 1e-3  # Learning rate

    def __post_init__(self):
        assert self.nn, "Expect nn to be non-empty."  # validation
        if self.n is LateInit:  # post processing
            self.n = self.nn[0]


expected_arg = MyTup(nn=[200, 300], a_tuple=("s", 5), h_param={"y": 1, "n": 0}, nested=NestedArg())

expected_argv = expected_arg.__to_argv__()
print(f"Serialized reference to the expected argument object:\n{expected_argv!r}")

print(f"Deserializing from {sys.argv[1:]!r}.")
deserialized = MyTup.__from_argv__()

re_deserialized = MyTup.__from_argv__(expected_argv)
if deserialized == expected_arg == re_deserialized:
    print(f"Matched: '{deserialized}'")
else:
    err_code = 168
    print(f"Error({err_code}):\n"
          f"Expected:\n {expected_arg}\n"
          f"Re-deserialized:\n {re_deserialized}\n"
          f"Deserialized:\n {deserialized}")
    exit(err_code)
