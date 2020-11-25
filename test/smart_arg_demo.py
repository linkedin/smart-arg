import sys
from enum import Enum, auto
from typing import List, NamedTuple, Tuple, Optional, Dict, Sequence

from smart_arg import arg_suite, LateInit, frozenlist


class Encoder(Enum):
    FASTTEXT = auto()
    WORD2VEC = auto()


class NestedArg(NamedTuple):
    class NNested(NamedTuple):
        n_name: str
    n_nested: NNested = NNested(n_name='nested name')


@arg_suite
class MyModelConfig(NamedTuple):
    """MyModelConfig docstring goes to description"""
    adp: bool
    nn: List[int]  # Comments go to argparse help
    a_tuple: Tuple[str, int]
    h_param: Dict[str, int] = {}  # Hyperparameters
    encoder: Encoder = Encoder.FASTTEXT  # Word encoder type
    nested: Optional[NestedArg] = None  # nested args
    n: Optional[int] = LateInit  # An argument that can be auto-set in post processing
    immutable_list: Sequence[int] = frozenlist((1, 2))  # A frozenlist is an alias to tuple
    embedding_dim: int = 100  # Size of embedding vector
    lr: float = 1e-3  # Learning rate

    def __post_init__(self):
        assert self.nn, "Expect nn to be non-empty."  # validation
        if self.n is LateInit:  # post processing
            self.n = self.nn[0]


if __name__ == '__main__':
    my_config = MyModelConfig(nn=[200, 300], a_tuple=("s", 5), adp=True, h_param={'n': 0, 'y': 1}, nested=NestedArg())

    my_config_argv = my_config.__to_argv__()
    print(f"Serialized reference to the expected argument container object:\n{my_config_argv!r}")

    print(f"Deserializing from the command line string list: {sys.argv[1:]!r}.")
    deserialized = MyModelConfig.__from_argv__()

    re_deserialized = MyModelConfig.__from_argv__(my_config_argv)
    if deserialized == my_config == re_deserialized:
        print(f"All matched, argument container object: '{deserialized!r}'")
    else:
        err_code = 168
        print(f"Error({err_code}):\n"
              f"Expected argument container object:\n {my_config!r}\n"
              f"Re-deserialized:\n {re_deserialized!r}\n"
              f"Deserialized:\n {deserialized!r}")
        exit(err_code)
