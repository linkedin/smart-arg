import sys
from typing import List, NamedTuple, Tuple, Dict
from linkedin.smart_arg import arg_suite


@arg_suite
class MockArgTup(NamedTuple):
    a_bool: bool
    b_list_int: List[int]
    d_tuple_2: Tuple[str, int]
    e_dict_str_int: Dict[str, int]


def main(args: List[str] = None):
    parsed_tup = MockArgTup(args)
    return parsed_tup


if __name__ == '__main__':
    main(sys.argv)
