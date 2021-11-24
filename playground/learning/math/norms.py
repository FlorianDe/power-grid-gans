from typing import Union

import numpy.typing as npt


def lx_norm(vectors: npt.ArrayLike, x: Union[int, str] = 1) -> npt.ArrayLike:
    if isinstance(x, int):
        sums = []
        for vec in vectors:
            s = 0.0
            for elem in vec:
                s = s + elem ** x
            s ** (1 / x)
            sums.append(s)
        return sums
    elif isinstance(x, str):
        if x == 'inf':
            return [max(vector) for vector in vectors]
        else:
            raise ValueError("Wrong norm input")
    else:
        raise ValueError("Wrong norm input")


if __name__ == '__main__':
    data = [
        [1, 0],
        [0.3, 0.7],
        [0.7071, 0.7071],
        [0.9, 0.1],
    ]

    print(lx_norm(data, 'inf'))
    # print([numpy.linalg.norm(elem, ord='fro') for elem in data])
