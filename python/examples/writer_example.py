import numpy as np

from sbcbinaryformat.files import Writer

with Writer("test.sbc.bin",
            ["t", "x", "y", "z", "momentum", "source"],
            ['i4', 'd', 'd', 'd', 'd', "U100"],
            [[1], [1], [1], [1], [3, 2], [1]]) as sbc_writer_example:

    sbc_writer_example.write({'t': [1],
                              'x': [2.0],
                              'y': [3.0],
                              'z': [4.0],
                              'momentum': [[1, 2], [4, 5], [7, 8]],
                              'source': ["Bg"]})

    rng = np.random.default_rng()

    for _ in range(128):
        sbc_writer_example.write({'t': rng.integers(-10, 10, (1)),
                                  'x': rng.random((1)),
                                  'y': rng.random((1)),
                                  'z': rng.random((1)),
                                  'momentum': rng.random((3, 2)),
                                  'source': rng.choice(["Bg", "Co-60", "Th-228", "Sb-124"], (1))})
