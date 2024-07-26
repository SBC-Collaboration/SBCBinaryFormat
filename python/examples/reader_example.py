import numpy as np
from sbcbinaryformat.files import Streamer

streamer = Streamer("test.sbc.bin")
data_dict = streamer.to_dict()
for key, value in data_dict.items():
    print(f"key: {key}\t shape: {value.shape}")
    print(value[:10])
