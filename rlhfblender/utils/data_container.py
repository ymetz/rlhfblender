from typing import Union

import numpy as np
from PIL import Image
from UltraDict import UltraDict


class DataContainer:
    def __init__(self, name):
        self.data = UltraDict(name=name)

    def set_data(self, result_hash: str, data: np.array, concatenate: bool = False):
        # we assume, that if the first dimensions is smaller than it the others, it is actually the number of channels,
        # and we switch axes to be able to return it as an image (happens e.g. with saved obs from frame stacked envs.)
        # print(self.data.shape, self.data[step])
        if (
            isinstance(data, np.ndarray)
            and len(data.shape) == 4
            and data.shape[1] < data.shape[2]
            and data.shape[1] < data.shape[3]
        ):
            data = np.moveaxis(data, [0, 1, 2, 3], [0, 3, 1, 2])
        if result_hash not in self.data or concatenate is False:
            self.data[result_hash] = data
        else:
            if (
                isinstance(self.data[result_hash], np.ndarray)
                and self.data[result_hash].shape[1:] == data.shape[1:]
            ):
                self.data = np.concatenate((self.data[result_hash], data), axis=0)
            elif isinstance(self.data[result_hash], list):
                self.data[result_hash] = self.data[result_hash] + data
            else:
                raise ValueError(
                    "Data shape does not match",
                    self.data[result_hash].shape,
                    data.shape,
                )

    def get_data(self, result_hash: str):
        return self.data[result_hash]

    def is_empty(self):
        return self.data is None

    def __contains__(self, result_hash: str):
        return result_hash in self.data.keys()

    def clear(self):
        self.data.clear()

    def get_single_entry(
        self,
        result_hash: str,
        step_or_id: Union[str, int],
        channels=None,
        as_original_type=False,
    ):
        if self.data is None:
            return None
        return_data = self.data[result_hash][step_or_id]
        if channels is None:
            pass
        elif len(channels) == 1 and channels[0]:
            return_data = return_data[:, :, channels[0]]
        elif len(channels) > 1:
            return_data = return_data[:, :, channels[0] : channels[1]]
        else:
            return None
        if as_original_type:
            return return_data
        elif return_data.dtype == np.uint8:
            return Image.fromarray(return_data)
        else:
            return Image.fromarray(return_data.astype(np.uint8))
