import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data_formatters.base import InputTypes, DataTypes
from data_formatters.utils import get_single_col_by_input_type



class TFTDataset(Dataset):
    def __init__(self, data, column_definition, params) -> None:

        # self.data = data
        self.column_definition = column_definition
        self.params = params

        self.data_map = self.convert_data(data)

    def convert_data(self, data):
        """
            Slice or not slice
        """

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        # Dev
        def convert_to_array(input_data, lags):
            time_steps = len(input_data)
            # lags = 257      # self.time_steps
            x = input_data.to_numpy()
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            if time_steps >= lags:
                return x

        def mapper(data_len, lags):
            start = np.arange(data_len - lags + 1)
            end = start + lags
            return np.stack([start, end], axis=1)
        
        data_map = {'index': []}
        # total_index = 0
        for _, sliced in data.groupby(id_col):

            col_mappings = {
                'identifier': [id_col],
                'time': [time_col],
                'outputs': [target_col],
                'inputs': input_cols
            }

            # No index here
            # total_index += (len(sliced) - 257)  # check here maybe + 1
            map = mapper(len(sliced), self.params['total_time_steps'])
            data_map['index'].append(map)

            for k in col_mappings:
                cols = col_mappings[k]

                arr = convert_to_array(sliced[cols], self.params['total_time_steps'])


                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)   # hangisi nerde belli değil

        return data_map

    
    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""

        return get_single_col_by_input_type(input_type, self.column_definition)

    def __len__(self):
        return len(self.data_map['index'])

    def __getitem__(self, idx):
        """
            Index to slices and then to tensors
        """

        index = self.data_map['index'][idx]
        start, end = index[0], index[1]

        inputs = self.data_map['inputs'][start:end]
        outputs = self.data_map['outputs'][start:end][self.params['num_encoder_steps']:, :]

        return inputs, outputs
        return {'inputs': inputs, 'outputs': outputs}


if __name__ == '__main__':
    dataset = TFTDataset('output/hourly_electricity.csv')

