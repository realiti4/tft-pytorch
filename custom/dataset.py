import numpy as np
import pandas as pd

from data.volatility import VolatilityFormatter
from data.utils import get_single_col_by_input_type
from data.base import InputTypes



def make_dataset():
    data_csv_path = 'output/formatted_omi_vol.csv'
    model_folder = 'output/volatility/'
    data_formatter = VolatilityFormatter()

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )

    # Dev
    train = _batch_data(train, data_formatter.get_column_definition())
    data, labels = train['inputs'], train['outputs']

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    return data, labels, params, fixed_params

def _batch_data(data, column_definition):
    """Batches data for training.

    Converts raw dataframe from a 2-D tabular format to a batched 3-D array
    to feed into Keras model.

    Args:
        data: DataFrame to batch

    Returns:
        Batched Numpy array with shape=(?, self.time_steps, self.input_size)
    """

    

    num_encoder_steps = 252

    # Functions.
    def _batch_single_entity(input_data):
        time_steps = len(input_data)
        lags = 257      # self.time_steps
        x = input_data.values
        if time_steps >= lags:
            return np.stack(
                [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)

        else:
            return None

    def _get_single_col_by_type(input_type):
        """Returns name of single column for input type."""

        return get_single_col_by_input_type(input_type, column_definition)

    id_col = _get_single_col_by_type(InputTypes.ID)
    time_col = _get_single_col_by_type(InputTypes.TIME)
    target_col = _get_single_col_by_type(InputTypes.TARGET)
    input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    data_map = {}
    for _, sliced in data.groupby(id_col):

        col_mappings = {
            'identifier': [id_col],
            'time': [time_col],
            'outputs': [target_col],
            'inputs': input_cols
        }

        for k in col_mappings:
            cols = col_mappings[k]
            arr = _batch_single_entity(sliced[cols].copy())

            if k not in data_map:
                data_map[k] = [arr]
            else:
                data_map[k].append(arr)

    # Combine all data
    for k in data_map:
        data_map[k] = np.concatenate(data_map[k], axis=0)

    # Shorten target so we only get decoder steps
    data_map['outputs'] = data_map['outputs'][:,
                                                num_encoder_steps:, :]

    active_entries = np.ones_like(data_map['outputs'])
    if 'active_entries' not in data_map:
        data_map['active_entries'] = active_entries
    else:
        data_map['active_entries'].append(active_entries)

    return data_map



if __name__ == '__main__':
    make_dataset()


    print('de')