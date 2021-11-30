import numpy as np
import pandas as pd


class tf_wrapper:
    def __init__(
        self,
        path = None,
        output_path = None,
        data_formatter = None,

    ) -> None:
        self.data_path = path
        self.output_path = output_path
        self.formatter = data_formatter

        # self._make_dataset()
        # Save Hparams
        # Sets up default params
        # self.fixed_params = self.formatter.get_experiment_params()
        # self.params = self.formatter.get_default_model_params()
        # self.params["model_folder"] = self.output_path
        self._hparams()

    def _hparams(self):
        # fixed_params = self.formatter.get_fixed_params()
        fixed_params = self.formatter.get_experiment_params()
        model_params = self.formatter.get_default_model_params()
        column_def = fixed_params['column_definition']

        h_size = model_params['hidden_layer_size']

        embedding_sizes = {
            item[0]: 
        }


        hparams_out = {
            'hidden_layer_size': h_size,
            'hidden_continuous_size': h_size,
            'embedding_sizes': {'day_of_week': (7, 160), 'day_of_month': (31, 160), 'week_of_year': (53, 160), 'month': (12, 160), 'Region': (4, 160)},
            'x_reals': ['log_vol', 'open_to_close', 'days_from_start'],
            'x_categoricals': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'Region'],
            'reals': ['log_vol', 'open_to_close', 'days_from_start'],
            'static_categoricals': ['Region'],
            'static_reals': [],
            'time_varying_categoricals_encoder': ['day_of_week', 'day_of_month', 'week_of_year', 'month'],
            'time_varying_categoricals_decoder': ['day_of_week', 'day_of_month', 'week_of_year', 'month'],
            'time_varying_reals_encoder': ['log_vol', 'open_to_close', 'days_from_start'],
            'time_varying_reals_decoder': ['days_from_start'],
            'lstm_layers': 1,
            'attention_head_size': model_params['num_heads'],
            'output_size': 3,
            'n_targets': 1,
            'static_variables': ['Region'],
            'encoder_variables': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'days_from_start', 'log_vol', 'open_to_close'],
            'decoder_variables': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'days_from_start'],
        }

        return hparams_out
    
    def _make_dataset(self):
        print("Loading & splitting data...")
        raw_data = pd.read_csv(self.data_path, index_col=0)
        train, valid, test = self.formatter.split_data(raw_data)
        train_samples, valid_samples = self.formatter.get_num_samples_for_calibration(
        )

        # Dev
        train = self._batch_data(train, self.formatter.get_column_definition())
        data, labels = train['inputs'], train['outputs']
        # Val
        valid = self._batch_data(valid, self.formatter.get_column_definition())
        valid_data, valid_labels = valid['inputs'], valid['outputs']

        # # Sets up default params
        # fixed_params = self.formatter.get_experiment_params()
        # params = self.formatter.get_default_model_params()
        # params["model_folder"] = self.output_path

        self.train = [data, labels]
        self.valid = [valid_data, valid_labels]

        self.model_params = params
        self.fixed_params = fixed_params

        return data, labels, valid_data, valid_labels, params, fixed_params

    def _batch_data(self, data, column_definition):
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
            