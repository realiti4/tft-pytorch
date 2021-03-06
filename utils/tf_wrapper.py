import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from data_formatters.base import InputTypes, DataTypes
from data_formatters.utils import get_single_col_by_input_type
from tft_dataset import TFTDataset


class tf_wrapper:
    def __init__(
        self,
        path = None,
        output_path = None,
        data_formatter = None,
        batch_size: int = 64,
        test = False,
    ) -> None:

        self.data_path = path
        self.output_path = output_path
        self.formatter = data_formatter
        self.batch_size = batch_size
        self.test = test


    def _hparams(self):
        # fixed_params = self.formatter.get_fixed_params()
        fixed_params = self.formatter.get_experiment_params()
        model_params = self.formatter.get_default_model_params()
        column_def = fixed_params['column_definition']

        h_size = model_params['hidden_layer_size']
        self.fixed_params = fixed_params

        # Dev
        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [
                tup for tup in defn if tup[1] == data_type and
                tup[2] not in {InputTypes.ID, InputTypes.TIME}
            ]

        input_cols = [
            tup for tup in column_def
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED, input_cols)
        cat_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL, input_cols)

        embedding_sizes = {
            tup[0]: (fixed_params['category_counts'][i], h_size)
            for i, tup in enumerate(cat_inputs)
        }


        hparams_out = {
            'hidden_layer_size': h_size,
            'hidden_continuous_size': h_size,
            'embedding_sizes': embedding_sizes,
            'x_reals': [tup[0] for tup in real_inputs],
            'x_categoricals': [tup[0] for tup in cat_inputs],
            'reals': [tup[0] for tup in real_inputs],       # Compare this with x_reals
            'static_categoricals': [input_cols[i][0] for i in fixed_params['static_input_loc']],
            'static_reals': [],     # passing this for now
            'time_varying_categoricals_encoder': [
                tup[0] for tup in cat_inputs
                if tup[2] not in {InputTypes.STATIC_INPUT}
            ],
            'time_varying_categoricals_decoder': [
                tup[0] for tup in cat_inputs
                if tup[2] not in {InputTypes.STATIC_INPUT}
            ],
            'time_varying_reals_encoder': [
                tup[0] for tup in real_inputs
                if tup[2] not in {InputTypes.STATIC_INPUT}
            ],
            'time_varying_reals_decoder': [input_cols[i][0] for i in fixed_params['known_regular_inputs']],
            'lstm_layers': 1,
            'attention_head_size': model_params['num_heads'],
            'output_size': 3,
            'n_targets': 1,
            'static_variables': [input_cols[i][0] for i in fixed_params['static_input_loc']],
            # 'encoder_variables': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'days_from_start', 'log_vol', 'open_to_close'],
            # 'decoder_variables': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'days_from_start'],
        }

        # This is a mess!!
        hparams_out.update({
            'encoder_variables': hparams_out['time_varying_categoricals_encoder'] \
                + [input_cols[i][0] for i in fixed_params['known_regular_inputs']] \
                    + [
                        item for item in hparams_out['time_varying_reals_encoder']
                        if item not in hparams_out['time_varying_reals_decoder']
                    ],
            'decoder_variables': hparams_out['time_varying_categoricals_decoder'] \
                + [input_cols[i][0] for i in fixed_params['known_regular_inputs']]
        })
        self.hparams = hparams_out

        hparams_example = {     # Old one for compare
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
    
    def make_dataset(self):
        # Load samples
        print("Loading & splitting data...")
        raw_data = pd.read_csv(self.data_path, index_col=0)
        # raw_data = pd.read_parquet(self.data_path)
        # # raw_data.pop('unique_id')
        # # raw_data = raw_data.astype({
        # #     'transactions': np.float32, 'class': np.int16, 'perishable': np.int8, 'item_nbr': np.int32,
        # #     'traj_id': 'category'
        # #     })
        # # # raw_data = raw_data.astype({'national_hol': 'category', 'regional_hol': 'category', 'local_hol': 'category', 'family': 'category'})
        # # raw_data.to_parquet('temporal3.parquet')
        # raw_data.dropna(inplace=True)
        # print('saved run!!')

        if self.test:
            raw_data = raw_data.iloc[:int(len(raw_data)/20)]

        train, valid, test = self.formatter.split_data(raw_data)
        train_samples, valid_samples = self.formatter.get_num_samples_for_calibration()
        # Calculate params
        self._hparams()

        # Dev - create datasets
        train_dataset = TFTDataset(train, self.formatter.get_column_definition(), self.fixed_params)
        val_dataset = TFTDataset(valid, self.formatter.get_column_definition(), self.fixed_params)
        # test
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        # self.test = train_dataloader

        return train_dataloader, val_dataloader
