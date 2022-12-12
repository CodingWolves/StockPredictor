import qlib as q
from qlib.contrib.data.handler import Alpha158, DataHandlerLP


class Alpha158TwoWeeks(Alpha158):
    """
    A data handler which labels the items with price increase in next 10 trading days.
    """

    def get_label_config(self):
        """
        Override the get_label_config method in Alpha158.
        This will build a data handler that labels the items with price increase in next 10 trading days.
        """
        return ["Ref($close, -10)/Ref($close, -1) - 1"], ["LABEL0"]


q.init(provider_uri='stockData/cn_data_1min')

instruments = 'csi300'
start_time = '2020-12-01 09:00:00'
end_time = '2020-12-08 14:55:00'
fit_start_time = '2020-12-01 09:00:00'
fit_end_time = '2020-12-06 14:55:00'
valid_start_time = '2020-12-07 09:00:00'
valid_end_time = '2020-12-07 14:55:00'
test_start_time = '2020-12-08 09:00:00'
test_end_time = '2020-12-08 14:55:00'
freq = '1min'

# instruments = D.instruments(market='all')

fields = ['$open', '$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
# dataFrame = D.features(instruments, fields, start_time=start_time, end_time=end_time, freq=freq)
# print(dataFrame.head(10).to_string())


h = Alpha158(
    instruments=instruments,
    freq='1min',
#     **{
#     "start_time": start_time,
#     "end_time": end_time,
#     "fit_start_time": fit_start_time,
#     "fit_end_time": fit_end_time,
#     "instruments": instruments,  # 'csi300', #
#     "freq": freq,
# }
)

from qlib.data.dataset import DatasetH

print(111)
# print(h.get_cols())
# print(h.fetch(col_set="label").head(10000000).to_string())
# print(h)

dataset = DatasetH(handler=Alpha158TwoWeeks(h), segments={
    'train': (start_time, fit_end_time),
    'valid': (start_time, fit_end_time),
    'test': (test_start_time, test_end_time),
})

#
df_train, df_valid = dataset.prepare(
    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
)
# x_train, y_train = df_train["feature"], df_train["label"]
# x_valid, y_valid = df_valid["feature"], df_valid["label"]
#
# import pandas as pd
# import numpy as np
#
# print(222)
# # get weights
# try:
#     wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"],
#                                            data_key=DataHandlerLP.DK_L)
#     w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
# except KeyError as e:
#     w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
#     w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)


from qlib.contrib.model.gbdt import LGBModel

model = LGBModel(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.0421,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
)

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

print("trained data")
example_df = dataset.prepare("train")
print(example_df.head())

with R.start(experiment_name="stock_predictor", recorder_name="new_model"):
    R.log_params(**flatten_dict({"loss": "mse",
                                 "colsample_bytree": 0.8879,
                                 "learning_rate": 0.0421,
                                 "subsample": 0.8789,
                                 "lambda_l1": 205.6999,
                                 "lambda_l2": 580.9768,
                                 "max_depth": 8,
                                 "num_leaves": 210,
                                 "num_threads": 20,
                                 }))
    print('model fit:')
    model.fit(dataset)

    R.save_objects(**{"model.pkl": model})

    print('pred:')
    print(model.predict(dataset))

    print('recorder:')
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
