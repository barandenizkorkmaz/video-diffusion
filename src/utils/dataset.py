import pandas as pd

def get_splitting_dates(train='~/PycharmProjects/timeseries-nowcasting/data/train_dates.csv',
                        valid='~/PycharmProjects/timeseries-nowcasting/data/valid_dates.csv',
                        test='~/PycharmProjects/timeseries-nowcasting/data/test_dates.csv',
                        col_name='date'):
    train_dates = pd.read_csv(train)
    train_dates = pd.DatetimeIndex(train_dates[col_name].values)
    valid_dates = pd.read_csv(valid)
    valid_dates = pd.DatetimeIndex(valid_dates[col_name].values)
    test_dates = pd.read_csv(test)
    test_dates = pd.DatetimeIndex(test_dates[col_name].values)
    return train_dates, valid_dates, test_dates

