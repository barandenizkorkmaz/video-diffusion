import h5py
import pandas as pd
import numpy as np

class ASIReader():
    def __init__(
            self,
            hdf5_file,
            key_group,
            key_array='asi_data',
            key_ts='timestamps',
            num_prev_asi=0,
            num_post_asi=0,
            time_delta=1,
            exclude_current_asi=False,
            tz=None, unit='min',
            squeeze_single_asi=True
    ):
        self.tz = tz
        self.file = h5py.File(hdf5_file, 'r')
        self.asi_group = self.file[key_group]
        self.key_array = key_array
        self.key_ts = key_ts
        self.squeeze_single_asi = squeeze_single_asi
        self.num_asi = num_prev_asi + num_post_asi + int(not exclude_current_asi)
        self.exclude_current_asi = exclude_current_asi
        self.prev_asi = [pd.Timedelta(i, unit=unit) for i in range(1, num_prev_asi * time_delta + 1, time_delta)]
        self.post_asi = [pd.Timedelta(i, unit=unit) for i in range(1, num_post_asi * time_delta + 1, time_delta)]
        self.all_timestamps = self._get_all_asi_timestamps()

    def __call__(self, o):
        if type(o) == pd.Series:
            timestamp = o.name
        else:
            timestamp = o
        key_asi_ds = timestamp.strftime('%Y/%m/%d')
        asi_ds = self.asi_group[key_asi_ds]
        timestamps = pd.to_datetime(np.array(asi_ds[self.key_ts]).astype("uint64"), utc=True)
        if self.tz:
            timestamps = timestamps.tz_convert(tz=self.tz)
        asi_ts = pd.DatetimeIndex([])
        if len(self.prev_asi) > 0:
            asi_ts = asi_ts.union(pd.to_datetime([timestamp - i for i in self.prev_asi]))
        if len(self.post_asi) > 0:
            asi_ts = asi_ts.union(pd.to_datetime([timestamp + i for i in self.post_asi]))
        if not self.exclude_current_asi:
            asi_ts = asi_ts.union([timestamp])
        int_idx = np.where(timestamps.isin(asi_ts))[0]
        assert len(int_idx) == self.num_asi, f'Could not fetch all {self.num_asi} images (only found {len(int_idx)}).'
        asi = np.array(asi_ds[self.key_array][int_idx])
        if self.squeeze_single_asi and self.num_asi == 1:
            asi = asi.squeeze()
        return asi.transpose((0, 3, 1, 2))

    def check_asi_availability(self, timestamps):
        all_asi_available = np.all(timestamps.isin(self.all_timestamps))
        for delta_t in self.prev_asi:
            if not np.all((timestamps - delta_t).isin(self.all_timestamps)):
                all_asi_available = False
                break
        for delta_t in self.post_asi:
            if not np.all((timestamps + delta_t).isin(self.all_timestamps)):
                all_asi_available = False
                break
        return all_asi_available

    def get_valid_timestamps(self, timestamps):
        valid_timestamps = timestamps.intersection(self.all_timestamps)
        for delta_t in self.prev_asi:
            shifted_ts = timestamps - delta_t
            prev_available = shifted_ts.isin(self.all_timestamps)
            valid_timestamps = valid_timestamps.intersection(timestamps[prev_available])
        for delta_t in self.post_asi:
            shifted_ts = timestamps + delta_t
            post_available = shifted_ts.isin(self.all_timestamps)
            valid_timestamps = valid_timestamps.intersection(timestamps[post_available])
        return pd.DatetimeIndex(valid_timestamps)

    def _get_all_asi_timestamps(self):
        all_timestamps = None
        for year in self.asi_group.keys():
            for month in self.asi_group[year].keys():
                for day in self.asi_group[year][month].keys():
                    timestamps = np.array(self.asi_group[year][month][day][self.key_ts]).astype("uint64")
                    if all_timestamps is None:
                        all_timestamps = timestamps
                    else:
                        all_timestamps = np.hstack([all_timestamps, timestamps])
        all_timestamps = pd.to_datetime(all_timestamps, utc=True)
        if self.tz:
            return all_timestamps.tz_convert(tz=self.tz)
        else:
            return all_timestamps