"""Real-scene SSL-labeled dataset (VisionSCO save_instance_labels exports).

Scenes are single compressed ``<split>/<scene>.npz`` files produced by
tools/convert_ssl_scenes.py (compressed because the per-asset .npy layout costs
~2.5x more disk — ~120 GB for 80k scenes). Each npz holds coord/color/normal/
segment/instance arrays. Label semantics differ from the synthetic set: `segment`
holds only OBJECT (6) and the UNKNOWN_BG sentinel (7, "known not-object");
`instance` is -1 for background AND for object points the matcher left ungrouped.
The MQ-v1m1 not-object loss / token-ignore path consumes this encoding; see
tools/convert_ssl_scenes.py for the full table.
"""
import glob
import os

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class RealSSLDataset(DefaultDataset):
    def get_data_list(self):
        split_list = [self.split] if isinstance(self.split, str) else self.split
        data_list = []
        for split in split_list:
            data_list += sorted(
                glob.glob(os.path.join(self.data_root, split, "*.npz"))
            )
        return data_list

    def get_data_name(self, idx):
        name = os.path.basename(self.data_list[idx % len(self.data_list)])
        return name[: -len(".npz")]

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with np.load(data_path) as npz:
            data_dict = dict(
                coord=npz["coord"].astype(np.float32),
                color=npz["color"].astype(np.float32),
                normal=npz["normal"].astype(np.float32),
                segment=npz["segment"].reshape([-1]).astype(np.int32),
                instance=npz["instance"].reshape([-1]).astype(np.int32),
            )
        data_dict["name"] = self.get_data_name(idx)
        data_dict["split"] = self.get_split_name(idx)
        return data_dict
