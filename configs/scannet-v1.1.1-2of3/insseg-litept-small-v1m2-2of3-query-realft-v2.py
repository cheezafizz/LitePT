# real-ssl-v2 leg of insseg-litept-small-v1m2-2of3-query-realft.py: identical model,
# losses, optimizer, and schedule — ONLY the real-scene dataset changes to the v2
# SSL label export (per-store classifiers + products.csv volume priors in the
# matcher; v2 seg_ids are NOT interchangeable with real-ssl-v1).
#
# Data: data/real-ssl-v2, produced from ~/workspace/jhp/ssl_labels_out_v2 by
#   /home/fai/miniconda3/envs/litept/bin/python tools/convert_ssl_scenes.py \
#     --ssl-root ~/workspace/jhp/ssl_labels_out_v2 --out-root data/real-ssl-v2 \
#     --camera-root ~/workspace/jhp/dataset/SSL_dataset/dataset_ssl_1_80k \
#     --val-stores 001,004 --workers 16
# (resumable: re-run the same command after the v2 export finishes to convert the
# remaining scenes; already-converted scenes are skipped). Same label encoding as
# v1: segment 6=object / 7=unknown-bg sentinel, instance dense 0..K-1 / -1 ignored.
#
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-v2 \
#     -n insseg-litept-small-v1m2-2of3-query-realft-v2 -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-v2"

real_data_root = "data/real-ssl-v2"

# The base config bakes real_data_root into the data dict, and list values
# (train.datasets) replace wholesale on merge — so the train branch is restated
# here with the pipelines pulled from the base via {{_base_.*}} substitution.
data = dict(
    train=dict(
        datasets=[
            dict(
                type="RealSSLDataset",
                split="train",
                data_root="data/real-ssl-v2",
                transform={{_base_.real_train_transform}},
                test_mode=False,
                loop=1,
            ),
            dict(
                type="ScanNetDataset",
                split="train",
                data_root={{_base_.syn_data_root}},
                transform={{_base_.syn_train_transform}},
                test_mode=False,
                loop=1,
            ),
        ],
    ),
    val=dict(data_root="data/real-ssl-v2"),
)
