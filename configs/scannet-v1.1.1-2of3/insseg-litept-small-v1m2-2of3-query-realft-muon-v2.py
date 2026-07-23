# real-ssl-v2 leg of the Muon real-scene finetune (-query-realft-muon): identical
# MuonAdamW optimizer/scheduler/init (embed model_best.pth backbone transfer) —
# ONLY the real-scene dataset changes to the v2 SSL label export. See
# -query-realft-v2.py for the v2 dataset provenance and conversion command.
#
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon-v2 \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon-v2 -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft-muon.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon-v2"

real_data_root = "data/real-ssl-v2"

# Same nested override as -query-realft-v2.py: train.datasets is a list and
# replaces wholesale on merge, so it is restated with base pipelines substituted.
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
