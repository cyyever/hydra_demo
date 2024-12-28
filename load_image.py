import argparse
import copy
import json
import os

import cyy_torch_vision  # noqa: F401
import torch
from cyy_naive_lib.algorithm.mapping_op import (
    change_mapping_keys,
    get_mapping_values_by_key_order,
)
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    ClassificationDatasetCollection,
    DatasetCollection,
    MachineLearningPhase,
)
from cyy_torch_toolbox.dataset import DatasetCollectionConfig
from cyy_torch_vision import VisionDatasetUtil


def get_instance_statistics(tester, instance_dataset) -> dict:
    tmp_validator = copy.deepcopy(tester)
    tmp_validator.dataset_collection.transform_dataset(
        MachineLearningPhase.Test, lambda _: instance_dataset
    )
    tmp_validator.inference(sample_prob=True)
    return tmp_validator.prob_metric.get_prob(1)[0]


def save_training_image(
    save_dir: str, dc: ClassificationDatasetCollection, index: int
) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"index_{index}.jpg")
    util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
    assert isinstance(util, VisionDatasetUtil)
    util.save_sample_image(index, image_path)
    labels = util.get_sample_label(index)
    assert len(labels) == 1
    return image_path, dc.get_label_names()[list(labels)[0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    dc_config = DatasetCollectionConfig(dataset_name=args.dataset_name)
    dc: DatasetCollection = dc_config.create_dataset_collection()
    assert isinstance(dc, ClassificationDatasetCollection)
    json_path = os.path.join(
        "contribution", args.dataset_name, "approx_hydra_contribution.json"
    )
    if not os.path.isfile(json_path):
        raise RuntimeError("unknown dataset:" + args.dataset_name)

    with open(json_path, encoding="utf8") as f:
        contribution_dict = json.load(f)
    contribution_dict = change_mapping_keys(contribution_dict, int, True)
    keys = list(sorted(contribution_dict.keys()))
    contribution = torch.Tensor(
        list(get_mapping_values_by_key_order(contribution_dict))
    )

    std, mean = torch.std_mean(contribution)
    max_contribution = torch.max(contribution)
    min_contribution = torch.min(contribution)

    log_info("std is %s", std)
    log_info("mean is %s", mean)
    log_info("max contribution is %s", max_contribution)
    log_info("min contribution is %s", min_contribution)
    log_info("positive contributions is %s", contribution[contribution >= 0].shape)
    log_info("negative contributions is %s", contribution[contribution < 0].shape)

    image_info = {}
    mask = contribution >= max_contribution * 0.9
    for idx in mask.nonzero().tolist():
        idx = keys[idx[0]]
        image_dir = os.path.join("image", args.dataset_name, "positive")
        image_path, label_name = save_training_image(image_dir, dc, idx)
        image_info[str(idx)] = [
            image_path,
            label_name,
            contribution_dict[idx],
        ]

    mask = contribution < min_contribution * 0.9
    for idx in mask.nonzero().tolist():
        idx = keys[idx[0]]
        image_dir = os.path.join("image", args.dataset_name, "negative")
        image_path, label_name = save_training_image(image_dir, dc, idx)
        image_info[str(idx)] = [
            image_path,
            label_name,
            contribution_dict[idx],
        ]
    with open(args.dataset_name + ".result.json", "w", encoding="utf8") as f:
        json.dump(image_info, f)
