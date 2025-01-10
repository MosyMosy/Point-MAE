import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from utils.misc import fps


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        test_data_list_file = os.path.join(self.data_root, "test.txt")

        self.sample_points_num = config.npoints
        self.whole = config.get("whole")

        print_log(
            f"[DATASET] sample out {self.sample_points_num} points",
            logger="ShapeNet-55",
        )
        print_log(f"[DATASET] Open file {self.data_list_file}", logger="ShapeNet-55")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, "r") as f:
                test_lines = f.readlines()
            print_log(
                f"[DATASET] Open file {test_data_list_file}", logger="ShapeNet-55"
            )
            lines = test_lines + lines

        self.stat_norm = config.stat_norm
            
        self.sample_method = config.sample_method


        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            model_id = line.split("-")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )
        print_log(
            f"[DATASET] {len(self.file_list)} instances were loaded",
            logger="ShapeNet-55",
        )

        self.permutation = np.arange(self.npoints)

    def pc_to_tensor(self, pc):
        """pc: NxC, return NxC"""
        pc = torch.from_numpy(pc).float()
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        torch.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample["file_path"])).astype(
            np.float32
        )
        data = self.pc_to_tensor(data)
        if self.sample_method == "random":
            data = self.random_sample(data, self.sample_points_num) # random sample
        elif self.sample_method == "fps":
            data = fps(data.unsqueeze(0), self.sample_points_num)[0] # fps sample
                    
        if self.stat_norm:
            data = (data - 0.5335) / 0.2276
        
        return sample["taxonomy_id"], sample["model_id"], data

    def __len__(self):
        return len(self.file_list)

    # mean = 0.5335
    # std = 0.2276
