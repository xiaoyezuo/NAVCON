import os 
import numpy as np
import gzip, json

file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

with gzip.open(file_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]

instruction_id_video = np.array([data['instruction_id'] for data in train_guide_data])
scene_ids = np.array([data['scan'] for data in train_guide_data])
instruction_id_language = np.load("instruction_id.npy")
extraction_idx = np.array([np.where(instruction_id_video == id)[0][0] for id in instruction_id_language])

print(extraction_idx.shape)