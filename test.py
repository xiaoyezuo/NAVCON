import numpy as np
import gzip, json
from scipy.spatial.transform import Rotation as R

file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

#load guide annotations 
with gzip.open(file_path, 'r') as f:
    train_guide_data = [json.loads(line) for line in f]
instruction_ids = np.array([data['instruction-id'] for data in train_guide_data])
scene_ids = np.array([data['scan'] for data in train_guide_data])

# sample_ins, sample_scene = load_annotation(file_path, idx)
# sample_pose_trace = np.load("/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/"+'{:0>6}'.format(sample_ins)+"_guide_pose_trace.npz")
# pose = sample_pose_trace['extrinsic_matrix']
# time = sample_pose_trace['time']
# print(time.shape)
# print(time[time.shape[0]-1])


# positions, rotations = extract_camera_params(sample_pose)
# sample_extrinsic_matrix = extrinsic_matrix[0]
# T, R = extract_one_camera_params(sample_extrinsic_matrix)
