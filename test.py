import os 
import numpy as np
import gzip, json

file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

with gzip.open(file_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]

instruction_id = np.array([data['instruction_id'] for data in train_guide_data])

# # scene_ids = np.array([data['scan'] for data in train_guide_data])
# instruction_id = np.load("instruction_id.npy")
# print(instruction_id[:10])
# # extraction_idx = np.array([np.where(instruction_id_video == id)[0][0] for id in instruction_id_language])
# instruction_id = 24
# # sample_pose_trace = np.load(f"/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/{instruction_id:06d}_guide_pose_trace.npz")
# # sample_time = sample_pose_trace['time']
# # time_new = sample_time[::10]
# # print(sample_time[:20], time_new[:20])

# concept_timestamp = np.load("/home/zuoxy/ceph_old/action_recognition/concept_timestamp.npy", allow_pickle=True).item()
# concept_time = concept_timestamp[instruction_id]
# print(concept_time)

#sample 100 instructions 
num_sample = 1000
i = 0
valid_id = []
for id in instruction_id:
    path = "/home/zuoxy/ceph_old/action_recognition/clips/"+'{:0>6}'.format(id)+'/'
    if(os.path.isdir(path)):
        valid_id.append(id)
valid_id = np.array(valid_id)

sampled_id = np.random.choice(valid_id, 100, replace=False)
for id in sampled_id:
    cmd = 'cp -r '+"/home/zuoxy/ceph_old/action_recognition/clips/"+'{:0>6}'.format(id)+'/ ' + "/home/zuoxy/ceph_old/action_recognition/eval_clips/"+'{:0>6}'.format(id)+'/ '
    os.system(cmd)
