import os
import numpy as np
import gzip, json, shutil
from scipy.spatial.transform import Rotation as R

instruction_id = 28

file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"
dir_path = os.path.dirname(os.path.realpath(__file__))
vid_path = os.path.join(dir_path, "output/"+'{:0>6}'.format(instruction_id)+"/")
output_path = os.path.join(dir_path, "extracted_output/"+'{:0>6}'.format(instruction_id)+"/")

# #load guide annotations 
# with gzip.open(file_path, 'r') as f:
#     train_guide_data = [json.loads(line) for line in f]

#load concept time mapping
concept_time_mapping = np.load("/home/zuoxy/ceph_old/action_recognition/concept_time_mapping.npy", allow_pickle=True)
concept_time_dict = concept_time_mapping.item()
# concept_time = concept_time_dict[str(instruction_id)]
concept_time = np.array([[],[1, 15.9, 18.8]])

#extract timestamps for each instruction id
pose_trace = np.load("/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/"+'{:0>6}'.format(instruction_id)+"_guide_pose_trace.npz")
timestamps = pose_trace['time']
#print(timestamp.shape)
# print(concept_time.shape)

def closest_index(t, timestamps):
    diff = abs(timestamps-t)
    return np.argmin(diff)

def concept_time_to_concept_index(concept_time, timestamps):
    concept_index = []
    for entry in concept_time:
        concept_id = entry[0]
        start_idx = closest_index(entry[1], timestamps)
        end_idx = closest_index(entry[2], timestamps)
        concept_index.append(np.array([concept_id, start_idx, end_idx]))
    return np.array(concept_index)

def extract_frames(instruction_id, concept_idx):

    vid_dir = os.path.join(dir_path, "output/"+'{:0>6}'.format(instruction_id)+"/")
    out_dir = os.path.join(dir_path, "extracted_output/"+'{:0>6}'.format(instruction_id)+"/")
    for i in range(concept_idx.shape[0]):
        dst_dir = os.path.join(out_dir, str(i)+"/")
        #check if directory exists
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        start_idx = int(concept_idx[i][1])
        end_idx = int(concept_idx[i][2])
        for j in range(start_idx, end_idx):
            src_file = vid_dir + str(j) + ".png"
            dst_file = dst_dir + str(j) + ".png"
            shutil.copyfile(src_file, dst_file)
        

concept_index = concept_time_to_concept_index(concept_time, timestamps)
# print(concept_index, concept_index[0][1])
extract_frames(instruction_id, concept_index)