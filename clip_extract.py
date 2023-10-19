import os
import numpy as np
import gzip, json, shutil
from scipy.spatial.transform import Rotation as R

file_path = "/home/.../rxr_train_guide.jsonl.gz" # path to rxr train guide data
dir_path = "/home/.../navcon_video/" # path to current directory 

#load instruction id
instruction_ids = np.load("instruction_id.npy")

#load concept time mapping
concept_timestamp = np.load("concept_timestamp.npy", allow_pickle=True).item()

#find the index of the closest timestamp to input time 
def closest_index(t, timestamps):
    diff = abs(timestamps-t)
    return np.argmin(diff)
#retrieve indices of start and end timestamp of all concepts in one instruction
def concept_time_to_concept_index(concept_time, timestamps):
    concept_index = []
    for entry in concept_time:
        concept_id = entry[0]
        start_idx = closest_index(entry[1], timestamps)
        end_idx = closest_index(entry[2], timestamps)
        concept_index.append(np.array([concept_id, start_idx, end_idx]))
    return np.array(concept_index)

#extract images corresponding to each concept for one instruction 
def extract_frames(instruction_id, concept_idx, vid_dir, out_dir):
    
    for i in range(concept_idx.shape[0]):
        dst_dir = os.path.join(out_dir, str(i)+"/")
        #check if directory exists
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        start_idx = int(concept_idx[i][1])
        end_idx = int(concept_idx[i][2])
        for j in range(start_idx, end_idx):
            src_file = vid_dir + '{:0>6}'.format(j) + ".png"
            dst_file = dst_dir + '{:0>6}'.format(j) + ".png"
            shutil.copyfile(src_file, dst_file)

#extract images corresponding to each concept for all instructions  
def extract_all(instruction_ids, concept_timestamp):
    #loop through all instruction ids
    for instruction_id in instruction_ids:
        vid_dir = "rxr_clips/"+'{:0>6}'.format(instruction_id)+"/"
        out_dir = dir_path + "concept_clips/"+'{:0>6}'.format(instruction_id)+"/"
        #skip existing clips 
        if os.path.isdir(out_dir) or (instruction_id not in concept_timestamp):
            continue
        #extract images for each instruction
        concept_time = concept_timestamp[instruction_id]
        pose_trace = np.load("/home/.../rxr-data/pose_traces/rxr_train/"+'{:0>6}'.format(instruction_id)+"_guide_pose_trace.npz")
        timestamps = pose_trace['time'][::10]
        concept_idx = concept_time_to_concept_index(concept_time, timestamps)
        extract_frames(instruction_id, concept_idx, vid_dir, out_dir)

#execute extraction
def main():
    extract_all(instruction_ids, concept_timestamp)

if __name__ == "__main__":
    main()