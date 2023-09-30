import numpy as np
import gzip, json
from scipy.spatial.transform import Rotation as R

#load guide annotations 
def load_annotation(file_path, idx):
    with gzip.open(file_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]
    # print(len(train_guide_data)) = 79467
    instruction_id = train_guide_data[idx]['instruction_id']
    scene_id = train_guide_data[idx]['scan']
    return instruction_id, scene_id

#extract rotation and position in world coordinate for one timestep 
def extract_one_camera_param(extrinsic_matrix):
    # transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    # pose_habitat = np.dot(extrinsic_matrix, transform)
    pose_habitat = extrinsic_matrix
    R = pose_habitat[:3,:3]
    T = pose_habitat[:3,-1]
    RC = np.transpose(R)
    TC = -np.dot(np.transpose(R), T)
    return T, R

#extract camera poses 
def extract_camera_params(extrinsic_matrices):
    num_poses = extrinsic_matrices.shape[0]
    rotations = np.zeros((num_poses, 3, 3))
    positions = np.zeros((num_poses, 3))
    for i in range(num_poses):
        positions[i],rotations[i] = extract_one_camera_param(extrinsic_matrices[i])
    return positions, rotations

