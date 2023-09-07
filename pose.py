import numpy as np
import gzip, json
from scipy.spatial.transform import Rotation as R

idx = 2000
file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

#load guide annotations 
def load_annotation(file_path, idx):
    with gzip.open(file_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]
    instruction_id = train_guide_data[idx]['instruction_id']
    scene_id = train_guide_data[idx]['scan']
    return instruction_id, scene_id

#extract rotation and position in world coordinate for one timestep 
def extract_one_camera_param(extrinsic_matrix):
    transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pose_habitat = np.dot(extrinsic_matrix, transform)
    # pose_habitat = extrinsic_matrix
    R = pose_habitat[:3,:3]
    T = pose_habitat[:3,-1]
    RC = np.transpose(R)
    TC = -np.dot(np.transpose(R), T)
    return TC, RC

#extract camera poses 
def extract_camera_params(extrinsic_matrices):
    num_poses = extrinsic_matrices.shape[0]
    rotations = np.zeros((num_poses, 3, 3))
    positions = np.zeros((num_poses, 3))
    for i in range(num_poses):
        positions[i],rotations[i] = extract_one_camera_param(extrinsic_matrices[i])
    return positions, rotations

sample_ins, sample_scene = load_annotation(file_path, idx)
sample_pose_trace = np.load("/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/"+'{:0>6}'.format(sample_ins)+"_guide_pose_trace.npz")
sample_pose = sample_pose_trace['extrinsic_matrix']

positions, rotations = extract_camera_params(sample_pose)
# sample_extrinsic_matrix = extrinsic_matrix[0]
# T, R = extract_one_camera_params(sample_extrinsic_matrix)

# rotation matrix to angle axis
# rotation = rotations[1290]
def R_to_angle_axis(R):
    angle = np.arccos((np.trace(R)-1)/2)
    axis = 1/(2*np.sin(angle))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return angle, axis
angle, axis = R_to_angle_axis(rotations[0])

def rotations_to_angles_axes(rotations):
    num_poses = rotations.shape[0]
    angles = np.zeros((num_poses))
    axes = np.zeros((num_poses, 3))
    for i in range(num_poses):
        angles[i], axes[i] = R_to_angle_axis(rotations[i])
    return angles, axes

angles, axes = rotations_to_angles_axes(rotations)

# pano = sample_pose['pano']
# time = sample_pose['time']

