# [setup]
import math
import os
import magnum as mn
import numpy as np
import gzip, json
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import time
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import pose
import quaternion 
from pathlib import Path


file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"
# save_root_dir = "/mnt/kostas-graid/datasets"
dir_path = Path("/mnt/kostas-graid/datasets/navcon/") #os.path.dirname(os.path.realpath(__file__))

def save_img(rgbd_obs, output_dir, save_index):
    rgb_obs = rgbd_obs[:, :, :3]
    before_from_array = time.time()
    img = Image.fromarray(rgb_obs)
    before_save = time.time()
    filepath = output_dir / f"{save_index:06d}.png"
    img.save(filepath)
    after_save = time.time()
    

def get_obs(sim, show, save):
    # render sensor ouputs and optionally show them
    rgb_obs = sim.get_sensor_observations()["rgba_camera"]
    semantic_obs = sim.get_sensor_observations()["semantic_camera"]
    return rgb_obs, semantic_obs

def place_agent_custom(sim, position, rotation):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = position
    agent_state.rotation = quaternion.from_rotation_matrix(rotation)
    sim.agents[0].set_state(agent_state)#set state 
    return sim.agents[0].scene_node.transformation_matrix()

def make_configuration(scene_id: str):
    scene_file = f"/home/zuoxy/ceph_old/data/mp3d/{scene_id}/{scene_id}.glb"
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup rgb and semantic sensors
    camera_resolution = [240, 320]
    sensors = {
        "rgba_camera": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],  # ::: fix y to be 0 later
        },
        "semantic_camera": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


# [/setup]

def process_extraction_idx(sim, instruction_id, scene_id):

    before_standup = time.time()
    output_dir = dir_path / f"sampled_output_10/{instruction_id:06d}/"

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_pose_trace = np.load(f"/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/{instruction_id:06d}_guide_pose_trace.npz")
    sample_pose = sample_pose_trace['extrinsic_matrix']
    
    positions, rotations = pose.extract_camera_params(sample_pose)
    positions = positions[::10]
    rotations = rotations[::10]
    after_standup = time.time()

    print("Standup time", after_standup - before_standup)
    for idx, (position, rotation) in enumerate(zip(positions, rotations)):
        frame_extract_start = time.time()
        agent_transform = place_agent_custom(sim, position, rotation)  # noqa: F841
        rgb_obs, semantic_obs = get_obs(sim, False, True)
        frame_extract_before_save = time.time()
        save_img(rgb_obs, output_dir, idx)
        frame_extract_end = time.time()
        # print("Save delta:", frame_extract_end - frame_extract_before_save)

def update_sim_instance(prior_sim, prior_scene_id : str, scene_id : str):
    if prior_scene_id == scene_id:
        return prior_sim
    
    if prior_sim is not None:
        prior_sim.close()

    return habitat_sim.Simulator(make_configuration(scene_id))


# This is wrapped such that it can be added to a unit test
def main(num_job_splits : int, job_idx : int):
    with gzip.open(file_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]

    instruction_id_video = np.array([data['instruction_id'] for data in train_guide_data])
    scene_ids = np.array([data['scan'] for data in train_guide_data])
    instruction_id_language = np.load("instruction_id.npy")
    extraction_idx = np.array([np.where(instruction_id_video == id)[0][0] for id in instruction_id_language])    

    setup_scene_id = None
    setup_sim = None

    # iterate through 
    for idx in extraction_idx[job_idx::num_job_splits]:
        instruction_id = instruction_id_video[idx] 
        scene_id = scene_ids[idx]
        setup_sim = update_sim_instance(setup_sim, setup_scene_id, scene_id)
        setup_scene_id = scene_id
        process_extraction_idx(setup_sim, instruction_id, scene_id)

        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_job_splits", type=int, default=1)
    parser.add_argument("--job_idx", type=int, default=0)
    args = parser.parse_args()
    assert args.job_idx < args.num_job_splits
    assert args.num_job_splits >= 1
    main(args.num_job_splits, args.job_idx)