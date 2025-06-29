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
import utils
import quaternion 
from pathlib import Path

project_dir = "" #path to project directory
rxr_train_guide_path = project_dir + "/rxr-data/rxr_train_guide.jsonl.gz"
class FrameExtractor:
    """
    A class for extracting video frames from Habitat simulator based on RXR instruction data.
    """
    
    def __init__(self, file_path=rxr_train_guide_path, 
                 dir_path=project_dir, pose_traces_dir=rxr_dir+"pose_traces/rxr_train/"):
        """
        Initialize the FrameExtractor with data paths and configuration.
        
        Args:
            file_path (str): Path to the RXR training guide JSONL file
            dir_path (str): Base directory for output
            pose_traces_dir (str): Directory containing pose trace files
        """
        self.file_path = file_path
        self.dir_path = Path(dir_path)
        self.pose_traces_dir = pose_traces_dir
        self.camera_resolution = [240, 320]
        
    def save_img(self, rgbd_obs, output_dir, save_index):
        """
        Save image observation to output directory.
        
        Args:
            rgbd_obs: RGBD observation from simulator
            output_dir: Directory to save the image
            save_index: Index for the image filename
        """
        rgb_obs = rgbd_obs[:, :, :3]
        before_from_array = time.time()
        img = Image.fromarray(rgb_obs)
        before_save = time.time()
        filepath = output_dir / f"{save_index:06d}.png"
        img.save(filepath)
        after_save = time.time()
    
    def get_obs(self, sim, show, save):
        """
        Get RGBD observation from habitat camera.
        
        Args:
            sim: Habitat simulator instance
            show: Whether to show the observation
            save: Whether to save the observation
            
        Returns:
            tuple: (rgb_obs, semantic_obs)
        """
        rgb_obs = sim.get_sensor_observations()["rgba_camera"]
        semantic_obs = sim.get_sensor_observations()["semantic_camera"]
        return rgb_obs, semantic_obs

    def place_agent_custom(self, sim, position, rotation):
        """
        Place our agent in the scene.
        
        Args:
            sim: Habitat simulator instance
            position: Agent position
            rotation: Agent rotation matrix
            
        Returns:
            Transformation matrix of the agent
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = quaternion.from_rotation_matrix(rotation)
        sim.agents[0].set_state(agent_state)  # set state 
        return sim.agents[0].scene_node.transformation_matrix()

    def make_configuration(self, scene_id: str):
        """
        Create habitat simulator configuration.
        
        Args:
            scene_id (str): Scene identifier
            
        Returns:
            habitat_sim.Configuration: Simulator configuration
        """
        scene_file = f"/data/mp3d/{scene_id}/{scene_id}.glb"
        
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_file
        backend_cfg.enable_physics = True

        # sensor configuration: setup rgb and semantic sensors
        # Note: all sensors must have the same resolution
        sensors = {
            "rgba_camera": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": self.camera_resolution,
                "position": [0.0, 0.0, 0.0],  # ::: fix y to be 0 later
            },
            "semantic_camera": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": self.camera_resolution,
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

    def process_extraction_idx(self, sim, instruction_id, scene_id):
        """
        Extract scene frames in habitat for each RXR instruction.
        
        Args:
            sim: Habitat simulator instance
            instruction_id: Instruction identifier
            scene_id: Scene identifier
        """
        # before_standup = time.time()
        output_dir = self.dir_path / f"rxr_clips/{instruction_id:06d}/"
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_pose_trace = np.load(f"{self.pose_traces_dir}/{instruction_id:06d}_guide_pose_trace.npz")
        sample_pose = sample_pose_trace['extrinsic_matrix']
        
        positions, rotations = utils.extract_camera_params(sample_pose)
        positions = positions[::10]
        rotations = rotations[::10]
        # after_standup = time.time()

        # print("Standup time", after_standup - before_standup)
        for idx, (position, rotation) in enumerate(zip(positions, rotations)):
            frame_extract_start = time.time()
            agent_transform = self.place_agent_custom(sim, position, rotation)  
            rgb_obs, semantic_obs = self.get_obs(sim, False, True)
            frame_extract_before_save = time.time()
            self.save_img(rgb_obs, output_dir, idx)
            frame_extract_end = time.time()
            # print("Save delta:", frame_extract_end - frame_extract_before_save)

    def update_sim_instance(self, prior_sim, prior_scene_id: str, scene_id: str):
        """
        Update habitat simulator for each instruction to prevent memory leak.
        
        Args:
            prior_sim: Previous simulator instance
            prior_scene_id: Previous scene identifier
            scene_id: Current scene identifier
            
        Returns:
            habitat_sim.Simulator: Updated simulator instance
        """
        if prior_scene_id == scene_id:
            return prior_sim
        
        if prior_sim is not None:
            prior_sim.close()

        return habitat_sim.Simulator(self.make_configuration(scene_id))

    def load_training_data(self):
        """
        Load training data from the JSONL file.
        
        Returns:
            tuple: (train_guide_data, instruction_id_video, scene_ids)
        """
        with gzip.open(self.file_path, 'r') as f:
            train_guide_data = [json.loads(line) for line in f]

        instruction_id_video = np.array([data['instruction_id'] for data in train_guide_data])
        scene_ids = np.array([data['scan'] for data in train_guide_data])
        
        return train_guide_data, instruction_id_video, scene_ids

    def get_extraction_indices(self, instruction_id_video):
        """
        Get extraction indices based on language instruction IDs.
        
        Args:
            instruction_id_video: Array of video instruction IDs
            
        Returns:
            numpy.ndarray: Extraction indices
        """
        instruction_id_language = np.load("instruction_id.npy")
        extraction_idx = np.array([np.where(instruction_id_video == id)[0][0] for id in instruction_id_language])
        return extraction_idx

    def process_job(self, num_job_splits: int, job_idx: int):
        """
        Process a specific job split for parallel processing.
        
        Args:
            num_job_splits: Total number of job splits
            job_idx: Current job index
        """
        train_guide_data, instruction_id_video, scene_ids = self.load_training_data()
        extraction_idx = self.get_extraction_indices(instruction_id_video)

        setup_scene_id = None
        setup_sim = None

        # iterate through all instructions in multiple jobs in parallel 
        for idx in extraction_idx[job_idx::num_job_splits]:
            instruction_id = instruction_id_video[idx] 
            scene_id = scene_ids[idx]
            setup_sim = self.update_sim_instance(setup_sim, setup_scene_id, scene_id)
            setup_scene_id = scene_id
            self.process_extraction_idx(setup_sim, instruction_id, scene_id)


def main(num_job_splits: int, job_idx: int):
    """
    Main function to run frame extraction.
    
    Args:
        num_job_splits: Total number of job splits
        job_idx: Current job index
    """
    extractor = FrameExtractor()
    extractor.process_job(num_job_splits, job_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_job_splits", type=int, default=1)
    parser.add_argument("--job_idx", type=int, default=0)
    args = parser.parse_args()
    assert args.job_idx < args.num_job_splits
    assert args.num_job_splits >= 1
    main(args.num_job_splits, args.job_idx)