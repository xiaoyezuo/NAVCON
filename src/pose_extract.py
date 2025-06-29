import numpy as np
import json, gzip, sys, os
from pathlib import Path

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from utils import extract_camera_params
from clip_extract import ClipExtractor


class PoseExtractor:
    """
    A class for extracting pose data from RXR instruction data and concept timestamps.
    """
    
    def __init__(self, data_root=None, project_root=None, rxr_train_path=None, end_offset=1):
        """
        Initialize the PoseExtractor with data paths and configuration.
        
        Args:
            data_root (str): Root data directory (if None, will be auto-detected)
            project_root (str): Project root directory (if None, will be auto-detected)
            rxr_train_path (str): Path to RXR training data (if None, will use default)
            end_offset (int): End offset multiplier (default: 1)
        """
        self.data_root = data_root if data_root else DATA_ROOT
        self.project_root = project_root if project_root else PROJECT_ROOT
        self.rxr_train_path = rxr_train_path if rxr_train_path else rxr_train_guide_path
        self.end_offset = end_offset
        
        # Initialize paths for output files
        self.positions_path = Path(self.project_root) / "pose" / "positions_all.json"
        self.rotations_path = Path(self.project_root) / "pose" / "rotations_all.json"
        self.concept_pose_path = Path(self.project_root) / "pose" / f"concept_pose_{end_offset}.json"
        
        # Load concept timestamp data
        self.concept_timestamp = np.load("concept_timestamp.npy", allow_pickle=True).item()
        
        # Initialize data containers
        self.positions_all = {}
        self.rotations_all = {}
        self.concept_pose = {}
        
    def load_rxr_data(self):
        """
        Load RXR training data from JSONL file.
        
        Returns:
            list: List of RXR training data entries
        """
        data = []
        with gzip.open(self.rxr_train_path, 'rb') as f:
            for line in f:
                l = json.loads(line)
                data.append(l)
        return data
    
    def process_instruction_pose(self, instruction_id, concept_time, pose_trace, sample_rate=10):
        """
        Process pose data for a single instruction.
        
        Args:
            instruction_id (int): Instruction identifier
            concept_time: Concept time data for the instruction
            pose_trace: Pose trace data from NPZ file
            sample_rate (int): Sample rate for timestamps (default: 10)
            
        Returns:
            tuple: (instruction_pose, positions, rotations) for the instruction
        """
        timestamps = pose_trace['time'][::sample_rate]
        num_frames = len(timestamps)
        
        # Use ClipExtractor's method for concept index calculation
        clip_extractor = ClipExtractor()
        concept_idx = clip_extractor.concept_time_to_concept_index(
            concept_time, timestamps, num_frames, self.end_offset * 0.05
        )
        
        pose = pose_trace['extrinsic_matrix'][::sample_rate]
        instruction_pose = {}
        
        # Process each concept
        for i in range(concept_idx.shape[0]):
            p = {}
            start_idx = int(concept_idx[i][1])
            end_idx = int(concept_idx[i][2])
            concept_p = pose[start_idx:end_idx]
            positions, rotations = extract_camera_params(concept_p)
            p["positions"] = positions.tolist()
            p["rotations"] = rotations.tolist()
            instruction_pose[i] = p
        
        # Process full pose trace
        full_pose = pose_trace['extrinsic_matrix']
        positions, rotations = extract_camera_params(full_pose)
        positions = positions[::sample_rate]
        rotations = rotations[::sample_rate]
        
        return instruction_pose, positions, rotations
    
    def extract_all_poses(self, sample_rate=10):
        """
        Extract pose data for all instructions.
        
        Args:
            sample_rate (int): Sample rate for timestamps (default: 10)
        """
        data = self.load_rxr_data()
        
        for d in data:
            instruction_id = int(d['instruction_id'])
            if instruction_id not in self.concept_timestamp:
                continue
                
            print(f"Processing instruction {instruction_id}")
            concept_time = self.concept_timestamp[instruction_id]
            
            pose_trace = np.load(
                Path(self.data_root) / "rxr-data" / "pose_traces" / "rxr_train" / 
                f"{instruction_id:06d}_guide_pose_trace.npz"
            )
            
            instruction_pose, positions, rotations = self.process_instruction_pose(
                instruction_id, concept_time, pose_trace, sample_rate
            )
            
            # Store concept pose data (commented out in original)
            # self.concept_pose[instruction_id] = instruction_pose
            
            # Store full pose data
            self.positions_all[instruction_id] = positions.tolist()
            self.rotations_all[instruction_id] = rotations.tolist()
    
    def save_pose_data(self, save_concept_pose=False):
        """
        Save extracted pose data to JSON files.
        
        Args:
            save_concept_pose (bool): Whether to save concept pose data (default: False)
        """
        # Ensure pose directory exists
        pose_dir = Path(self.project_root) / "pose"
        pose_dir.mkdir(exist_ok=True)
        
        # Save concept pose data (if requested)
        if save_concept_pose:
            with open(self.concept_pose_path, 'w') as f:
                json.dump(self.concept_pose, f)
        
        # Save positions and rotations
        with open(self.positions_path, 'w') as f:
            json.dump(self.positions_all, f)
            
        with open(self.rotations_path, 'w') as f:
            json.dump(self.rotations_all, f)
    
    def process_single_instruction(self, instruction_id, sample_rate=10):
        """
        Process pose data for a single instruction.
        
        Args:
            instruction_id (int): Instruction identifier to process
            sample_rate (int): Sample rate for timestamps
            
        Returns:
            tuple: (instruction_pose, positions, rotations) or None if instruction not found
        """
        if instruction_id not in self.concept_timestamp:
            return None
            
        concept_time = self.concept_timestamp[instruction_id]
        pose_trace = np.load(
            Path(self.data_root) / "rxr-data" / "pose_traces" / "rxr_train" / 
            f"{instruction_id:06d}_guide_pose_trace.npz"
        )
        
        return self.process_instruction_pose(instruction_id, concept_time, pose_trace, sample_rate)
    
    def get_pose_data(self):
        """
        Get the extracted pose data.
        
        Returns:
            tuple: (positions_all, rotations_all, concept_pose)
        """
        return self.positions_all, self.rotations_all, self.concept_pose


def main():
    """
    Main function to execute pose extraction.
    """
    extractor = PoseExtractor()
    extractor.extract_all_poses()
    extractor.save_pose_data()


if __name__ == "__main__":
    main()