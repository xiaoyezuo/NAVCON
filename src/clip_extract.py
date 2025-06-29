import os
import numpy as np
import gzip, json, shutil
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class ClipExtractor:
    """
    A class for extracting video clips based on concept timestamps from RXR instruction data.
    """
    
    def __init__(self, project_dir="", rxr_dir=None):
        """
        Initialize the ClipExtractor with data paths and configuration.
        
        Args:
            project_dir (str): Path to project directory
            rxr_dir (str): Path to RXR data directory (if None, will be constructed from project_dir)
        """
        self.project_dir = project_dir
        self.rxr_dir = rxr_dir if rxr_dir else project_dir + "/rxr-data/"
        self.rxr_train_guide_path = self.rxr_dir + "rxr_train_guide.jsonl.gz"
        
        # Load instruction IDs and concept timestamps
        self.instruction_ids = np.load(project_dir + "instruction_id.npy")
        self.concept_timestamp = np.load(project_dir + "concept_timestamp.npy", allow_pickle=True).item()
    
    def closest_index(self, t, timestamps):
        """
        Find the index of the closest timestamp to input time.
        
        Args:
            t (float): Target timestamp
            timestamps (numpy.ndarray): Array of available timestamps
            
        Returns:
            int: Index of the closest timestamp
        """
        diff = abs(timestamps - t)
        return np.argmin(diff)

    def concept_time_to_concept_index(self, concept_time, timestamps, num_frames, end_offset):
        """
        Retrieve indices of start and end timestamps of all concepts in one instruction.
        TODO: increase end index by 5% offset
        
        Args:
            concept_time: Concept time data for the instruction
            timestamps (numpy.ndarray): Array of timestamps
            num_frames (int): Number of frames
            end_offset (float): End offset percentage
            
        Returns:
            numpy.ndarray: Array of concept indices with [concept_id, start_idx, end_idx]
        """
        concept_index = []
        max_idx = len(timestamps) - 1
        for entry in concept_time:
            concept_id = entry[0]
            start_idx = self.closest_index(entry[1], timestamps)
            end_idx = min(self.closest_index(entry[2], timestamps) + np.ceil(num_frames * end_offset), max_idx)
            concept_index.append(np.array([concept_id, start_idx, end_idx]))
        return np.array(concept_index)

    def extract_frames(self, instruction_id, concept_idx, vid_dir, out_dir):
        """
        Extract images corresponding to each concept for one instruction.
        
        Args:
            instruction_id: Instruction identifier
            concept_idx (numpy.ndarray): Concept indices with start/end frames
            vid_dir (str): Source video directory
            out_dir (str): Output directory for extracted clips
        """
        for i in range(concept_idx.shape[0]):
            dst_dir = os.path.join(out_dir, str(i) + "/")
            # Check if directory exists
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            start_idx = int(concept_idx[i][1])
            end_idx = int(concept_idx[i][2])
            for j in range(start_idx, end_idx):
                src_file = vid_dir + '{:0>6}'.format(j) + ".png"
                dst_file = dst_dir + '{:0>6}'.format(j) + ".png"
                if not os.path.exists(src_file):
                    continue
                shutil.copyfile(src_file, dst_file)

    def extract_all(self, end_offset=0.05, sample_rate=10):
        """
        Extract images corresponding to each concept for all instructions.
        
        Args:
            end_offset (float): End offset percentage (default: 0.05)
            sample_rate (int): Sample rate for timestamps (default: 10)
        """
        # Loop through all instruction ids
        for instruction_id in self.instruction_ids:
            vid_dir = "rxr_clips/" + '{:0>6}'.format(instruction_id) + "/"
            if not os.path.exists(vid_dir):
                continue
            out_dir = self.project_dir + "clips_v2/" + '{:0>6}'.format(instruction_id) + "/"
            # num_frames = len(os.listdir(vid_dir))
            # Skip existing clips 
            if os.path.isdir(out_dir) or (instruction_id not in self.concept_timestamp):
                continue
            # Extract images for each instruction

            concept_time = self.concept_timestamp[instruction_id]
            pose_trace = np.load(self.rxr_dir + "pose_traces/rxr_train/" + '{:0>6}'.format(instruction_id) + "_guide_pose_trace.npz")
            timestamps = pose_trace['time'][::sample_rate]
            num_frames = len(timestamps)
            concept_idx = self.concept_time_to_concept_index(concept_time, timestamps, num_frames, end_offset)
            self.extract_frames(instruction_id, concept_idx, vid_dir, out_dir)

    def process_instruction(self, instruction_id, end_offset=0.05, sample_rate=10):
        """
        Process a single instruction for clip extraction.
        
        Args:
            instruction_id: Instruction identifier to process
            end_offset (float): End offset percentage
            sample_rate (int): Sample rate for timestamps
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        vid_dir = "rxr_clips/" + '{:0>6}'.format(instruction_id) + "/"
        if not os.path.exists(vid_dir):
            return False
            
        out_dir = self.project_dir + "clips_v2/" + '{:0>6}'.format(instruction_id) + "/"
        
        # Skip existing clips 
        if os.path.isdir(out_dir) or (instruction_id not in self.concept_timestamp):
            return False
            
        # Extract images for the instruction
        concept_time = self.concept_timestamp[instruction_id]
        pose_trace = np.load(self.rxr_dir + "pose_traces/rxr_train/" + '{:0>6}'.format(instruction_id) + "_guide_pose_trace.npz")
        timestamps = pose_trace['time'][::sample_rate]
        num_frames = len(timestamps)
        concept_idx = self.concept_time_to_concept_index(concept_time, timestamps, num_frames, end_offset)
        self.extract_frames(instruction_id, concept_idx, vid_dir, out_dir)
        return True


def main():
    """
    Main function to execute clip extraction.
    """
    extractor = ClipExtractor()
    extractor.extract_all()


if __name__ == "__main__":
    main()