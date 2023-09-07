# [setup]
import math
import os
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import pose

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../../data")
output_path = os.path.join(dir_path, "sample_output/")

save_index = 0

idx = 2000
file_path = "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

sample_ins, sample_scene = pose.load_annotation(file_path, idx)
sample_pose_trace = np.load("/home/zuoxy/ceph_old/rxr-data/pose_traces/rxr_train/"+'{:0>6}'.format(sample_ins)+"_guide_pose_trace.npz")
sample_pose = sample_pose_trace['extrinsic_matrix']

# positions, rotations = pose.extract_camera_params(sample_pose)
# angles, axes = pose.rotations_to_angles_axes(rotations)

# def save_img(data):
#     global save_index
#     img = Image.fromarray(data[0], "RGB")
#     img.save(output_path + str(save_index) + ".jpg")
#     save_index += 1

def save_img(rgb_obs):
    global save_index

    colors = []
    for row in rgb_obs:
        for rgba in row:
            colors.extend([rgba[0], rgba[1], rgba[2]])

    resolution_x = len(rgb_obs[0])
    resolution_y = len(rgb_obs)

    colors = bytes(colors)
    img = Image.frombytes("RGB", (resolution_x, resolution_y), colors)
    filepath = f"{output_path}/{save_index}.png"
    img.save(filepath)
    print(f"Saved image: {filepath}")
    save_index += 1

def get_obs(sim, show, save):
    # render sensor ouputs and optionally show them
    rgb_obs = sim.get_sensor_observations()["rgba_camera"]
    semantic_obs = sim.get_sensor_observations()["semantic_camera"]
    # if show:
    #     show_img((rgb_obs, semantic_obs), save)
    if save: 
        save_img(rgb_obs)

def place_agent_custom(sim, position, rotation):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = position
    angle, axis = pose.R_to_angle_axis(rotation)
    agent_state.rotation = quat_from_angle_axis(angle, axis)
    agent = sim.initialize_agent(0, agent_state) 
    return agent.scene_node.transformation_matrix()

def make_configuration(scene_file):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup rgb and semantic sensors
    camera_resolution = [1080, 960]
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

# This is wrapped such that it can be added to a unit test
def main(show_imgs=False, save_imgs=True):
    if save_imgs and not os.path.exists(output_path):
        os.mkdir(output_path)
        save_index = 0
    # [semantic id]

    # create the simulator and render flat shaded scene
    cfg = make_configuration(scene_file="NONE")
    sim = habitat_sim.Simulator(cfg)

    test_scenes = [
        "/home/zuoxy/ceph_old/data/mp3d/"+sample_scene+"/"+sample_scene+".glb",
    ]

    for scene in test_scenes:
        # reconfigure the simulator with a new scene asset
        cfg = make_configuration(scene_file=scene)
        sim.reconfigure(cfg)
        positions, rotations = pose.extract_camera_params(sample_pose)
        for i in range(positions.shape[0]):
            rotation = rotations[i]
            position = positions[i]
            agent_transform = place_agent_custom(sim, position, rotation)  # noqa: F841
            get_obs(sim, False, True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-images", dest="show_images", action="store_false")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(show_images=False, save_images=True)
    args = parser.parse_args()
    main(show_imgs=args.show_images, save_imgs=args.save_images)


# def place_agent(sim):
#     # place our agent in the scene
#     agent_state = habitat_sim.AgentState()
#     agent_state.position = [ 9.20822144, 1.58746791, -17.47610855 ]
#     agent_state.rotation = quat_from_angle_axis(
#         2.09065578771244, np.array([-0.31976206, -0.82970047, -0.45754706]))
#     agent = sim.initialize_agent(0, agent_state)
#     return agent.scene_node.transformation_matrix()

# def agent_init(sim, position, rotation):
#     #initialize agent in habitat
#     agent_state = habitat_sim.AgentState()
#     agent_state.position = position
#     angle, axis = rot_to_angle_axis(rotation)
#     agent_state.rotation = quat_from_angle_axis(angle, axis)
#     agent = sim.initialize_agent(0, agent_state) 
#     return agent

# def show_img(data, save):
#     # display rgb and semantic images side-by-side
#     fig = plt.figure(figsize=(12, 12))
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.axis("off")
#     ax1.imshow(data[0], interpolation="nearest")
#     ax2 = fig.add_subplot(1, 2, 2)
#     ax2.axis("off")
#     ax2.imshow(data[1], interpolation="nearest")
#     plt.axis("off")
#     plt.show(block=False)
#     if save:
#         global save_index
#         plt.savefig(
#             output_path + str(save_index) + ".jpg",
#             bbox_inches="tight",
#             pad_inches=0,
#             quality=50,
#         )
#         save_index += 1
#     plt.pause(1)