import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

from habitat_sim.utils.data import ImageExtractor

save_path = "/home/zuoxy/action_recognition/images"

# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

def save_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]
    rgb_img = img[:,:,:3]
    rgb_img = Image.fromarray(rgb_img, "RGB")
    rgb_img.save("/home/zuoxy/action_recognition/images/sample_rgb.jpg")
    
                
scene_filepath = "/home/zuoxy/RxR/visualizations/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('train')

# Index in to the extractor like a normal python list
sample = extractor[0]

# Or use slicing
samples = extractor[1:4]
for sample in samples:
    display_sample(sample)
    save_sample(sample)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()