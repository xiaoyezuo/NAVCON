# Introducing NAVCON: A Large Scale Cognitively Inspired and Linguistically Grounded Corpus for Vision-Language Navigation

## NAVCON Language 
- NAVCON contains annotations of instructions taken from the following two [VLN datasets](https://github.com/jacobkrantz/VLN-CE): a) [R2R VLNCE](https://bringmeaspoon.org/): Room-to-Room Vision and Language Navigation in Continuous Environments and b) [RXR VLNCE](https://ai.google.com/research/rxr/): Room-Across-Room Vision and Language Navigation in Continuous Environments.
- We leverage the Train data splits from these two publicly available VLN datasets to extract 30,815 (19,996 from RXR and 10,819 from R2R) English language instructions to release 236,316 concept instantiations. These instantiations can be found [here]().
-- The dataset is a json with the following architecture:


## NAVCON Concept-video Clips

The [concept-video clips dataset](s3://navcondata/rxr_clips/) contains sequential image frames corresponding to each concept for 19074 RXR instructions. Specifically, the top-level folders inside rxr_clip are named after instruction ids in the RXR dataset. Inside each instruction's folder, each concept identified in the instruction has a subfoler of corresponding images in chronological order. For example, folder 000000 contains the concept-video clips for instruction 000000 and the subfoler 0 contains clips for the first concept identified in this instruction. 
