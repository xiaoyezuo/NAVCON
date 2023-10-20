# Introducing NAVCON: A Large Scale Cognitively Inspired and Linguistically Grounded Corpus for Vision-Language Navigation

## NAVCON Language 

## NAVCON Concept-video Clips

The concept-video clips dataset(s3://navcondata/rxr_clips/) contains sequential image frames corresponding to each concept for 19074 RXR instructions. Specifically, the top-level folders inside rxr_clip are named after instruction ids in the RXR dataset. Inside each instruction's folder, each concept identified in the instruction has a subfoler of corresponding images in chronological order. For example, folder 000000 contains the concept-video clips for instruction 000000 and the subfoler 0 contains clips for the first concept identified in this instruction.    

The dataset is hosted on AWS S3 and can be [accessed](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html) through the S3 URL: s3://navcondata/rxr_clips/ 


