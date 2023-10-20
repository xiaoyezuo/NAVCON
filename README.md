# Introducing NAVCON: A Large Scale Cognitively Inspired and Linguistically Grounded Corpus for Vision-Language Navigation

## NAVCON Language 
- NAVCON contains annotations of instructions taken from the following two [VLN datasets](https://github.com/jacobkrantz/VLN-CE): a) [R2R VLNCE](https://bringmeaspoon.org/): Room-to-Room Vision and Language Navigation in Continuous Environments and b) [RXR VLNCE](https://ai.google.com/research/rxr/): Room-Across-Room Vision and Language Navigation in Continuous Environments.
- We leverage the Train data splits from these two publicly available VLN datasets to extract 30,815 (19,996 from RXR and 10,819 from R2R) English language instructions to release 236,316 concept instantiations. These instantiations can be found [here]().
  - The dataset is a json of the following structure:
```JSON
{
  "sentence": [
    "You will start by standing in front of a glass door and on your right is a doorway. Turn around and you will see a doorway to the washroom. Walk towards the doorway and inside the washroom. Once you're there, stand in between the sink and the bathtub and once you're there, you're done."
  ],
  "final_phrase": [
    [
      "standing in front of a glass door",
      "Turn around",
      "see a doorway to the washroom",
      "Walk towards the doorway and inside the washroom",
      "stand in between the sink and the bathtub"
    ]
  ],
  "final_concept": [
    [
      "situate",
      "change direction",
      "situate",
      "move",
      "situate"
    ]
  ],
  "meta_dict": [
    [
      {
        "phrase": [
          "standing in front of a glass door",
          "Turn around",
          "see a doorway to the washroom",
          "Walk towards the doorway and inside the washroom",
          "stand in between the sink and the bathtub"
        ],
        "start_idx": [
          18,
          84,
          109,
          140,
          209
        ],
        "stop_idx": [
          52,
          96,
          139,
          189,
          251
        ],
        "concept": [
          "situate",
          "change direction",
          "situate",
          "move",
          "situate"
        ],
        "rem_start_idx": [
          0,
          52,
          96,
          139,
          189,
          251
        ],
        "rem_stop_idx": [
          18,
          84,
          109,
          140,
          209,
          286
        ],
        "concept_words": [
          [
            "standing",
            "in",
            "front",
            "of",
            "a",
            "glass",
            "door"
          ],
          [
            "Turn",
            "around"
          ],
          [
            "see",
            "a",
            "doorway",
            "to",
            "the",
            "washroom."
          ],
          [
            "Walk",
            "towards",
            "the",
            "doorway",
            "and",
            "inside",
            "the",
            "washroom."
          ],
          [
            "stand",
            "in",
            "between",
            "the",
            "sink",
            "and",
            "the",
            "bathtub"
          ]
        ],
        "concept_idx": [
          [
            4,
            11
          ],
          [
            18,
            20
          ],
          [
            23,
            29
          ],
          [
            29,
            37
          ],
          [
            40,
            48
          ]
        ],
        "remaining_words": [
          [
            "You",
            "will",
            "start",
            "by"
          ],
          [
            "and",
            "on",
            "your",
            "right",
            "is",
            "a",
            "doorway."
          ],
          [
            "and",
            "you",
            "will"
          ],
          [
            "Once",
            "you're",
            "there,"
          ],
          [
            "and",
            "once",
            "you're",
            "there,",
            "you're",
            "done."
          ]
        ],
        "remaining_idx": [
          [
            0,
            4
          ],
          [
            11,
            18
          ],
          [
            20,
            23
          ],
          [
            37,
            40
          ],
          [
            48,
            54
          ]
        ]
      }
    ]
  ]
}
```


## NAVCON Concept-video Clips
