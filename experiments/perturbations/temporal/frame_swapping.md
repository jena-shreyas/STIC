# Objective

To teach the model two things :

- (1) Detect events properly
- (2) Memorize the event order properly
- (3) (In case of too many events) figure out a way to help model :
    - Extract the relevant context from such events
    - Summarize it through compression for easier retention
    - Reuse step 2

- Since, `detecting events` and `memorizing order` simultaneously is HARD!!

    - Divide these two tasks into :
        - Subtask 1 : Teaching event detection
        - Subtask 2 : Teaching the order of detected events

    - SUBTASK 1
        
        - Tentative Prompt formats

            "Look at the video and answer if the following events/scenes occur/are mentioned or not:

            LIST OF SCENES

                For correct scenes, [**as many as present**]
                    - <thematicElements of scene>

                For wrong scenes, (3-4)
                    - <thematicElements of scene from some other video>

        - Preferred Response

            LIST OF SCENES

                - <thematicElements> -> yes for correct, no for wrong

        - Rejected Response 

            Sample any wrong assignment of yes/no from the above


    - SUBTASK 2

        - Prompt formats

            The following is a list of scene descriptions in the video (in random order). Each description has a thematic element, along with a detailed description of the events occurring in that scene, provided for easier localisation of the events. Order the scenes in the exact order as they occur in the video. Mention the order in the following format :

                <sceneOrder> <sceneThematicElement>

        - Preferred

            Correct Order + Thematic Element

        - Rejected

            Any wrong order + Thematic Element


# Formats

## Prompt Format



## Preferred Response Format

- Divide the description into scenes, like : 

    Scene X :
        <All activity lines from scene X, in order of the activities>

- Add filler lines in the scene description for new character intros (e.g., A new character Sara is introduced)

## Rejected Response Format

- TP (Temporal Perturbation) Type 1 [Scene, Simple]:

    - Reorder the scenes, keeping internal activity lines order same

- TP Type 2 [Activity, Fine-Grained]: [WARNING] (ONLY USABLE FOR HIGHLY ANNOTATED DATASETS, e.g., FineVideo)

    - Reorder the activity lines, keeping scene order same

# Steps