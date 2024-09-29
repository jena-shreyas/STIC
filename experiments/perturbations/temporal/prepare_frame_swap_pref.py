import os
import random
import numpy as np
from typing import List
from datasets import load_dataset
from tqdm import tqdm
import json
import io
import copy
import time


data_dir = "/home/shreyasj/BTP/datasets/FineVideo/HuggingFaceFV___finevideo/default/0.0.0/0d751f5e6563946f310c8a8e77a118e1f78af437"
# data_files = {"train": os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".arrow")}
total_dataset = load_dataset(data_dir, split="train", streaming=True)

NUM_SAMPLES = 4000
dataset = total_dataset.take(NUM_SAMPLES)

subtask1_prompt = '''
You are a video understanding expert. 
Based on the video clip provided, answer (in YES/NO) whether the following scenes/events occur, or are mentioned (by some character) in the video clip or not. 
For each scene/event, you are provided with a short description of the activities that occur in the scene.

Note that the scenes/events provided are not necessarily in the order they occur in the video. They may/may not even occur in the video.

Input Format:

  1) <event1thematicElements>

    <event1Description>

  2) <event2thematicElements>

    <event2Description>

  3) <event3thematicElements>

    <event3Description>

Expected Output Format:

    1) <event1thematicElements> -> <YES/NO>

    2) <event2thematicElements> -> <YES/NO>

    3) <event3thematicElements> -> <YES/NO>

LIST OF EVENTS

'''

subtask2_prompt = '''
You are a video understanding expert. 
You are provided with a video clip and a list of events/scenes that occurred, or are mentioned in the video clip.
For each scene/event, you are provided with a short description of the activities that occur in the scene.

Note that the scenes/events provided are not necessarily in the order they occur in the video.

Your task is to determine the order in which these events occur in the video clip.

Input Format:

  1) <event1thematicElements>

    <event1Description>

  2) <event2thematicElements>

    <event2Description>

  3) <event3thematicElements>

    <event3Description>

Assumption : Actual chronological order of events is : 3 -> 1 -> 2

Expected Output Format:

    3) <event3thematicElements>

    1) <event1thematicElements>

    2) <event2thematicElements>

LIST OF EVENTS

'''

# Event description consists of character introduction and description of each activity in a scene
# Create 2 Subtask 1 Type preferences and 2 Subtask 2 Type preferences

NUM_SUBTASK1_REJS = 2
NUM_SUBTASK2_REJS = 2
NUM_PREV_SAMPLE_SCENES = 3
MAX_SCENES = 6

output_dir = "outputs/FineVideo"
output_filename = os.path.join(output_dir, "frame_swapping_pref.jsonl")

if os.path.exists(output_filename):
    os.remove(output_filename)

f = open(output_filename, 'a')

def getSceneDescriptionsFromSampleData(sample_data, num_scenes=-1) -> List[str]:
    global MAX_SCENES
    scenes = sample_data['scenes']
    if num_scenes != -1:
        scenes = scenes[:num_scenes]

    sceneDescriptions = list()
    for i, scene in enumerate(scenes):
        activities = scene['activities']
        sceneThematicElement = scene['thematicElements']

        sceneCast = scene['cast']
        sceneActors = ', '.join(sceneCast)
        sceneDescription = sceneActors + ' ' + ('is' if len(sceneCast) == 1 else 'are') + ' the main ' + \
                            ('actor' if len(sceneCast) == 1 else 'actors') + ' in the scene.'
        description = []
        for activity in activities:
            description.append(activity['description'])

        sceneDescription = sceneDescription + ' ' + ' '.join(description)
        sceneDescriptions.append([sceneThematicElement, sceneDescription])

    order = list(range(len(sceneDescriptions)))
    if len(sceneDescriptions) > MAX_SCENES:     # sample only MAX_SCENES scenes from the video
        order = sorted(random.sample(order, MAX_SCENES))
        sceneDescriptions = [sceneDescriptions[i] for i in order]
        order = list(range(len(sceneDescriptions)))
        
    random.shuffle(order)
    sceneDescriptions = [sceneDescriptions[i] for i in order]       # ensure random order of scenes

    index2Order = zip(list(range(len(order))), order)
    sortedIndex2Order = sorted(index2Order, key=lambda x: x[1])
    correctOrder = [x[0] for x in sortedIndex2Order]
    
    return sceneDescriptions, correctOrder



# Store the preferences in a list of dicts, where each dict has :

#   - video_id
#   - video (in bytestring format)
#   - data (in standard Huggingface TRL format prompt, preferred, rejected)


sceneEntryTemplate = \
'''
    {}) {}

        {}

'''
sceneTask1AnswerTemplate = \
'''
    {}) {} -> {}

'''
sceneTask2AnswerTemplate = \
'''
    {}) {}

'''
prev_sample = None

video_save_dir = "/home/shreyasj/BTP/datasets/FineVideo/videos"
if not os.path.exists(video_save_dir):
    os.makedirs(video_save_dir)
elif os.listdir(video_save_dir):
    print("Video save directory is not empty. Deleting all videos ...")
    # Delete all files in the directory
    for file in os.listdir(video_save_dir):
        os.remove(os.path.join(video_save_dir, file))

# video_save_dir = "data"
print("Preparing preferences...")

start_time = time.time()

for sample in tqdm(dataset, total=NUM_SAMPLES):   
    data = sample['json'] 
    sample_data = data['content_metadata']

    task1Prompt = subtask1_prompt
    task2Prompt = subtask2_prompt

    sceneDescriptions, order = getSceneDescriptionsFromSampleData(sample_data)
    assert len(order) <= MAX_SCENES

    sceneLabelledDescriptions = list()
    if prev_sample is not None:
        prev_sample_data = prev_sample['json']['content_metadata']
        prev_sample_sceneDescriptions, _ = getSceneDescriptionsFromSampleData(prev_sample_data, num_scenes=NUM_PREV_SAMPLE_SCENES)
        sceneLabelledDescriptions = [[*sceneDesc, 'YES'] for sceneDesc in sceneDescriptions]
        prevSceneLabelledDescriptions = [[*sceneDesc, 'NO'] for sceneDesc in prev_sample_sceneDescriptions]
        sceneLabelledDescriptions = sceneLabelledDescriptions + prevSceneLabelledDescriptions
        random.shuffle(sceneLabelledDescriptions)

        sceneEntries = [sceneEntryTemplate.format(i+1, d[0], d[1]) for i, d in enumerate(sceneLabelledDescriptions)]
        task1Prompt = task1Prompt + '\n' + '\n'.join(sceneEntries)
    else:
        sceneLabelledDescriptions = [[*sceneDesc, 'YES'] for sceneDesc in sceneDescriptions]
        sceneEntries = [sceneEntryTemplate.format(i+1, d[0], d[1]) for i, d in enumerate(sceneLabelledDescriptions)]
        task1Prompt = task1Prompt + '\n' + '\n'.join(sceneEntries)
        
    # Prepare Task 1 Preferences and Rejections
    sceneTask1Prefs = [sceneTask1AnswerTemplate.format(i+1, d[0], d[2]) for i, d in enumerate(sceneLabelledDescriptions)]
    task1Pref = '\n'.join(sceneTask1Prefs)

    task1Rejs = list()
    sceneLabels = [d[2] for d in sceneLabelledDescriptions]
    if prev_sample is not None:
        randLabels = [np.random.permutation(sceneLabels).tolist() for _ in range(NUM_SUBTASK1_REJS)]
    else:
        num_desc = len(sceneLabelledDescriptions)
        randLabel = ['NO'] * (num_desc//2) + ['YES'] * (num_desc - num_desc//2)
        randLabels = list()
        for _ in range(NUM_SUBTASK1_REJS):
            random.shuffle(randLabel)
            tmp = copy.deepcopy(randLabel)
            randLabels.append(tmp)

    for randLabel in randLabels:
        wrongTask1Answers = [sceneTask1AnswerTemplate.format(i+1, d[0], l) for i, (d, l) in enumerate(zip(sceneLabelledDescriptions, randLabel))]
        task1Rej = '\n'.join(wrongTask1Answers)
        task1Rejs.append(task1Rej)

    # Prepare Task 2 Preferences and Rejections
    sceneTask2Entries = [sceneEntryTemplate.format(i+1, d[0], d[1]) for i, d in enumerate(sceneDescriptions)]
    task2Prompt = task2Prompt + '\n' + '\n'.join(sceneTask2Entries)

    sceneTask2Prefs = [sceneTask2AnswerTemplate.format(i+1, sceneDescriptions[i][0]) for i in order]
    task2Pref = '\n'.join(sceneTask2Prefs)

    task2Rejs = list()
    randOrders = [np.random.permutation(order).tolist() for _ in range(NUM_SUBTASK2_REJS)]

    for randOrder in randOrders:
        wrongTask2Orders = [sceneTask2AnswerTemplate.format(i+1, sceneDescriptions[i][0]) for i in randOrder]
        task2Rej = '\n'.join(wrongTask2Orders)
        task2Rejs.append(task2Rej)

    for i in range(NUM_SUBTASK1_REJS):
        data_dict = dict()
        data_dict['video_id'] = data['original_video_filename'].split('.')[0]
        data_dict['type'] = 'subtask1'
        data_dict['prompt'] = task1Prompt
        data_dict['chosen'] = [
            {
                "role": "user",
                "content": task1Prompt
            },
            {
                "role": "assistant",
                "content": task1Pref
            }
        ]

        data_dict['rejected'] = [
            {
                "role": "user",
                "content": task1Prompt
            },
            {
                "role": "assistant",
                "content": task1Rejs[i]
            }
        ]

        f.write(json.dumps(data_dict))
        f.write("\n")

    for i in range(NUM_SUBTASK2_REJS):
        data_dict = dict()
        data_dict['video_id'] = data['original_video_filename'].split('.')[0]
        data_dict['type'] = 'subtask2'
        data_dict['prompt'] = task2Prompt
        data_dict['chosen'] = [
            {
                "role": "user",
                "content": task2Prompt
            },
            {
                "role": "assistant",
                "content": task2Pref
            }
        ]

        data_dict['rejected'] = [
            {
                "role": "user",
                "content": task2Prompt
            },
            {
                "role": "assistant",
                "content": task2Rejs[i]
            }
        ]

        f.write(json.dumps(data_dict))
        f.write("\n")

    iobj = io.BytesIO(sample['mp4'])
    with open(os.path.join(video_save_dir, data['original_video_filename'].split('.')[0]+'.mp4'), 'wb') as bf:
        bf.write(iobj.getbuffer())

    prev_sample = sample

f.close()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")