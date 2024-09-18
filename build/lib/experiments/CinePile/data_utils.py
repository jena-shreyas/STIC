import cv2
import shutil
from PIL import Image
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import subprocess

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images


# TEXT FORMATTING
def format_question_and_options(question, options):
    """
    Formats a question and a list of options into a single string with options labeled A, B, C, etc.

    Parameters:
    - question (str): The question to be formatted.
    - options (list of str): The options for the question.

    Returns:
    - str: The formatted question and options.
    """
    formatted_string = f"{question}\n"
    option_labels = [chr(ord('A') + i) for i in range(len(options))]  # Generate option labels dynamically

    for label, option in zip(option_labels, options):
        formatted_string += f"- {label}) {option}\n"

    return formatted_string


def print_qa(rows):
    """
    Prints formatted questions and answers from a dataset.

    Parameters:
    - rows (Iterable[Dict[str, Any]]): An iterable (e.g., list or Hugging Face dataset slice)
      where each element is a dictionary with keys 'question', 'answer_key', 'answer_key_position',
      and 'choices'. 'question' is a string representing the question text; 'answer_key' is the correct
      answer text; 'answer_key_position' is the index of the correct answer in 'choices';
      'choices' is a list of strings representing the answer options.

    Returns:
    - None: This function does not return any value but prints the questions and answers to the console.
    """
    count = 1
    ans_key_to_option = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    for row in rows:
        question, answer_key, answer_key_position, choices = row['question'], row['answer_key'], row['answer_key_position'], row['choices']

        question_choices = format_question_and_options(question, choices)
        print(f'Question {count}:')
        print(question_choices)

        print("\nAnswer Key:")
        print(f"{ans_key_to_option[answer_key_position]}) {answer_key}")

        count += 1
        print('-'*30)


# VIDEO PROCESSING
def download_video(video_url, filename, root):
    """
    Download and convert a video from a URL and save it to a specified directory.

    Parameters:
    - video_url (str): The URL of the video to be downloaded.
    - filename (str): The base name for the output file, without file extension.
    - root (str): The root directory where the 'yt_videos' folder will be created.

    Returns:
    - tuple: A tuple containing the video URL and a boolean. The boolean is True if the
      download and conversion was successful, and False otherwise.
    """

    dir_path=f"{root}/yt_videos"

    try:
        vid_prefix = os.path.join(dir_path, filename)
        full_command = [
            "yt-dlp",
            "-S",
            "height:224,ext:mp4:m4a",
            "--recode",
            "mp4",
            "-o",
            f"{vid_prefix}.mp4",
            video_url
        ]

        print(f'saving path: {vid_prefix}.mp4')

        result = subprocess.run(full_command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Downloaded: {vid_prefix}; {video_url}")
            return video_url, True
        else:
            print(f"Failed to download or convert {video_url}. Error: {result.stderr}")
            return video_url, False

    except Exception as e:
        print(f"Exception during download or conversion of {video_url}: {e}")
        return video_url, False
    

def find_scenes(video_path, threshold=30.0):
    """
    Detects important scenes in a video by analyzing changes between frames and identifying significant content changes that exceed a specified threshold.

    Parameters:
    video_path (str): The file path to the video file for which scenes are to be detected.
    threshold (float): The sensitivity threshold for detecting scene changes.

    Returns:
    list of tuples: A list where each tuple contains the start and end `FrameTimecodes` of a detected scene.
    """

    # Create a video manager object for the video.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Add ContentDetector algorithm (with a threshold).
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start the video manager and perform scene detection.
    video_manager.set_downscale_factor()
    video_manager.start()

    # Perform scene detection and return scene list.
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each scene is a tuple of (start, end) FrameTimecodes.
    return scene_manager.get_scene_list()


def save_frames_from_scenes(video_path, scenes, output_folder):
    """
    Extracts and saves the first frame from each detected scene in a video.

    Parameters:
    - video_path (str): The file path to the video from which frames are to be extracted.
    - scenes (list): A list of scene boundaries or metadata that specifies where each scene begins and ends.
    - output_folder (str): The directory path where the extracted frames should be saved.

    Returns:
    - None: The function saves the frames to the specified directory and does not return any value.
    """
    # Ensure output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize video manager for frame extraction.
    video_manager = VideoManager([video_path])
    video_manager.start()

    # Save the first frame of each detected scene.
    save_images(scenes, video_manager, num_images=1, output_dir=output_folder, image_name_template='$SCENE_NUMBER')

    video_manager.release()


def get_uniform_frames(video_path, num_frames=10):
    """
    This function takes a video file and returns a list of uniform frames from the video.
    :param video_path: str, path to the video file
    :param num_frames: int, number of uniform frames to return

    :return: list of frames
    """
    # check if path exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    uniform_frames = np.linspace(0, total_frames-1, num_frames, dtype=int)  # picking n frames uniformly from the video
    # random_frames = random.sample(range(total_frames), num_frames)
    frames = []
    for frame_num in uniform_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames

def save_frames_as_jpg(frames, frames_dir):
    # create directory if it doesn't exist
    pathlib.Path(frames_dir).mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{frames_dir}/{i}.jpg", frame)


def process_video(test_dataset, root_dir, base_folder_name='new_yt_videos', max_num_frames=10, visualize=False, ques_idx=None):
    if ques_idx is None:
        ques_idx = random.randint(0, len(test_dataset))
    data = test_dataset[ques_idx]
    clip_title, yt_link = data['yt_clip_title'], data['yt_clip_link']
    vid = yt_link.split('=')[-1]

    video_path = f"{root_dir}/{base_folder_name}/{vid}.mp4"
    frames_dir = f"{root_dir}/{base_folder_name}_frames/_{vid}"

    # download_video(yt_link, f"{data['movie_name']}_{yt_link.split('/')[-1]}", root=root_dir)
    scenes = find_scenes(video_path)
    save_frames_from_scenes(video_path, scenes, frames_dir)

    image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])[:-2]
    if len(image_files) < max_num_frames:
        shutil.rmtree(frames_dir)
        frames = get_uniform_frames(video_path, max_num_frames+2)[:-2]
        save_frames_as_jpg(frames, frames_dir)
        image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    img_file_paths = [os.path.join(frames_dir, image_file) for image_file in image_files]
    num_frames = max_num_frames if len(img_file_paths) > max_num_frames else len(img_file_paths)
    img_file_paths = [img_file_paths[i] for i in np.linspace(0, len(img_file_paths)-1, num_frames, dtype=int)]

    frames = list()
    for img_file_path in img_file_paths:
        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    frames = np.stack(frames)

    if visualize:
        num_cols = 5  # Number of columns in the grid
        num_rows = (num_frames + num_cols - 1) // num_cols  # Calculate the necessary number of rows to display all images

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        axs = axs.flatten() if num_frames > 1 else [axs]

        for i, img_path in enumerate(img_file_paths):
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].set_title(os.path.basename(img_path), fontsize=8)
            axs[i].axis('off')

        for ax in axs[len(img_file_paths):]:  # Hide unused axes
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return frames


# PROMPT GENERATION
vision_and_language_dependence_prompt = '''You will be provided with subtitles from a specific scene of a movie and a few frames from that scene. After going through the movie scene and seeing the frames, please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and nothing else.

**Output Format:**
    **Answer:** <Option_key>

**Subtitles:** \n{subs}\n\nQuestion: {question}

Note: Follow the output format strictly. Only answer with the option key (A, B, C, D, E) and nothing else.'''

def get_prompt(data):
    formatted_subs = data['subtitles']
    options = data['choices']
    formatted_question = format_question_and_options(data['question'], options)

    prompt = vision_and_language_dependence_prompt.format(subs=formatted_subs, question=formatted_question)
    return prompt



# EVALUATION
def normalize_string(input_string):
    """
    Extracts and returns the option number and option text from a given string.
    The option number is expected to be a single letter followed by an optional bracket and/or period.
    The option text is any text following the option number and its bracket/period.
    If the string does not contain an option number, the entire string is considered as the option text.
    """
    input_string = input_string.replace("*", "")
    match = re.search(r"Answer:\s*([A-E])\)?\.?\s*(.*)", input_string, re.IGNORECASE)
    if match:
        option_number = match.group(1).upper()  # Normalize option number to uppercase
        option_text = match.group(2).strip()
        # option_text = None if len(option_text) == 0 else option_text
        return option_number, option_text
    else:
        # If no option number is found after 'Answer:', consider it as no valid answer provided
        return None, input_string.strip()

def evaluate_semantic_similarity(response, answer_key_number, answer_key_text):
    """
    Evaluates whether the answer key and student response are semantically the same.
    Returns a score of 1 if they match, otherwise 0.
    """
    student_response_number, student_response_text = normalize_string(response)

    # Compare option numbers and option texts (if available) to determine a match
    if answer_key_number and student_response_number:
        # print(f"Answer Key Number: {answer_key_number} | Student Response Number : {student_response_number}")
        if answer_key_number == student_response_number:
            if answer_key_text and student_response_text:
                # If both strings have option texts, they must match as well
                return 1 if answer_key_text.lower() == student_response_text.lower() else 0
            # If only option numbers are provided or one string lacks option text, it's a match
            return 1
    elif answer_key_text.lower() == student_response_text.lower():
        # If no option numbers are present, but the option texts match, it's also considered a match
        return 1

    return 0

def eval_response(response, answer_key_number, answer_key_text):
    return evaluate_semantic_similarity(response, answer_key_number, answer_key_text)


if __name__ == "__main__":
    print(evaluate_semantic_similarity('Answer:A', 'A', 'This is the correct answer'))