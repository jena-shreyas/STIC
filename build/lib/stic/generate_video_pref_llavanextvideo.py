import random
import numpy as np
import os
import argparse
import json
import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
from video_perturbations import perturb_video


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True, default='LanguageBind/Video-LLaVA-7B')
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--corrupt_dir', help='Directory containing corrupted video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--device_id", type=int, required=False, default=0)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--video_pert_prob", type=float, required=False, default=0.3)

    return parser.parse_args()

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_model_output(model, processor, video_path, qs, video_corruption=False):
    # define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": qs},
                {"type": "video"},
                ],
        },
    ]

    processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    container = av.open(video_path)

    # sample uniformly 16 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    num_frames = 16
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_length=1024)
    output_conv = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # parse output_conv so that the text after "ASSISTANT:" is returned
    return output_conv.split("ASSISTANT:")[1].strip()


def generate_pref(args):
    
    model_id = args.model_path
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        attn_implementation="flash_attention_2",
            ).to(0)
    
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    prompt_list = ["Illustrate the details of the video.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the video.",
                   "Outline the key elements captured in the video.",
                   "Detail the composition and subjects within the various video frames.",
                   "Interpret the scene shown in the video.",
                   "Identify and describe the main focal points in the video."]
    
    jitter_prompt = "Describe the video by mentioning the colors of various consituent objects and characters in the scene, exactly as they appear. Do not instead report the expected colors of the objects or characters if they seem to be contrary to the expected real-life colors for those objects."
    noise_extra_prompt = " However, do not mention the presence of any noise or distortion in the video."
    
    full_prompt = """Please provide a detailed description of the video, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the video and describe what they are doing.
    Describe the setting of the video. Is it indoors or outdoors? What kind of environment or location does it depict? 
    If applicable, are there any specific elements (such as lighting, weather, expressions) that contribute to the atmosphere depicted in the video? 
    Describe the dominant colors, including the colors of the objects present, and the overall composition. How do these elements affect the video's impact?
    If applicable, provide interpretations of what the video might represent or communicate."""
    
    hallu_prompt_list = [
                         "Modify the description by replacing objects or characters in the video with hypothetical objects or characters that could be part of the scene.",
                         "Enrich the video description by incorporating elements that, though absent, would seamlessly fit into the context of the video.",
                         "Enhance the video description by adding and describing additional everyday objects or activities that could be taking place just out of frame of the video being shown.",
                         "Augment the video description by inventing and incorporating details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]
    
    # NOTE : Try to keep the hallu_prompt list small, as variance in prompts would adversely vary the output descriptions.

    # 1. Horrible, gives bullet points.
    # 2. Modified version of 2 adds "hypothetical" objects very well.
    # 3. Pretty good, performs as expected.
    # 4. Performs well, adds subtle wrong details.
    # 6. Description too similar to preferred, not much to be reject-worthy
    
# "Suggest and detail practical items or people that could logically inhabit the video's setting."

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    video_filenames = os.listdir(args.video_dir)

    with open("/home/shreyasjena/BTP/datasets/WebVid/inf_left_vids.txt", 'r') as f:
        video_filenames = f.readlines()
        video_filenames = [vid.strip() + ".mp4" for vid in video_filenames]

    print("Total number of videos: ", len(video_filenames))
    
    # # Split the video files between the two GPUs
    # device_id = int(args.device_id)
    # num_videos = len(video_filenames) // 2
    # video_filenames = video_filenames[:num_videos] if device_id == 0 else video_filenames[num_videos:]
    
    video_corruptions = [
        "frame_noise",
        "frame_jitter"
    ]

    video_pert_prob = args.video_pert_prob
    print("Video Perturbation Probability: ", video_pert_prob)

    for filename in tqdm(video_filenames):
        video_path = os.path.join(args.video_dir, filename)
        video_name = filename.split(".")[0]
        video_corruption = False
        
        hallu_prompt = ""
        sample_corruption = ""
        prompt = random.choice(prompt_list)

        args.query = full_prompt
        try: 
            preferred_output = get_model_output(model, processor, video_path, args.query, video_corruption)

            # random sample a number between 0 and 1
            if random.random() > video_pert_prob:
                hallu_prompt = random.choice(hallu_prompt_list)
                args.query = hallu_prompt
                video_corruption = False
                corrupted_output = get_model_output(model, processor, video_path, args.query, video_corruption)
            else:
                video_corruption = True
                sample_corruption = random.choice(video_corruptions)

                # color jitter needs a separate prompt to force model to mention the changed color in the perturbed video
                if sample_corruption == "frame_jitter":
                    args.query = jitter_prompt
                elif sample_corruption == "frame_noise":
                    args.query = prompt + noise_extra_prompt
                # elif sample_corruption == "frame_noise":
                #     prompt = random.choice(prompt_list)
                
                video_corruption_dir = args.corrupt_dir
                corrupted_video_path = perturb_video(sample_corruption, video_path, video_corruption_dir)
                corrupted_output = get_model_output(model, processor, corrupted_video_path, args.query, video_corruption)

            d = {"video": video_name, 
                "video_corruption": video_corruption,
                "corruption_type": sample_corruption,
                "hallu_prompt": hallu_prompt,
                "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
                "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}
            
            answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
            
            with open(answers_file, "a") as f:
                f.write(json.dumps(d))
                f.write(",\n")

        except Exception as e:
            print(f"Error with video {video_path}: {e}")


if __name__ == "__main__":
    args = parse_args()
    generate_pref(args)