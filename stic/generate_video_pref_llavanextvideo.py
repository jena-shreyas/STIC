import random
import numpy as np
import os
import argparse
import json
import av
import torch
import pyarrow.parquet as pq
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
from .video_perturbations import perturb_video
import io

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True, default='LanguageBind/Video-LLaVA-7B')
    parser.add_argument('--parquet_dir', help='Directory containing parquet files.', required=True)
    parser.add_argument('--corrupt_dir', help='Directory containing corrupted video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--device_id", type=int, required=False, default=0)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--video_pert_prob", type=float, required=False, default=0.3)

    return parser.parse_args()

def read_video_pyav(video_bytes, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        video_bytes (bytes): Byte stream of the video.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    video_buffer = io.BytesIO(video_bytes)
    container = av.open(video_buffer)
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


def get_model_output(model, processor, video_bytes, qs, video_corruption=False):
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

    video_buffer = io.BytesIO(video_bytes)
    container = av.open(video_buffer)

    # sample uniformly 16 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    num_frames = 16
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(video_bytes, indices)
    inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_length=1024)
    output_conv = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
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
    
    full_prompt =   """
                        Please provide a detailed description of the entire video, considering both spatial and temporal elements as they evolve over time. Focus on the following:
                        Identify the main subjects (people, animals, objects) throughout the video and describe their actions as they unfold over different moments. How do their movements and interactions change from the beginning to the end of the video?
                        Based on the evolving sequence of events and visuals, what might the video represent or communicate as a whole? Consider how the video's progression contributes to its meaning, rather than focusing solely on a static frame.
                        What are the various events in the video and their resulting consequences in the video?
                        If applicable, specifically locate movements of objects as they evolve over time and how they interact in the video?
                    """
    
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
    
    used_parquet_files = set()
    parquet_files = [os.path.join(args.parquet_dir, f) for f in os.listdir(args.parquet_dir) if f.endswith('.parquet')]
    selected_parquet_files = random.sample(parquet_files, 468)

    video_corruptions = [
        "frame_noise",
        "frame_jitter"
    ]

    video_pert_prob = args.video_pert_prob
    print("Video Perturbation Probability: ", video_pert_prob)

    for parquet_file in tqdm(selected_parquet_files):
        if parquet_file in used_parquet_files:
            continue
        used_parquet_files.add(parquet_file)
        
        data = pq.read_table(parquet_file).to_pandas().to_dict()
        
        for video_batch in data['mp4']:
            video_bytes = data['mp4'][video_batch]
            video_corruption = False
            hallu_prompt = ""
            sample_corruption = ""
            prompt = random.choice(prompt_list)

            args.query = full_prompt
            preferred_output = get_model_output(model, processor, video_bytes, args.query, video_corruption)
            
            temp = "/scratch/svani/d"parquet_file.split('/')[-1].split('.parquet')[0] + "_" + str(video_batch)
            output_video_filename = parquet_file.split('.parquet')[0] + "_" + str(video_batch) + ".mp4"

            # random sample a number between 0 and 1
            if random.random() > video_pert_prob:
                hallu_prompt = random.choice(hallu_prompt_list)
                args.query = hallu_prompt
                video_corruption = False
                corrupted_output = get_model_output(model, processor, video_bytes, args.query, video_corruption)
                print(corrupted_output)
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
                corrupted_video_path = perturb_video(sample_corruption, video_bytes, args.corrupt_dir, video_corruption_dir, output_video_filename)
                corrupted_output = get_model_output(model, processor, corrupted_video_path, args.query, video_corruption)
                print(corrupted_output)
                
            d = {"video": parquet_file, 
                "video_corruption": video_corruption,
                "corruption_type": sample_corruption,
                "hallu_prompt": hallu_prompt,
                "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
                "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}
            
            answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
            
            with open(answers_file, "a") as f:
                f.write(json.dumps(d))
                f.write(",\n")


if __name__ == "__main__":
    args = parse_args()
    generate_pref(args)