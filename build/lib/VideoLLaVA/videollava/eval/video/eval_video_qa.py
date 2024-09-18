import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--api_type", default="chat", help="OpenAI API type.")                        
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--api_version", default="2020-05-03", help="OpenAI API")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # Set the OpenAI API key.
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    if args.api_type is not None:
        openai.api_type = args.api_type
    if args.api_version is not None:
        openai.api_version = args.api_version
    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                engine="gpt35tdec23",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ],
                temperature=0.1,
                max_tokens=40,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            # print(response_message)
            response_dict = ast.literal_eval(response_message)
            # print(response_dict)
            result_qa_pair = [response_dict, qa_set]
            # print(result_qa_pair)
            # exit(0)

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    print("Number of questions: ", len(new_pred_contents))
    # new_pred_contents = new_pred_contents[:10]

    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    '''
    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qtype = sample['type']
        qa_set = {"q": question, "a": answer, "pred": pred, "type": qtype}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    fails = 0
    cat_acc = {}
    for key, result in tqdm(combined_contents.items()):
        if key == args.output_json.split('/')[-1][:-5]:     # results_json
            continue
        try:
            # Computing score
            qtype = result[1]['type']
            if qtype not in cat_acc:
                cat_acc[qtype] = {
                    'total': 0,
                    'yes': 0,
                    'no': 0,
                    'score_sum': 0,
                    'accuracy': 0.0,
                    'average_score': 0.0
                }

            count += 1
            cat_acc[qtype]['total'] += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
            cat_acc[qtype]['score_sum'] += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
                cat_acc[qtype]['yes'] += 1
            elif "no" in pred.lower():
                no_count += 1
                cat_acc[qtype]['no'] += 1
            
            cat_acc[qtype]['accuracy'] = cat_acc[qtype]['yes'] / (cat_acc[qtype]['yes'] + cat_acc[qtype]['no'])
            cat_acc[qtype]['average_score'] = cat_acc[qtype]['score_sum'] / cat_acc[qtype]['total']
        except Exception as e:
            print("Key : ", key)
            print("Exception : ", e)
            print()
            fails += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Number of fails:", fails)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

    print(cat_acc)
    print("CATEGORY-WISE RESULTS -------------------")
    for key, value in cat_acc.items():
        print(f"Category: {key}")
        print(f"Total: {value['total']}")
        print(f"Yes: {value['yes']}")
        print(f"No: {value['no']}")
        print(f"Accuracy: {value['accuracy']}")
        print(f"Average Score: {value['average_score']}")
        print("\n")


if __name__ == "__main__":
    main()

