import os
import json
from data_utils import eval_response
from tqdm import tqdm

cat_scores = {}
question_scores = {}

num2key = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

f = open("results/inference/CinePile/cinepile_mcqa_llavanextvideo_responses.jsonl", 'r')
for line in tqdm(f.readlines()):
    data = json.loads(line)
    qid = data['id']
    vid = data['video']
    category = data['type']
    if category not in cat_scores:
        cat_scores[category] = {
            'correct': 0,
            'total': 0,
            'scores': 0.0
        }

    cat_scores[category]['total'] += 1
    response, answer_key_number, answer_key_text = data['pred'], data['correct_option'], data['answer']
    answer_key_number = num2key[answer_key_number]
    score = eval_response("Answer:"+response, answer_key_number, answer_key_text)
    question_scores[qid] = {
        'id': qid,
        'video': vid,
        'type': category,
        'pred': response,
        'correct_option': answer_key_number,
        'answer': answer_key_text,
        'score': score
    }
    if score == 1:
        cat_scores[category]['correct'] += 1
    # break

f.close()

with open("results/inference/CinePile/cinepile_mcqa_llavanextvideo_scores.json", 'w') as f:
    json.dump(question_scores, f)

print("*** CinePile MCQA LLaVA NextVideo Evaluation ***\n\n")
for cat in cat_scores:
    cat_scores[cat]['scores'] = cat_scores[cat]['correct']/cat_scores[cat]['total']
    print(f"{cat} ({cat_scores[cat]['total']}): {cat_scores[cat]['scores']}")

total_qns = sum([cat_scores[cat]['total'] for cat in cat_scores])
total_correct = sum([cat_scores[cat]['correct'] for cat in cat_scores])
total_score = total_correct/total_qns
print(f"\nTotal Score : {total_score}")
