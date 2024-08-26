'''
Script to check repetition of inferred questions in PerceptionTest results.
'''

import json
import os

dir_path = "results/inference/PerceptionTest"
vids = {}
key_freqs = {}      # to check if questions have been repeated or not in inference

fnew = open("results/inference/PerceptionTest/perceptiontest_mcqa_llavanextvideo_responses_final.jsonl", 'w')

for i in range(1,5):
    filename = f"perceptiontest_mcqa_llavanextvideo_responses_p{i}.jsonl"
    f = open(os.path.join(dir_path, filename), 'r')
    for line in f.readlines():
        d = json.loads(line)
        vid = d['video']
        qn = d['question']
        area = d['area']
        reasoning = d['reasoning']

        if i not in vids:
            vids[i] = list()
        vids[i].append(vid)

        key = f"{vid}_{area}_{reasoning}_{qn}"
        if key not in key_freqs:
            key_freqs[key] = 0
        
        key_freqs[key] += 1
        # if question is not repeated earlier, only then write it to new file
        if key_freqs[key] == 1:
            fnew.write(json.dumps(d) + '\n')

    f.close()        

print("Length of p1: {}".format(len(vids[1])))
print("Length of p2: {}".format(len(vids[2])))
print("Length of p3: {}".format(len(vids[3])))
print("Length of p4: {}".format(len(vids[4])))
print("Total questions : {}".format(len(vids[1])+len(vids[2])+len(vids[3])+len(vids[4])))
      
for i in range(1,5):
    for j in range(1,5):
        if i==j:
            continue
        overlap = list(set(vids[i]).intersection(vids[j]))
        print("Overlap between p{} and p{}: {}".format(i,j,overlap))

count = 0
max_freq = 0
rep_qns_freq = {}

for key in key_freqs:
    if key_freqs[key] > 1:
        # print(key, key_freqs[key])
        # print('\n\n')

        if key_freqs[key] not in rep_qns_freq:
            rep_qns_freq[key_freqs[key]] = 0
        rep_qns_freq[key_freqs[key]] += 1

        count += 1
        max_freq = max(max_freq,  key_freqs[key])

print("Total repeated questions: {}".format(count))
print("Maximum repetition : ", max_freq)
print("Frequency of repeated questions: ", rep_qns_freq)

fnew.close()
