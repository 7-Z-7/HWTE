import matplotlib.pyplot as plt
import pickle
import csv
import os
import numpy as np

level3base_data_path='../../data/case_data/level3base/test_results.tsv'
perf_data_path='../../data/case_data/perf/test_results.tsv'

load_path=perf_data_path

save_path=os.path.join(os.path.dirname(load_path),'result_dict.pickle')

day_len=256
class_num=2
with open(load_path,'r') as f:
    data=f.read()
    data=data.replace('\n[',',[')
    # data=data.replace('\n',' ')
    data_one=data.split(',')

    # probabilities = prediction["probabilities"]
    # labels = prediction["labels"]
    # inputs = prediction["inputs"]
    # durings = prediction["durings"]
    # masks = prediction["masks"]
    examples={"probabilities":[],"labels":[],"inputs":[],"durings":[],"masks":[]}
    probabilities=[]
    labels=[]
    inputs=[]
    durings=[]
    masks=[]
    class_num_1=class_num-1
    example_num=0
    for i,one in enumerate(data_one):
        if i<class_num_1:
            probabilities.append(float(one))
        elif (i-class_num_1)%(day_len*3+class_num_1)==0:
            one=one.split('\n')
            probabilities.append(float(one[0]))
            labels.append(int(one[1]))
        elif 0<(i-class_num_1)%(day_len*3+class_num_1)<=day_len:
            input=one.replace('\n','')
            # input=input.replace('  ',',')
            input=','.join(input.split())
            input=input.replace('[,','[')
            input=input.replace(',]',']')
            input=eval(input)
            inputs.append(input)
        elif day_len<(i-class_num_1)%(day_len*3+class_num_1)<=day_len*2:
            input = one.replace('\n', '')
            # input = input.replace('  ', ',')
            input=','.join(input.split())
            input = input.replace('[,', '[')
            input = input.replace(',]', ']')
            input = eval(input)
            durings.append(input)
        elif day_len*2<(i-class_num_1)%(day_len*3+class_num_1)<=day_len*3-1:
            input = one.replace('\n', '')
            # input = input.replace('  ', ',')
            input=','.join(input.split())
            input = input.replace('[,', '[')
            input = input.replace(',]', ']')
            input = eval(input)
            masks.append(input)
        elif (i-class_num_1)%(day_len*3+class_num_1)==day_len*3:
            input = one.replace(']\n', '],')
            # print(input)
            input =input.split(',')
            # print(input)
            next_p=input[1].replace('\n','')
            input=input[0]
            input = input.replace('\n', '')
            # input = input.replace('  ', ',')
            input=','.join(input.split())
            input = input.replace('[,', '[')
            input = input.replace(',]', ']')
            input = eval(input)
            masks.append(input)
            examples["probabilities"].append(probabilities)
            examples["labels"].append(labels)
            examples["inputs"].append(inputs)
            examples["durings"].append(durings)
            examples["masks"].append(masks)
            example_num+=1
            probabilities = []
            labels = []
            inputs = []
            durings = []
            masks = []
            if len(next_p) > 0:
                print(next_p)
                probabilities.append(float(next_p))
        elif (i-class_num_1)%(day_len*3+class_num_1)>day_len*3:
            probabilities.append(float(one))
        else:
            print('error')

        if example_num%100==0:
            print(example_num)
            # print(examples["probabilities"])
            # print(examples["labels"])
            # print(np.shape(examples["inputs"]))
            # print(len(examples["durings"][0]))
            # print(len(examples["masks"][0]))
            # exit()
##examples={"probabilities":[],"labels":[],"inputs":[],"durings":[],"masks":[]}
print(len(examples["probabilities"]))
with open(save_path,'wb') as f:
    pickle.dump(examples,f)
