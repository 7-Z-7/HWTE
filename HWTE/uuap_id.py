import os
import pickle

path='../../data/part-00000-8c3f2ac5-ec84-40e2-95b7-45bae51e5d8b-c000.csv'
ex_path='../../data/ex_uuap2id.pickle'

id2uuap={}
uuap2id={}
with open(path, 'r') as f:
    data = f.read().split('\n')
for line in data:
  split_line=line.split(',')
  if len(split_line)>=2:
    id2uuap[int(split_line[0])]=split_line[1]
    uuap2id[split_line[1]]=int(split_line[0])

ex_id=1
if os.path.exists(ex_path):
    with open(ex_path,'rb') as f:
        ex_uuap2id,ex_id=pickle.load(f)
else:
    ex_uuap2id={}
    ex_id=1

def Id2Uup(id):
    return id2uuap[int(id)]

def Uuap2Id(uuap):
    uuap=str(uuap)
    global ex_uuap2id
    if uuap in uuap2id:
        return uuap2id[uuap],True
    elif uuap in ex_uuap2id:
        return ex_uuap2id[uuap],False
    else:
        global ex_id
        Id=ex_id
        ex_uuap2id[uuap]=Id
        with open(ex_path,'wb') as f:
            pickle.dump((ex_uuap2id,ex_id),f)
        ex_id = ex_id + 1
        return Id,False

