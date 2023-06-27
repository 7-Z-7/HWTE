import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tqdm
import csv
import pandas as pd

level3base_data_path='../../data/case_data/level3base/result_dict.pickle'
perf_data_path='../../data/case_data/perf/result_dict.pickle'
ap_stamp_path='../../data/ap_stamp.csv'
ap_vocab_path='ap_vocab.txt'

# load_path=perf_data_path
load_path=level3base_data_path

day_len=256
class_num=3

def ap_load(stamp_path,vocab_path):
    with open(vocab_path,'r') as f:
        data=f.readlines()
        vocab_dict={}
        for i,line in enumerate(data):
            vocab_dict[i]=line.replace('\n','')
        # print(data)
        # print(vocab_dict)
    data=pd.read_csv(stamp_path)
    stamp_dict = {}
    for one in data.values:
        # print(one)
        # print(one[5])
        # print(one[4])
        stamp_dict[str(one[5])]=one[4]
    # print('------')
    # print(stamp_dict[vocab_dict[200]])
    return stamp_dict,vocab_dict
    # print(data)

stamp_dict,vocab_dict=ap_load(ap_stamp_path,ap_vocab_path)


def format_route(inputs,durings,start_day=0,end_day=7):
    inputs=inputs[start_day:end_day]
    durings=durings[start_day:end_day]
    position_list=[]
    time_list=[]
    now_time=0
    for i,day in enumerate(inputs):
        during=durings[i]
        for j,one in enumerate(day):
            if one not in [101,102,0]:
                for time_index in range(during[j]):
                    time_list.append(now_time)
                    position_list.append(one)
                    now_time+=1
    return position_list,time_list

def split_none_position(position_list,time_list):
    new_position_list=[]
    new_time_list=[]
    guess_list=[]
    guess_time_list=[]
    none_list=[]
    none_time_list=[]
    now_station=106
    none_time=0
    break_time=24
    for i,position in enumerate(position_list):
        time=time_list[i]
        if position==106:
            if now_station==106:
                none_list.append(106)
                none_time_list.append(time)
            else:
                if none_time<break_time-1:
                    guess_list.append(now_station)
                    guess_time_list.append(time)
                    none_time+=1
                else:
                    guess_list.append(now_station)
                    guess_time_list.append(time)
                    none_list.extend([106]*break_time)
                    none_time_list.extend(guess_time_list[-break_time:])
                    del(guess_list[-break_time:])
                    del(guess_time_list[-break_time:])
                    now_station=106
                    none_time=0
        else:
            new_position_list.append(position)
            new_time_list.append(time)
            none_time=0
            now_station=position
    return new_position_list,new_time_list,guess_list,guess_time_list,none_list,none_time_list

def statistic(position_list,time_list):
    main_position=max(position_list, key=position_list.count)
    position_num=len(set(position_list))
    if 109 in position_list:
        flag_109=True
    else:
        flag_109=False
    rate=float(len(position_list))/(time_list[-1]-time_list[0])
    if (not flag_109) and rate>0.3 and position_num>10:
        flag=True
    else:
        flag=False
    return flag,main_position

def unin_order(position_list,time_list):
    position_list, time_list=zip(*sorted(zip(position_list,time_list),key=lambda x:x[1]))
    return list(position_list),list(time_list)

def None_Fill(position_list,time_list):
    fill_list=[]
    fill_time_list=[]
    if len(time_list)==0:
        return position_list,time_list
    for i in range(time_list[-1]):
        if i in time_list:
            pass
        else:
            fill_time_list.append(i)
            fill_list.append(None)
    position_list.extend(fill_list)
    time_list.extend(fill_time_list)
    position_list,time_list=unin_order(position_list,time_list)
    return position_list,time_list

def plot_route(new_position_list,new_time_list,guess_list,guess_time_list,title,path=None,day_range=None,guess_com=False):
    if guess_com:
        new_position_list.extend(guess_list)
        new_time_list.extend(new_time_list)
        new_position_list, new_time_list=unin_order(new_position_list, new_time_list)
    new_position_list, new_time_list = None_Fill(new_position_list, new_time_list)
    guess_list, guess_time_list = None_Fill(guess_list, guess_time_list)
    # print(len(new_time_list))
    # print(len(guess_time_list))
    if day_range:
        new_position_list = new_position_list[day_range[0] * 288:day_range[1] * 288]
        new_time_list = new_time_list[day_range[0] * 288:day_range[1] * 288]
        guess_list = guess_list[day_range[0] * 288:day_range[1] * 288]
        guess_time_list = guess_time_list[day_range[0] * 288:day_range[1] * 288]
    else:
        pass
    y_numbers=set(new_position_list)
    y_numbers=[one for one in y_numbers if one]
    # print(y_numbers)
    y_labels=[]
    for one in y_numbers:
        y_labels.append(stamp_dict[vocab_dict[one]])
    y_clear=True
    if y_clear:
        #two ap one floor
        # new_y_numbers_list=[y_number_list[0]]
        new_y_numbers_list=[]
        # new_y_label_list=[y_label_list[0]]
        new_y_label_list=[]
        # y_label_set={y_label_list[0][:8]:{y_number_list[0]:y_label_list[0]}}
        y_label_set={}
        for one_number,one_label in zip(y_numbers,y_labels):
            if one_label[:8] in y_label_set:
                y_label_set[one_label[:8]][one_number]=one_label
            else:
                y_label_set[one_label[:8]]={one_number:one_label}
                # y_label_set.append(one_label[:8])
                # new_y_label_list.append(one_label)
                # new_y_numbers_list.append(one_number)
        for one_floor in y_label_set:
            one_floor=y_label_set[one_floor]
            max_ap=one_floor[max(one_floor)]
            new_y_numbers_list.append(max(one_floor))
            new_y_label_list.append(max_ap)
            min_ap=one_floor[min(one_floor)]
            new_y_numbers_list.append(min(one_floor))
            new_y_label_list.append(min_ap)
        y_numbers=new_y_numbers_list
        y_labels=new_y_label_list
    plt.figure()
    plt.title(title)
    plt.plot(np.array(new_time_list) / 288, new_position_list, '.-', label='Route', markersize=4, color='b')
    if guess_com:
        pass
    else:
        plt.plot(np.array(guess_time_list) / 288, guess_list, '.-', markersize=4, color='b')
    plt.yticks(y_numbers, y_labels)
    plt.ylim(100,710)
    if path:
        plt.savefig(path,dpi=300,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def class_dict(load_path,start_day=10,end_day=17,pickle_name='test.pickle'):
    with open(load_path,'rb') as f:
        data=pickle.load(f)
        probabilities=data["probabilities"]
        labels=data["labels"]
        inputs=data["inputs"]
        durings=data["durings"]
        masks=data["masks"]
        samepalce_data={}
        for i,probability in tqdm.tqdm(enumerate(probabilities)):
            position_list, time_list=format_route(inputs[i],durings[i],start_day=start_day,end_day=end_day)
            new_position_list, new_time_list, guess_list, guess_time_list, none_list, none_time_list=split_none_position(position_list,time_list)
            if len(new_position_list)>20:
                flag, main_position=statistic(new_position_list,new_time_list)
                result=probability.index(max(probability))
                result_type=str(result)+str(labels[i][0])
                title='type(result+label):',format(result_type)
                if result_type in ['00','11'] and flag:
                    if main_position in samepalce_data:
                        if result_type in samepalce_data[main_position]:
                            samepalce_data[main_position][result_type].append((new_position_list,new_time_list,guess_list,guess_time_list))
                        else:
                            samepalce_data[main_position][result_type]=[(new_position_list,new_time_list,guess_list,guess_time_list)]
                    else:
                        samepalce_data[main_position]={result_type:[(new_position_list,new_time_list,guess_list,guess_time_list)]}
    with open(os.path.join(os.path.dirname(load_path),pickle_name), 'wb') as f:
        pickle.dump(samepalce_data,f)
    # if flag:
    #     plot_route(new_position_list,new_time_list,guess_list,guess_time_list,title)

        # print(new_position_list)
        # print(new_time_list)
        # print(guess_list)
        # print(guess_time_list)
        # print(none_list)
        # print(none_time_list)
        # print('..........................')
        # print(position_list)
        # print(time_list)
        # print(len(position_list))
        # print(len(time_list))
        # print(probability)
        # print(labels[i])
        # print(inputs[i])
        # print(durings[i])

def analyze(load_path,start_day=10,end_day=17,pickle_name='test.pickle'):
    with open(load_path,'rb') as f:
        data=pickle.load(f)
        probabilities=data["probabilities"]
        labels=data["labels"]
        inputs=data["inputs"]
        durings=data["durings"]
        masks=data["masks"]
        samepalce_data={}
        for i,probability in tqdm.tqdm(enumerate(probabilities)):
            position_list, time_list=format_route(inputs[i],durings[i],start_day=start_day,end_day=end_day)
            new_position_list, new_time_list, guess_list, guess_time_list, none_list, none_time_list=split_none_position(position_list,time_list)
            if len(new_position_list)>20:
                flag, main_position=statistic(new_position_list,new_time_list)
                result=probability.index(max(probability))
                result_type=str(result)+str(labels[i][0])
                title='type(result+label):',format(result_type)
                if result_type in ['00','11'] and flag:
                    if main_position in samepalce_data:
                        if result_type in samepalce_data[main_position]:
                            samepalce_data[main_position][result_type].append((new_position_list,new_time_list,guess_list,guess_time_list))
                        else:
                            samepalce_data[main_position][result_type]=[(new_position_list,new_time_list,guess_list,guess_time_list)]
                    else:
                        samepalce_data[main_position]={result_type:[(new_position_list,new_time_list,guess_list,guess_time_list)]}
    with open(os.path.join(os.path.dirname(load_path),pickle_name), 'wb') as f:
        pickle.dump(samepalce_data,f)
    # if flag:
    #     plot_route(new_position_list,new_time_list,guess_list,guess_time_list,title)

        # print(new_position_list)
        # print(new_time_list)
        # print(guess_list)
        # print(guess_time_list)
        # print(none_list)
        # print(none_time_list)
        # print('..........................')
        # print(position_list)
        # print(time_list)
        # print(len(position_list))
        # print(len(time_list))
        # print(probability)
        # print(labels[i])
        # print(inputs[i])
        # print(durings[i])

def plot_all(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    print(len(data))
    main_path = path.split('.pickle')[0]
    os.makedirs(main_path, exist_ok=True)
    for main_place in data:
        if len(data[main_place])>1:
            print(main_place)
            sub_path = os.path.join(main_path,str(main_place))
            os.makedirs(sub_path, exist_ok=True)
            for type in data[main_place]:
                for i,one in enumerate(data[main_place][type]):
                    print(1)
                    new_position_list, new_time_list, guess_list, guess_time_list=one
                    save_path=os.path.join(sub_path,str(main_place)+'_'+type+'_'+str(i)+'.png')
                    plot_route(new_position_list, new_time_list, guess_list, guess_time_list,'result/label:{}'.format(type),save_path)

def plot_all_format(path,day_range=None,main_place_list=None,show=False,guess_com=False):
    with open(path,'rb') as f:
        data=pickle.load(f)
    print(len(data))
    main_path = path.split('.pickle')[0]
    if day_range:
        os.makedirs(main_path+'_'+str(day_range[0])+'_'+str(day_range[1]), exist_ok=True)
    else:
        os.makedirs(main_path, exist_ok=True)
    for main_place in data:
        if len(data[main_place])>1:
            if main_place_list:
                if main_place in main_place_list:
                    pass
                else:
                    continue
            else:
                pass
            print(main_place)
            sub_path = os.path.join(main_path,str(main_place))
            os.makedirs(sub_path, exist_ok=True)
            for type in data[main_place]:
                for i,one in enumerate(data[main_place][type]):
                    print(1)
                    new_position_list, new_time_list, guess_list, guess_time_list=one
                    save_path=os.path.join(sub_path,str(main_place)+'_'+type+'_'+str(i)+'.png')
                    if show:
                        plot_route(new_position_list, new_time_list, guess_list, guess_time_list,'result/label:{}'.format(type),day_range=day_range,guess_com=guess_com)
                    else:
                        plot_route(new_position_list, new_time_list, guess_list, guess_time_list,'result/label:{}'.format(type),save_path,day_range=day_range,guess_com=guess_com)




if __name__=='__main__':
    # analyze(level3base_data_path,start_day=10,end_day=17,pickle_name='mainplace_00_11_dict.pickle')
    # class_dict(level3base_data_path,start_day=10,end_day=17,pickle_name='mainplace_00_11_dict.pickle')
    class_dict(level3base_data_path,start_day=0,end_day=255,pickle_name='mainplace_00_11_dict_0_255.pickle')
    # class_dict(perf_data_path,start_day=0,end_day=255,pickle_name='mainplace_00_11_dict_0_255.pickle')
    # plot_path=os.path.join(os.path.dirname(level3base_data_path),'mainplace_00_11_dict.pickle')
    plot_path=os.path.join(os.path.dirname(level3base_data_path),'mainplace_00_11_dict_0_255.pickle')
    # plot_path=os.path.join(os.path.dirname(perf_data_path),'mainplace_00_11_dict_0_255.pickle')
    # plot_path=os.path.join(os.path.dirname(perf_data_path),'mainplace_00_11_dict.pickle')
    # plot_all_format(plot_path,day_range=[0,7],main_place_list=[385,734],show=True)
    # plot_all_format(plot_path,day_range=[0,255],main_place_list=[385],show=True)
    # plot_all_format(plot_path,main_place_list=[385],show=True,guess_com=True)
    # plot_all_format(plot_path)
    # plot_all(plot_path)
