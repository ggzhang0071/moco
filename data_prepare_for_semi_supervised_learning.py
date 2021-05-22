from abc import abstractproperty
import os,random
from posixpath import splitext 


def split(full_list,shuffle=True,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

    
file_path="/git/moco/croped_images_part_with_classification"

save_folder="/git/moco/data_prepare_for_semi_supervised_learning"
choose_labels=[0,5,7]
label_mapping={}
start_index=0
for label in choose_labels:
    label_mapping[label]=start_index
    start_index+=1
image_name_label=[]
for label in choose_labels:
    image_name_list=os.listdir(os.path.join(file_path,str(label)))
    print(len(image_name_list))
    for image_name in image_name_list:
        image_name_label.append(os.path.join(str(label),image_name)+" "+str(label_mapping[label])+"\n")

data_train_list,data_list=split(image_name_label,ratio=0.7)
data_val_list, data_test_list=split(data_list,ratio=0.333)
print("train data num:{}, val data num:{}, test data num:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))


save_data_list=[data_train_list,data_val_list,data_test_list]
save_data_name=["train.txt","val.txt","test.txt"]
for i in range(len(save_data_list)):
    with open(os.path.join(save_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
        fid.writelines(save_data_list[i])
        print("{} is written".format(save_data_name[i]))



