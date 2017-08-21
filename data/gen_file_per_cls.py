import codecs
import sys
import random
import numpy
import os

def shuffle_in_cls(lines):
    result = []
    curCls = "0"
    curIndex = 0
    curCount = 0
    for i in range(len(lines)):
        if lines[i].strip().split(" ")[1] == curCls:
            curCount = curCount + 1
        else:
            perm = numpy.random.permutation(curCount)
            for j in range(len(perm)):
                result.append(lines[curIndex + perm[j]])
            curCount = 1
            curIndex = i
            curCls = lines[i].strip().split(" ")[1]
    perm = numpy.random.permutation(curCount)
    for j in range(len(perm)):
        result.append(lines[curIndex + perm[j]])
    return result

def extract_source(lines, num_per_cls):
    result = []
    curCls = "0"
    curNum = 0
    temp = []
    for i in range(len(lines)):
        if curNum < num_per_cls:
            result.append(lines[i])
            curNum = curNum + 1
        elif lines[i].strip().split(" ")[1] == curCls:
            temp.append(lines[i])
        else:
            result.append(lines[i])
            curCls = lines[i].strip().split(" ")[1]
            curNum = 1
    return result
            
def extract_target(lines, num_per_cls, semi, labeled_target):
    result = []
    lresult = []
    curCls = "0"
    curNum = 0
    temp = []
    if semi:
        if num_per_cls > 0:
            for i in range(len(lines)):
                if curNum < num_per_cls:
                    result.append(lines[i])
                    curNum = curNum + 1
                elif curNum < num_per_cls + labeled_target:
                    lresult.append(lines[i])
                    curNum = curNum + 1
                elif lines[i].strip().split(" ")[1] == curCls:
                    continue
                else:
                    result.append(lines[i])
                    curCls = lines[i].strip().split(" ")[1]
                    curNum = 1
            return result, lresult
        else:
            for i in range(len(lines)):
                if curNum < labeled_target:
                    lresult.append(lines[i])
                    curNum = curNum + 1
                elif lines[i].strip().split(" ")[1] == curCls:
                    result.append(lines[i])
                else:
                    lresult.append(lines[i])
                    curNum = 1
                    curCls = lines[i].strip().split(" ")[1]
            return result, lresult
    else:
        if num_per_cls > 0:
            for i in range(len(lines)):
                if curNum < num_per_cls:
                    result.append(lines[i])
                    curNum = curNum + 1
                elif lines[i].strip().split(" ")[1] == curCls:
                    temp.append(lines[i])
                else:
                    result.append(lines[i])
                    curCls = lines[i].strip().split(" ")[1]
                    curNum = 1
            return result, []
        else:
            return lines, []

list_path = "./List.txt"

list_file = open(list_path, 'r')
lines = list_file.readlines()           

class_num = 21
nthreads = int(4)
train_num_per_class = 500
validation_num_per_class = 100

random.shuffle(lines)
labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]
dic = {}

for i in range(len(lines)):
    dic[i] = lines[i]

train = []
test = []
database = []

num_in_class = 0
for i in range(class_num):
    for index in dic.keys():
        if labels[index][i] == 1 and num_in_class < train_num_per_class:
            train.append(dic.pop(index))
            num_in_class = num_in_class + 1
            if num_in_class == train_num_per_class:
                print i
                num_in_class = 0
                break
num_in_class = 0
for i in range(class_num):
    for index in dic.keys():
        if labels[index][i] == 1 and num_in_class < validation_num_per_class:
            test.append(dic.pop(index))
            num_in_class = num_in_class + 1
            if num_in_class == validation_num_per_class:
                num_in_class = 0
                print i
                break
for index in dic.keys():
    database.append(dic.pop(index))

train_num = len(train)
print train_num
validation_num = len(test) / nthreads * nthreads
print validation_num
database_num = len(database) / nthreads * nthreads
print database_num

train_path = "./train_single_label.txt"
#test_path = "./parallel/test"
#database_path = "./parallel/database"

train_file = open(train_path, 'w')
#test_files = [open(test_path + str(i) + ".txt", "w") for i in range(nthreads)]
#database_files = [open(database_path + str(i) + ".txt", "w") for i in range(nthreads)]

random.shuffle(train)
for i in range(train_num):
    train_file.write(train[i])
#random.shuffle(test)
#for i in range(validation_num):
    #test_files[i / (validation_num / nthreads)].write(test[i])
#random.shuffle(database)
#for i in range(database_num):
    #database_files[i / (database_num / nthreads)].write(database[i])
