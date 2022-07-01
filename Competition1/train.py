# =============================================================================
# 데이터 전처리
# =============================================================================
import os
os.chdir(r'C:\Users\bigcompetmgr014\Desktop\temp')

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from grid_mask import GridMask
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from get_competition_dataset import Competition_No1_dataset
from resnet import ResNet
from resnet import BasicBlock
from resnet import BottleNeck

os.chdir(r'F:\Competition_No1')
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# device = torch.device("cpu")

#########################################################################################
# 하이퍼 파라미터 & 경로 설정
#########################################################################################
train_set_nobg = 'F:/Competition_No1/nobg_datasets'
train_set_onlybg = 'F:/Competition_No1/onlybg_datasets'
test_data_set = 'F:/Competition_No1/datasets'
saved_loc = 'F:/Competition_No1/code/saved_model/ckpt.pth' # best 모델 저장 경로
test_result_path = 'F:/Competition_No1/test_result/validation'

epochs=200
batch_size=128
image_size=112 # 28, 56, 112, 224

train_meanR, train_meanG, train_meanB = [0.24434853, 0.19431949, 0.22406203]
train_stdR, train_stdG, train_stdB = [0.30652842, 0.25167993, 0.28387874]
test_meanR, test_meanG, test_meanB = [0.58641684, 0.5074417, 0.49483952]
test_stdR, test_stdG, test_stdB = [0.20517103, 0.19996276, 0.19438764]

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)), # 28, 56, 112, 224
        #transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)), # 28, 56, 112, 224
        #transforms.Normalize([test_meanR, test_meanG, test_meanB], [test_stdR, test_stdG, test_stdB]),
    ])

#train_set = Competition_No1_dataset(path='nobg_datasets', train=True, transform=train_transform)
train_set = Competition_No1_dataset(path= train_set_nobg, train=True, transform=train_transform, index_shuf = []) #nobg dataset
index_shuf = train_set.get_index_shuf()
train_set_onlybg = Competition_No1_dataset(path= train_set_onlybg, train=True, transform=train_transform, index_shuf = index_shuf) #only bg dataset
#sample = {'image': img, 'label': label, 'filename': img_path}
train_loader = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False)
train_loader_onlybg = DataLoader(dataset=train_set_onlybg,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        drop_last=False)
#test_set = Competition_No1_dataset(path='datasets', train=False, transform=test_transform)
test_set = Competition_No1_dataset(path= test_data_set, train=False, transform=test_transform, index_shuf = [])
test_loader = DataLoader(dataset=test_set,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False)

def resnet18():
    return ResNet(BasicBlock, image_size, [2,2,2,2])
# def resnet34():
#     return ResNet(BasicBlock, [3,4,6,3])
def resnet50():
    return ResNet(BottleNeck, image_size, [3,4,6,3])
# def resnet101():
#     return ResNet(BottleNeck, [3,4,23,3])
# def resnet152():
#     return ResNet(BottleNeck, [3,8,36,3])

model = resnet18().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0

#########################################################################################
# Training
#########################################################################################
def train(epoch):

    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0

    # ratio : 정사각형 중 마스크 이외의 비율,
    # prob : 랜덤 확률, 수치가 작을수록 본 이미지 출력,  수치가 클수록  랜덤 이미지 출력
    grid = GridMask(d1=2, d2=10, rotate=1, ratio=0.6, mode=1, prob=0.8)

    for batch_idx, (sample) in enumerate(zip(train_loader,  train_loader_onlybg)):

        # inputs = sample['image'].to(device)
        # targets = sample['label'].to(device)
        # names = sample['filename']

        inputs_nobg = sample[0]['image'].to(device)
        label_nobg = sample[0]['label'].to(device)
        inputs_onlybg = sample[1]['image'].to(device)
        label_onlybg = sample[1]['label'].to(device)
        #label_nobg[12]
        #inputs_nobg_grid_mask = grid(inputs_nobg)
        inputs_onlybg_grid_mask = grid(inputs_onlybg)
        #input_new = inputs_nobg_grid_mask + inputs_onlybg
        input_new = inputs_nobg + inputs_onlybg_grid_mask
        #input_new = inputs_nobg + inputs_onlybg

        #img_grid_masked_onlybg = transforms.ToPILImage()(input_new[12])
        #plt.imshow(img_grid_masked_onlybg)
        # plt.imshow(temp)
        # plt.show()
        
        optimizer.zero_grad()
        outputs = model(input_new)
        #_, pred = outputs.max(1)
        #print(outputs.squeeze(1))
        # print(targets)
        loss = criterion(outputs.squeeze(1), label_onlybg.float())
        print(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred = outputs.squeeze(1)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # _, predicted = outputs.max(1)
        total += label_onlybg.size(0)
        correct += y_pred.eq(label_onlybg).sum().item()
        # correct += predicted.eq(targets).sum().item()
        batch_count += 1

    print("Train Loss : {:.3f} | Train Acc: {:.3f}".format(train_loss / batch_count, 100.*correct/total))
    final_loss = train_loss / batch_count
    final_acc = 100.*correct / total

    return final_loss, final_acc

#########################################################################################
# test
#########################################################################################
def test(epoch):

    global best_acc
    global test_result
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (sample) in enumerate(test_loader):
            inputs = sample['image'].to(device)
            targets = sample['label'].to(device)
            
            # test 이미지 뽑기
            # img_test = transforms.ToPILImage(mode="RGB")(inputs[0])
            # plt.imshow(img_test)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets.float())
            # loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            y_pred = outputs.squeeze(1)
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            
            # _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += y_pred.eq(targets).sum().item()
            # correct += predicted.eq(targets).sum().item()
            batch_count += 1
    
    print("Test Loss : {:.3f} | Test Acc: {:.3f}".format(test_loss / batch_count, 100.*correct/total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        torch.save(model.state_dict(), saved_loc)
        best_acc = acc
        
        columns = ['Image list', 'GT', 'Prediction', 'Accuracy']
        test_file_name_list = test_loader.sampler.data_source.img_list
        GT_list = list(targets.cpu().numpy())
        prediction_list = list(y_pred.cpu().numpy())
        accuracy_list = list(y_pred.eq(targets).cpu().numpy())
        test_result = pd.DataFrame([test_file_name_list, GT_list, prediction_list, accuracy_list])
        test_result = test_result.T
        test_result.columns = columns
        calculated_times = (time.time()-start_time)*1000
        total_accuracy = pd.DataFrame([['', '', 'Total Accuracy', best_acc]], columns= columns)
        inference_speed = pd.DataFrame([['', '', 'Inference Speed (ms)', calculated_times/targets.size(0)]], columns= columns)

        test_result_final = pd.concat([test_result, total_accuracy, inference_speed], axis = 0)
        # if not os.path.isdir(os.path.join(os.path.join('./'), 'test_result')):
        #     os.mkdir(os.path.join(os.path.join('./'), 'test_result'))
        test_result_final.to_csv(test_result_path+'/test_result_grid_' + str(epoch) + '_' + str(image_size) + '.csv', index=False)

    final_loss = test_loss / batch_count

    return final_loss, acc


for epoch in tqdm(range(epochs)):

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    print("Train Accu : " + str(train_acc) + "Test Accu: " + str(test_acc) + "epoch:" + str(epoch))
    print("Train Loss : " + str(train_loss) + "Test Loss: " + str(test_loss) + "epoch:" + str(epoch))
    scheduler.step()

#writer.close()