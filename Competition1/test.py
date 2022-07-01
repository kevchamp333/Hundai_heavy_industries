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
#  경로 설정
#########################################################################################
test_data_set = 'F:/Competition_No1/test_datasets'
saved_loc = 'F:/Competition_No1/code/saved_model/ckpt.pth' # best 모델 저장 경로
test_result_path = 'F:/Competition_No1/test_result/test'

epochs=1
batch_size=128
image_size=112 # 28, 56, 112, 224

# train_meanR, train_meanG, train_meanB = [0.24434853, 0.19431949, 0.22406203]
# train_stdR, train_stdG, train_stdB = [0.30652842, 0.25167993, 0.28387874]
# test_meanR, test_meanG, test_meanB = [0.58641684, 0.5074417, 0.49483952]
# test_stdR, test_stdG, test_stdB = [0.20517103, 0.19996276, 0.19438764]

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)), # 28, 56, 112, 224
        #transforms.Normalize([test_meanR, test_meanG, test_meanB], [test_stdR, test_stdG, test_stdB]),
    ])

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
# load model
model.load_state_dict(torch.load(saved_loc, map_location=device))
print(model)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0

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
    test_loss, test_acc = test(epoch)

#writer.close()