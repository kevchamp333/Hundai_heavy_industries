import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from OCR.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from OCR.dataset import AlignCollate, Batch_Balanced_Dataset, Make_custom_dataset
from OCR.model import Model
from OCR.test import validation
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(opt):

    ################################################################################
    # dataset preparation
    ################################################################################
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')

    #Train data
    AlignCollate_ = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    train_dataset = Batch_Balanced_Dataset(opt, AlignCollate_)
    log = open(f'./saved_models/log_dataset.txt', 'a')

    # Validation data
    valid_dataset = Make_custom_dataset(data_1=opt.valid_data_1, label_1=opt.valid_label_1, data_2=opt.valid_data_2, label_2=opt.valid_label_2, data_3=opt.valid_data_3, label_3=opt.valid_label_3, opt=opt)
    file_name_list = valid_dataset.get_file_name_list()
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, # batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_, pin_memory=True)

    ################################################################################
    # converter setting
    ################################################################################
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    ################################################################################
    #Model initialization
    ################################################################################
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    ################################################################################
    # CTC / Atten Loss setting
    ################################################################################
    """ setup loss """
    # if 'CTC' in opt.Prediction:
    #     if opt.baiduCTC:
    #         # need to install warpctc. see our guideline.
    #         from warpctc_pytorch import CTCLoss
    #         criterion = CTCLoss()
    #     else:
    #         criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    # else:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    best_accuracy = -1
    iteration = start_iter

    ################################################################################
    # Train
    ################################################################################
    while (True):

        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text, grid_mask = True)  #모델 예측
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size  # loss 계산
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1], grid_mask = True)  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)) # loss 계산

        model.zero_grad() # 기울기 초기화
        cost.backward() # 모든 가중치의 기울기 계산 (w.grad += dloss/dw)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step() # backprop
        loss_avg.add(cost)

        ################################################################################
        # Validation
        ################################################################################
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:  # To see training progress, we also conduct validation when 'iteration == 0'

            # for log
            with open(f'./saved_models/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy = validation(model, criterion, valid_loader, converter, opt, file_name_list, iteration)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/best_accuracy.pth')

                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1

def make_character_list(train_label_path, test_result_path) :

    total_label_list = []

    train_image_list = open(train_label_path, "r")
    train_image_list = train_image_list.readlines()

    test_image_list = open(test_result_path, "r")
    test_image_list = test_image_list.readlines()

    for i in train_image_list:
        if '.png' in i:
            label_name = i.split('\t')[1].split('\n')[0]
            [total_label_list.append(i) for i in label_name]
    for i in test_image_list:
        if '.png' in i:
            label_name = i.split('\t')[1].split('\n')[0]
            [total_label_list.append(i) for i in label_name]

    total_label_list = list(set(total_label_list))
    total_label_list = sorted(total_label_list)

    str = ""
    for i in total_label_list :
        str = str + i
    return str

if __name__ == '__main__':
    ################################################################################
    # 데이터 경로 & 하이퍼 파라미터 설정
    ################################################################################
    parser = argparse.ArgumentParser()
    # 01
    train_data_path_1 = 'F:/Competition_No2/datasets/train/01'
    valid_data_path_1 = 'F:/Competition_No2/datasets/test/01'
    train_label_path_1 = 'F:/Competition_No2/datasets/train/01/gt_train_01.txt'
    valid_label_path_1 = 'F:/Competition_No2/datasets/test/01/gt_test_01.txt'
    #parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ. ', help='character label')

    # 02
    train_data_path_2 = 'F:/Competition_No2/datasets/train/02'
    valid_data_path_2 = 'F:/Competition_No2/datasets/test/02'
    train_label_path_2 = 'F:/Competition_No2/datasets/train/02/gt_train_02.txt'
    valid_label_path_2 = 'F:/Competition_No2/datasets/test/02/gt_test_02.txt'
    #parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ. ', help='character label')

    # 03
    train_data_path_3 = 'F:/Competition_No2/datasets/train/03'
    valid_data_path_3 = 'F:/Competition_No2/datasets/test/03'
    train_label_path_3 = 'F:/Competition_No2/datasets/train/03/gt_train_03.txt'
    valid_label_path_3 = 'F:/Competition_No2/datasets/test/03/gt_test_03.txt'

    val_result_path = 'F:/Competition_No2/test_result/validation'
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ', help='character label')

    # data 1번 추가
    parser.add_argument('--train_data_1', default=train_data_path_1, help='path to training dataset | None')
    parser.add_argument('--train_label_1', default=train_label_path_1, help='path to training dataset | None')
    parser.add_argument('--valid_data_1', default= valid_data_path_1, help='path to validation dataset | None')
    parser.add_argument('--valid_label_1', default=valid_label_path_1, help='path to training dataset | None')

    # data 2번 추가
    parser.add_argument('--train_data_2', default=train_data_path_2, help='path to training dataset | None')
    parser.add_argument('--train_label_2', default=train_label_path_2, help='path to training dataset | None')
    parser.add_argument('--valid_data_2', default=valid_data_path_2, help='path to validation dataset | None')
    parser.add_argument('--valid_label_2', default=valid_label_path_2, help='path to training dataset | None')

    # data 3번 추가
    parser.add_argument('--train_data_3', default=train_data_path_3, help='path to training dataset | None')
    parser.add_argument('--train_label_3', default=train_label_path_3, help='path to training dataset | None')
    parser.add_argument('--valid_data_3', default=valid_data_path_3, help='path to validation dataset | None')
    parser.add_argument('--valid_label_3', default=valid_label_path_3, help='path to training dataset | None')
    #character = make_character_list(train_label_path, valid_label_path)     # character check

    parser.add_argument('--test_result_path', default=val_result_path, help='path to test result')
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    #parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

    """ Grid mask """
    parser.add_argument('--d1', type=int, default=5,  help='d1')   # 랜덤 사각형 최소값
    parser.add_argument('--d2', type=int, default=int(96/4), help='d2') # 랜덤 사각형 최대값
    parser.add_argument('--rotate', type=int, default=1, help='rotate the mask') # mask 회전 각도
    parser.add_argument('--ratio', type=float, default=0.6,help='ratio')  # 정사각형 중 마스크 이외의 비율
    parser.add_argument('--prob', type=float, default=0.7,help='max prob') # 랜덤 확률, 수치가 작을수록 본 이미지 출력,  수치가 클수록  랜덤 이미지 출력
    parser.add_argument('--mode', type=int, default=1,help='GridMask (1) or revised GridMask (0)')

    """ Data processing """
    #parser.add_argument('--select_data', type=str, default='MJ-ST', help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--select_data', type=str, default='/', help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    #parser.add_argument('--batch_ratio', type=str, default='0.5-0.5', help='assign ratio for each selected data in the batch')
    parser.add_argument('--batch_ratio', type=str, default='1',help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0', help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=96, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')

    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--GridMask', type=str, default='GridMask', help='GridMask|None')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')

    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.parse_args()
    opt = parser.parse_args()

    # if not opt.exp_name:
    #     opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    #     # print(opt.exp_name)
    # os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)

    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)