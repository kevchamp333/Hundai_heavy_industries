import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pandas as pd
from OCR.utils import CTCLabelConverter, AttnLabelConverter, Averager
from OCR.dataset import  AlignCollate, Make_custom_dataset_test
from OCR.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, evaluation_loader, converter, opt, file_name_list, iteration):

    """ validation or evaluation """
    length_of_data = 0
    infer_time = []
    valid_loss_avg = Averager()

    GT_list = []
    prediction_list = []
    accuracy_list = []

    #총 데이터의 개수 = len(evaluation_loader) * evaluation_loader.batch_size
    #test data 실제 이미지, 레이블, 한번 돌때 배치 사이즈 만큼 꺼냄
    for i, (image_tensors, labels) in enumerate(evaluation_loader):

        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = int(round(time.time() * 1000))
        #start_time = datetime.now().microsecond
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred, grid_mask = False)
            forward_time = int(round(time.time() * 1000)) - start_time
            #forward_time = datetime.now().microsecond - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False, grid_mask = False)
            forward_time = int(round(time.time() * 1000)) - start_time
            #forward_time = datetime.now().microsecond - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time.append(forward_time)
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        # 배치 개수만큼
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):

            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if opt.sensitive and opt.data_filtering_off:
            #     pred = pred.lower()
            #     gt = gt.lower()
            #     alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            #     out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            #     pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            #     gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            GT_list.append(gt)
            prediction_list.append(pred)

            #####################################################################################
            # 1 - NED Evaluation
            #####################################################################################
            incorrect_count = 0
            if(len(gt) > len(pred)) :
                diff =len(gt) - len(pred)
                for i in range(diff) :
                    pred = pred + " "
            for idx, val in enumerate(gt):
                if (val != pred[idx]):
                    incorrect_count += 1
            accuracy = 1 - (incorrect_count / len(gt))
            accuracy = np.round(accuracy, 2)
            accuracy_list.append(accuracy)

    #####################################################################################
    #평가표 작성
    #####################################################################################
    columns = ['Image list', 'GT', 'Prediction', 'Accuracy', 'Inference Time']
    test_result = pd.DataFrame([file_name_list, GT_list, prediction_list, accuracy_list, infer_time])
    test_result = test_result.T
    test_result.columns = columns
    test_result['Accuracy'] = [str(i * 100) + '%' for i in test_result['Accuracy']]
    test_result['Inference Time'] = [str(i) + 'ms' for i in test_result['Inference Time']]

    average_accuracy_numeric = np.sum(accuracy_list) / len(test_result)
    average_accuracy = str(average_accuracy_numeric * 100) + '%'
    average_accuracy = pd.DataFrame([['', '', '', 'Average Accuracy', average_accuracy]], columns= columns)
    test_result_final = pd.concat([test_result, average_accuracy], axis = 0)

    inference_time_per_image = str(np.sum(infer_time) / len(evaluation_loader.dataset))+'ms'
    inference_time_per_image = pd.DataFrame([['', '', '', 'Inference Speed(ms)', inference_time_per_image]], columns=columns)
    test_result_final = pd.concat([test_result_final, inference_time_per_image], axis=0)
    test_result_final.to_csv(opt.test_result_path + '/test_result_' + str(iteration+1) + '.csv', index=False)

    print(test_result_final.to_string())

    return valid_loss_avg.val(), average_accuracy_numeric


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    #os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    #os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        test_dataset = Make_custom_dataset_test(data = opt.test_data, label = opt.test_label, opt=opt)
        file_name_list = test_dataset.get_file_name_list()

        evaluation_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        test_loss, current_accuracy = validation(model, criterion, evaluation_loader, converter, opt, file_name_list, 0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #########################################################################################
    #  경로 설정
    #########################################################################################
    # test_data_path = 'F:/Competition_No2/test_datasets/test/01'
    # test_label_path = 'F:/Competition_No2/test_datasets/test/01/gt_test_01.txt'
    # test_result_path = 'F:/Competition_No2/test_result/test/01'
    #
    # test_data_path = 'F:/Competition_No2/test_datasets/test/02'
    # test_label_path = 'F:/Competition_No2/test_datasets/test/02/gt_test_02.txt'
    # test_result_path = 'F:/Competition_No2/test_result/test/02'

    test_data_path = 'F:/Competition_No2/test_datasets/test/03'
    test_label_path = 'F:/Competition_No2/test_datasets/test/03/gt_test_03.txt'
    test_result_path = 'F:/Competition_No2/test_result/test/03'

    saved_model_path = 'F:/Competition_No2/code/OCR/saved_models/best_accuracy.pth'

    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ', help='character label')
    parser.add_argument('--test_data', default=test_data_path, help='path to evaluation dataset')
    parser.add_argument('--test_label', default=test_label_path, help='path to evaluation dataset')
    parser.add_argument('--test_result_path', default=test_result_path, help='path to test result')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--saved_model', default=saved_model_path, help="path to saved_model to evaluation")
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=96, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--sensitive', default=True, help='for sensitive character mode')

    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--GridMask', type=str, default='None', help='GridMask|None')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')

    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)