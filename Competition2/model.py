import torch.nn as nn
from OCR.modules.grid_mask import GridMask
from OCR.modules.transformation import TPS_SpatialTransformerNetwork
from OCR.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from OCR.modules.sequence_modeling import BidirectionalLSTM
from OCR.modules.prediction import Attention
import matplotlib.pyplot as plt
class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Grid' : opt.GridMask,'Feat': opt.FeatureExtraction,
                                'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        # if opt.Transformation == 'TPS':
        #     self.Transformation = TPS_SpatialTransformerNetwork(
        #         F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        # else:
        #     print('No Transformation module specified')

        self.Transformation = TPS_SpatialTransformerNetwork(F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)

        """ GridMask """
        if opt.GridMask == 'GridMask':
            self.Gridmask = GridMask(self.opt.d1, self.opt.d2, self.opt.rotate, self.opt.ratio, self.opt.mode, self.opt.prob)

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True, grid_mask = True):

        aa = 3
        #plt.imshow(input[10].squeeze().cpu().data.numpy())
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
            #plt.imshow(input[5].squeeze().cpu().data.numpy())

        """grid mask"""
        if self.stages['Grid'] == "GridMask":
            if (grid_mask):
                input = self.Gridmask(input)
                # plt.imshow(input[10].squeeze().cpu().data.numpy())
                # plt.imshow(input[0].squeeze().cpu().data.numpy(), cmap='rainbow', origin='lower')

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction