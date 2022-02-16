import torch
import torch.nn as nn
# from pts3d import *
from ops import *
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
#from convolutional_rnn import Conv2dGRU
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Lm_encoder(nn.Module):
    def _init_(self):
        super(Lm_encoder, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )

    def forward(self, example_landmark):
        example_landmark_f = self.lmark_encoder(example_landmark)
        return example_landmark_f

class Ct_encoder(nn.Module):
    def __init__(self):
        super(Ct_encoder, self).__init__()
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )

    def forward(self, audio):

        feature = self.audio_eocder(audio)
        feature = feature.view(feature.size(0),-1)
        x = self.audio_eocder_fc(feature)

        return x


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()

        self.emotion_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),

            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),

            conv2d(128,256,3,1,1),

            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]

            conv2d(256,512,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 512, 1, 6]

            )
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(512 *6,2048),
            nn.ReLU(True),
            nn.Linear(2048,128),
            nn.ReLU(True),

            )

        self.last_fc = nn.Linear(128,8)

        self.re_id = nn.Sequential(
            conv2d(512,1024,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)), #[1, 1024, 1, 3]
            conv2d(1024,1024,3,1,1),

            conv2d(1024,2048,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 2048, 1, 1]


            )
        self.re_id_fc = nn.Sequential(

            nn.Linear(2048,512),
            nn.ReLU(True),
            nn.Linear(512,128),
            nn.ReLU(True),
            )


    def forward(self, mfcc):
       # mfcc= torch.unsqueeze(mfcc, 1)
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_eocder(mfcc)

   #     id_feature = feature.detach()

        feature = feature.view(feature.size(0),-1)
        x = self.emotion_eocder_fc(feature)


  #      remove_feature = self.re_id(id_feature)
  #      remove_feature = remove_feature.view(remove_feature.size(0),-1)
  #      y = self.re_id_fc(remove_feature)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(384, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=(4,2), stride=2, padding=1, bias=True),#8,6
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,12
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(2,1), padding=(3,1), bias=True),#28,12
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),#28,12

                nn.Tanh(),
                )

    def forward(self, content,emotion):
        features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        x = 90*self.decon(features) #[1, 1,28, 12]


        return x

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()



        self.last_fc = nn.Linear(128,8)

    def forward(self, feature):
       # mfcc= torch.unsqueeze(mfcc, 1)

        x = self.last_fc(feature)

        return x


'''
class RemoveID(nn.Module):
    def __init__(self):
        super(RemoveID, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(128, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=(4,2), stride=2, padding=1, bias=True),#8,6
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,12
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(2,1), padding=(3,1), bias=True),#28,12
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),#28,12

                nn.Tanh(),
                )

    def forward(self, content,emotion):
        features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        x = 90*self.decon(features) #[1, 1,28, 12]


        return x
'''

class AutoEncoder2x(nn.Module):
    def __init__(self,config):
        super(AutoEncoder2x, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = Decoder()
        self.classify = Classify()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.triplet_weight = config.triplet_weight

        self.use_triplet = config.use_triplet

        self.labels_name = ['label1','label2']
        self.inputs_name = ['input11',  'input12', 'input21', 'input32']
        self.targets_name = ['target11', 'target22',"target12","target21"]

        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.classify.parameters()), config.lr,betas=(config.beta1, config.beta2))



    def cross(self, x1, x2, x3, x4):
        c1 = self.con_encoder(x1)
        e1 = self.emo_encoder(x2)
        c2 = self.con_encoder(x4)
        e2 = self.emo_encoder(x3)

        out1 = self.decoder(c1,e1)
        out2 = self.decoder(c1,e2)
        out3 = self.decoder(c2,e1)
        out4 = self.decoder(c2,e2)

        return out1, out2, out3, out4

    def transfer(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.static_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2], dim=1))

        return out12

    def cross_with_triplet(self, x1, x2, x12, x21):
        c1 = self.con_encoder(x1)
        e1 = self.emo_encoder(x1)
        c2 = self.con_encoder(x2)
        e2 = self.emo_encoder(x2)

        out1 = self.decoder(c1,e1)
        out2 = self.decoder(c2,e2)
        out12 = self.decoder(c1,e2)
        out21 = self.decoder(c2,e1)


        c12 = self.con_encoder(x12)
        e12 = self.emo_encoder(x12)
        c21 = self.con_encoder(x21)
        e21 = self.emo_encoder(x21)

        outputs = [out1, out2, out12, out21]
        contentvecs = [c1.reshape(c1.shape[0], -1),
                      c2.reshape(c2.shape[0], -1),
                      c12.reshape(c12.shape[0], -1),
                      c21.reshape(c21.shape[0], -1)]
        emotionvecs = [e1.reshape(e1.shape[0], -1),
                      e2.reshape(e2.shape[0], -1),
                      e21.reshape(e21.shape[0], -1),
                      e12.reshape(e12.shape[0], -1)]

        return outputs, contentvecs, emotionvecs

    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self, data):



        labels = [data[name] for name in self.labels_name]
        inputs = [data[name] for name in self.inputs_name]
        targets = [data[name] for name in self.targets_name]

        losses = {}
        acces = {}

        if self.use_triplet:
            outputs, contentvecs, emotionvecs = self.cross_with_triplet(*inputs)
            losses['c_tpl1'] = self.triplet_weight * self.tripletloss(contentvecs[2], contentvecs[0], contentvecs[1])
            losses['c_tpl2'] = self.triplet_weight * self.tripletloss(contentvecs[3], contentvecs[1], contentvecs[0])
            losses['e_tpl1'] = self.triplet_weight * self.tripletloss(emotionvecs[2], emotionvecs[0], emotionvecs[1])
            losses['e_tpl2'] = self.triplet_weight * self.tripletloss(emotionvecs[3], emotionvecs[1], emotionvecs[0])
        else:
            outputs = self.cross(inputs[0], inputs[1], inputs[2], inputs[3])

        for i, target in enumerate(targets):
            losses['rec' + self.targets_name[i][6:]] = self.l1loss(outputs[i], target)

        c1 = self.con_encoder(inputs[0])

        c2 = self.con_encoder(inputs[3])

        e1 = self.emo_encoder(inputs[1])

        e2 = self.emo_encoder(inputs[2])

        losses['con_feature'] = self.l1loss(c1, c2)


        label1 = labels[0]
        label1=torch.squeeze(label1)
        label2 = labels[1]
        label2=torch.squeeze(label2)

        fake1 = self.classify(e1)
        fake2 = self.classify(e2)

        losses['cla_1'] = self.CroEn_loss(fake1,label1)
        losses['cla_2'] = self.CroEn_loss(fake2,label2)

        acces['acc_1'] = self.compute_acc(label1,fake1)
        acces['acc_2'] = self.compute_acc(label2,fake2)

        outputs_dict = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output3": outputs[2],
            "output4": outputs[3],
        }
        return outputs_dict, losses , acces

    def forward(self, x):
        c = self.con_encoder(x)
        e = self.emo_encoder(x[:, :-2, :])

        d = torch.cat([c, e], dim=1)
        d = self.decoder(d)
        return d

    def update_network(self, loss_dcit):

        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data):

        self.classify.train()
        self.decoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        outputs, losses, acces = self.process(data)

        self.update_network(losses)

        return outputs, losses, acces

    def val_func(self, data):
        self.classify.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()

        with torch.no_grad():
            outputs, losses, acces = self.process(data)

        return outputs, losses, acces

    def save_fig(self,data,outputs,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']

        a=['output1','output2','output3','output4']
        b=['target11','target22','target12','target21']

        for j in range(len(a)):
            output = outputs[a[j]]
            target = data[b[j]]

            for i in range(output.size(0)):
                g = target[i,:,:,:].squeeze()
                g = g.cpu().numpy()
           # plt.figure()
                ax = sns.heatmap(g, vmin=-100, vmax=100,cmap='rainbow')      #frames

                filepath = os.path.join(save_path,a[j])
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                plt.savefig(os.path.join(filepath,'real_'+str(i)+'.png'))
                plt.close()
      #      plt.show()
                o = output[i,:,:,:].squeeze()
                o = o.cpu().detach().numpy()
      #      plt.figure()
                ax = sns.heatmap(o, vmin=-100, vmax=100,cmap='rainbow')      #frames

                plt.savefig(os.path.join(filepath,'fake_'+str(i)+'.png'))
                plt.close()
      #      plt.show()


class AutoEncoder3x(nn.Module):
    def __init__(self,config):
        super(AutoEncoder3x, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = Decoder()
        self.classify = Classify()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.triplet_weight = config.triplet_weight

        self.use_triplet = config.use_triplet

        self.labels_name = ['label1','label2']
        self.inputs_name = ['inputi1',  'inputi2', 'inputi3', 'inputi4', "inputt1", "inputt2", "inputt3"]
        self.targets_name = ['targeto1', 'targeto2',"targeto3","targeto4"]

        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.classify.parameters()), config.lr,betas=(config.beta1, config.beta2))



    def cross(self, x1, x2, x3, x4):
        c1 = self.con_encoder(x1)
        e1,_ = self.emo_encoder(x2)
        c2 = self.con_encoder(x4)
        e2,_ = self.emo_encoder(x3)

        out1 = self.decoder(c1,e1)
        out2 = self.decoder(c1,e2)
        out3 = self.decoder(c2,e1)
        out4 = self.decoder(c2,e2)

        return out1, out2, out3, out4

    def transfer(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.static_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2], dim=1))

        return out12

    def cross_with_triplet(self, x1, x2, x3):

        _,e1 = self.emo_encoder(x1)

        _,e2 = self.emo_encoder(x2)

        _,e3 = self.emo_encoder(x2)





        emotionvecs = [e1.reshape(e1.shape[0], -1),
                      e2.reshape(e2.shape[0], -1),
                      e3.reshape(e3.shape[0], -1)]

        return  emotionvecs

    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self, data):



        labels = [data[name] for name in self.labels_name]
        inputs = [data[name] for name in self.inputs_name]
        targets = [data[name] for name in self.targets_name]

        losses = {}
        acces = {}

        if self.use_triplet:
            outputs, contentvecs, emotionvecs = self.cross_with_triplet(*inputs)
            losses['c_tpl1'] = self.triplet_weight * self.tripletloss(contentvecs[2], contentvecs[0], contentvecs[1])
            losses['c_tpl2'] = self.triplet_weight * self.tripletloss(contentvecs[3], contentvecs[1], contentvecs[0])
            losses['e_tpl1'] = self.triplet_weight * self.tripletloss(emotionvecs[2], emotionvecs[0], emotionvecs[1])
            losses['e_tpl2'] = self.triplet_weight * self.tripletloss(emotionvecs[3], emotionvecs[1], emotionvecs[0])
        else:
            outputs = self.cross(inputs[0], inputs[1], inputs[2], inputs[3])

        for i, target in enumerate(targets):
            losses['rec' + self.targets_name[i][6:]] = self.l1loss(outputs[i], target)

        emotionvecs = self.cross_with_triplet(inputs[4], inputs[5], inputs[6])
        losses['tpl'] = self.triplet_weight * self.tripletloss(emotionvecs[0], emotionvecs[1], emotionvecs[2])

        c1 = self.con_encoder(inputs[0])

        c2 = self.con_encoder(inputs[3])

        _,e1 = self.emo_encoder(inputs[4])

        _,e2 = self.emo_encoder(inputs[6])

        losses['con_feature'] = self.l1loss(c1, c2)


        label1 = labels[0]
        label1=torch.squeeze(label1)
        label2 = labels[1]
        label2=torch.squeeze(label2)

        fake1 = self.classify(e1)
        fake2 = self.classify(e2)

        losses['cla_1'] = self.CroEn_loss(fake1,label1)
        losses['cla_2'] = self.CroEn_loss(fake2,label2)

        acces['acc_1'] = self.compute_acc(label1,fake1)
        acces['acc_2'] = self.compute_acc(label2,fake2)

        outputs_dict = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output3": outputs[2],
            "output4": outputs[3],
        }
        return outputs_dict, losses , acces

    def forward(self, x):
        c = self.con_encoder(x)
        e = self.emo_encoder(x[:, :-2, :])

        d = torch.cat([c, e], dim=1)
        d = self.decoder(d)
        return d

    def update_network(self, loss_dcit):

        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data):

        self.classify.train()
        self.decoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        outputs, losses, acces = self.process(data)

        self.update_network(losses)

        return outputs, losses, acces

    def val_func(self, data):
        self.classify.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()

        with torch.no_grad():
            outputs, losses, acces = self.process(data)

        return outputs, losses, acces

    def save_fig(self,data,outputs,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']

        a=['output1','output2','output3','output4']
        b=['targeto1','targeto2','targeto2','targeto4']

        for j in range(len(a)):
            output = outputs[a[j]]
            target = data[b[j]]

            for i in range(output.size(0)):
                g = target[i,:,:,:].squeeze()
                g = g.cpu().numpy()
           # plt.figure()
                ax = sns.heatmap(g, vmin=-100, vmax=100,cmap='rainbow')      #frames

                filepath = os.path.join(save_path,a[j])
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                plt.savefig(os.path.join(filepath,'real_'+str(i)+'.png'))
                plt.close()
      #      plt.show()
                o = output[i,:,:,:].squeeze()
                o = o.cpu().detach().numpy()
      #      plt.figure()
                ax = sns.heatmap(o, vmin=-100, vmax=100,cmap='rainbow')      #frames

                plt.savefig(os.path.join(filepath,'fake_'+str(i)+'.png'))
                plt.close()
      #      plt.show()

