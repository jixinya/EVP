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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class Lm_encoder(nn.Module):
    def __init__(self):
        super(Lm_encoder, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(16,256),
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

    def forward(self, mfcc):
       # mfcc= torch.unsqueeze(mfcc, 1)
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_eocder(mfcc)
        feature = feature.view(feature.size(0),-1)
        x = self.emotion_eocder_fc(feature)


        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(128*7,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,16),#20
            )

    def forward(self, lstm_input):
        hidden = ( torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                      torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        fc_out   = []
        for step_t in range(lstm_out.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))


  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]


        return torch.stack(fc_out, dim = 1)



class AT_emoiton(nn.Module):
    def __init__(self,config):
        super(AT_emoiton, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = Decoder()
        self.lm_encoder = Lm_encoder()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1loss = nn.L1Loss()

        self.pca = torch.FloatTensor(np.load('../basics/U_106.npy')[:, :16]).cuda()
        self.mean = torch.FloatTensor(np.load('../basics/mean_106.npy')).cuda()



        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.lm_encoder.parameters()), config.lr,betas=(config.beta1, config.beta2))






    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self,example_landmark, landmark, mfccs):


        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
            e_feature = self.emo_encoder(mfcc)

            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)

     #   real = landmark - example_landmark.expand_as(landmark)



        loss_pca = self.mse_loss_fn(fake, landmark)

        fake_result = torch.mm(fake[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)
        for i in range(1,len(fake)):
            fake_result = torch.cat((fake_result,torch.mm(fake[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)),0)



        result = torch.mm(landmark[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)
        for i in range(1,len(landmark)):
            result = torch.cat((result,torch.mm(landmark[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)),0)



      #  result = torch.mm(landmark,self.pca.transpose(0,1))+self.mean.expand(len(fake),16,212)

        loss_lm = self.mse_loss_fn(fake_result, result)
       # loss = self.l1loss(fake, landmark)



        return fake, loss_pca,10*loss_lm

    def forward(self, example_landmark, mfccs,emo_mfcc):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            emo = emo_mfcc[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
            e_feature = self.emo_encoder(emo)

            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)


        return fake

    def feature_input(self, example_landmark, mfccs,emo_feature):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = emo_feature[:,step_t,:]
            c_feature = self.con_encoder(mfcc)
     #       e_feature = self.emo_encoder(emo)



            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)


        return fake

    def update_network(self, loss_pca, loss_lm):

        loss = loss_pca + loss_lm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, example_landmark, landmark, mfccs):

        self.lm_encoder.train()
        self.decoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        output, loss_pca, loss_lm = self.process(example_landmark, landmark, mfccs)

        self.update_network(loss_pca, loss_lm )

        return output, loss_pca, loss_lm

    def val_func(self, example_landmark, landmark, mfccs):
        self.lm_encoder.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()

        with torch.no_grad():
            output, loss_pca, loss_lm  = self.process(example_landmark, landmark, mfccs)

        return output, loss_pca, loss_lm

    def save_fig(self,data,output,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']


        return 0


class Con_Decoder(nn.Module):
    def __init__(self):
        super(Con_Decoder, self).__init__()
        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,16),#20
            )

    def forward(self, lstm_input):
        hidden = ( torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                      torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        fc_out   = []
        for step_t in range(lstm_out.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))


  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]


        return torch.stack(fc_out, dim = 1)



class AT_net(nn.Module):
    def __init__(self,config):
        super(AT_net, self).__init__()

        self.con_encoder = Ct_encoder()
    #    self.emo_encoder = EmotionNet()
        self.decoder = Con_Decoder()
        self.lm_encoder = Lm_encoder()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1loss = nn.L1Loss()

        self.pca = torch.FloatTensor(np.load('../basics/U_106.npy')[:, :16]).cuda()
        self.mean = torch.FloatTensor(np.load('../basics/mean_106.npy')).cuda()

      #  self.pca = torch.FloatTensor(np.load('/home/thea/data/MEAD/ATnet_emotion/basics/U_106.npy')[:, :16]).cuda()
      #  self.mean = torch.FloatTensor(np.load('/home/thea/data/MEAD/ATnet_emotion/basics/mean_106.npy')).cuda()

        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.lm_encoder.parameters()), config.lr,betas=(config.beta1, config.beta2))






    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self,example_landmark, landmark, mfccs):


        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
        #    e_feature = self.emo_encoder(mfcc)

            current_feature = c_feature
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)

     #   real = landmark - example_landmark.expand_as(landmark)



        loss_pca = self.mse_loss_fn(fake, landmark)

        fake_result = torch.mm(fake[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)
        for i in range(1,len(fake)):
            fake_result = torch.cat((fake_result,torch.mm(fake[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)),0)



        result = torch.mm(landmark[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)
        for i in range(1,len(landmark)):
            result = torch.cat((result,torch.mm(landmark[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)),0)



      #  result = torch.mm(landmark,self.pca.transpose(0,1))+self.mean.expand(len(fake),16,212)

        loss_lm = self.mse_loss_fn(fake_result, result)
       # loss = self.l1loss(fake, landmark)



        return fake, loss_pca,10*loss_lm

    def forward(self, example_landmark, mfccs):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
          #  emo = emo_mfcc[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
        #    e_feature = self.emo_encoder(emo)

            current_feature = c_feature
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)


        return fake

    def update_network(self, loss_pca, loss_lm):

        loss = loss_pca + loss_lm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, example_landmark, landmark, mfccs):

        self.lm_encoder.train()
        self.decoder.train()
        self.con_encoder.train()
    #    self.emo_encoder.train()

        output, loss_pca, loss_lm = self.process(example_landmark, landmark, mfccs)

        self.update_network(loss_pca, loss_lm )

        return output, loss_pca, loss_lm

    def val_func(self, example_landmark, landmark, mfccs):
        self.lm_encoder.eval()
        self.decoder.eval()
        self.con_encoder.eval()
     #   self.emo_encoder.eval()

        with torch.no_grad():
            output, loss_pca, loss_lm  = self.process(example_landmark, landmark, mfccs)

        return output, loss_pca, loss_lm

    def save_fig(self,data,output,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']


        return 0

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
     #   self.lstm_fc = nn.Sequential(
     #       nn.Linear(256,16),#20
     #       )

    def forward(self, lstm_input):
        hidden = ( torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                      torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        fc_out   = []
        for step_t in range(lstm_out.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(fc_in)


  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]


        return torch.stack(fc_out, dim = 1)

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128*3,256),
            nn.ReLU(True),
            nn.Linear(256,16)


            )
     #   self.lstm_fc = nn.Sequential(
     #       nn.Linear(256,16),#20
     #       )

    def forward(self, x):
        result = self.net(x)


        return result


class Emotion_After(nn.Module):
    def __init__(self,config):
        super(Emotion_After, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.lstm = LSTM()
        self.fc = Simple()
        self.lm_encoder = Lm_encoder()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1loss = nn.L1Loss()

        self.pca = torch.FloatTensor(np.load('../basics/U_106.npy')[:, :16]).cuda()
        self.mean = torch.FloatTensor(np.load('../basics/mean_106.npy')).cuda()



        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.lstm.parameters())
                                            +list(self.fc.parameters())
                                            +list(self.lm_encoder.parameters()), config.lr,betas=(config.beta1, config.beta2))






    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self,example_landmark, landmark, mfccs):


        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.lstm(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = self.emo_encoder(mfcc)
            c_part = c_result[:,step_t]

            features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)
        print(fc_input.type())
        fake = self.fc(fc_input)

     #   real = landmark - example_landmark.expand_as(landmark)



        loss_pca = self.mse_loss_fn(fake, landmark)

        fake_result = torch.mm(fake[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)
        for i in range(1,len(fake)):
            fake_result = torch.cat((fake_result,torch.mm(fake[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)),0)



        result = torch.mm(landmark[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)
        for i in range(1,len(landmark)):
            result = torch.cat((result,torch.mm(landmark[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)),0)



      #  result = torch.mm(landmark,self.pca.transpose(0,1))+self.mean.expand(len(fake),16,212)

        loss_lm = self.mse_loss_fn(fake_result, result)
       # loss = self.l1loss(fake, landmark)



        return fake, loss_pca,10*loss_lm

    def forward(self, example_landmark, mfccs,emo_mfcc):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.lstm(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
            mfcc = emo_mfcc[:,step_t,:,:].unsqueeze(1)
            e_feature = self.emo_encoder(mfcc)
            c_part = c_result[:,step_t]

            features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)

        fake = self.fc(fc_input)


        return fake

    def feature_input(self, example_landmark, mfccs,emo_feature):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.LSTM(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
      #      mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = emo_feature[:,step_t]
            c_part = c_result[:,step_t]

            features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)

        fake = self.FC(fc_input)


        return fake

    def update_network(self, loss_pca, loss_lm):

        loss = loss_pca + loss_lm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, example_landmark, landmark, mfccs):

        self.lm_encoder.train()

        self.con_encoder.train()
        self.emo_encoder.train()

        self.lstm.train()
        self.fc.train()


        output, loss_pca, loss_lm = self.process(example_landmark, landmark, mfccs)

        self.update_network(loss_pca, loss_lm )

        return output, loss_pca, loss_lm

    def val_func(self, example_landmark, landmark, mfccs):
        self.lm_encoder.eval()
        self.lstm.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()
        self.fc.eval()

        with torch.no_grad():
            output, loss_pca, loss_lm  = self.process(example_landmark, landmark, mfccs)

        return output, loss_pca, loss_lm

    def save_fig(self,data,output,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']


        return 0



class Con_lstm(nn.Module):
    def __init__(self):
        super(Con_lstm, self).__init__()
        self.lstm = nn.LSTM(256,256,3,batch_first = True)
     #   self.lstm_fc = nn.Sequential(
     #       nn.Linear(256,16),#20
     #       )

    def forward(self, lstm_input):
        hidden = ( torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                      torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        fc_out   = []
        for step_t in range(lstm_out.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(fc_in)


  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]


        return torch.stack(fc_out, dim = 1)

class Fc(nn.Module):
    def __init__(self):
        super(Fc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128*7,256),
            nn.ReLU(True),
            nn.Linear(256,16)


            )
     #   self.lstm_fc = nn.Sequential(
     #       nn.Linear(256,16),#20
     #       )

    def forward(self, x):
        result = self.net(x)


        return result


class Landmark_After(nn.Module):
    def __init__(self,config):
        super(Landmark_After, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.lstm = Con_lstm()
        self.fc = Fc()
        self.lm_encoder = Lm_encoder()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1loss = nn.L1Loss()

        self.pca = torch.FloatTensor(np.load('/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/basics/U_106.npy')[:, :16]).cuda()
        self.mean = torch.FloatTensor(np.load('/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/basics/mean_106.npy')).cuda()



        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.lstm.parameters())
                                            +list(self.fc.parameters())
                                            +list(self.lm_encoder.parameters()), config.lr,betas=(config.beta1, config.beta2))






    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self,example_landmark, landmark, mfccs):


        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
          #  features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(c_feature)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.lstm(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = self.emo_encoder(mfcc)
            c_part = c_result[:,step_t]

            current_features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            features = torch.cat([l,  current_features], 1)
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)
        print(fc_input.type())
        fake = self.fc(fc_input)

     #   real = landmark - example_landmark.expand_as(landmark)



        loss_pca = self.mse_loss_fn(fake, landmark)

        fake_result = torch.mm(fake[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)
        for i in range(1,len(fake)):
            fake_result = torch.cat((fake_result,torch.mm(fake[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],212).unsqueeze(0)),0)



        result = torch.mm(landmark[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)
        for i in range(1,len(landmark)):
            result = torch.cat((result,torch.mm(landmark[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],212).unsqueeze(0)),0)



      #  result = torch.mm(landmark,self.pca.transpose(0,1))+self.mean.expand(len(fake),16,212)

        loss_lm = self.mse_loss_fn(fake_result, result)
       # loss = self.l1loss(fake, landmark)



        return fake, loss_pca,10*loss_lm

    def forward(self, example_landmark, mfccs,emo_mfcc):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
          #  features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(c_feature)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.lstm(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
            mfcc = emo_mfcc[:,step_t,:,:].unsqueeze(1)
            e_feature = self.emo_encoder(mfcc)
            c_part = c_result[:,step_t]

            current_features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            features = torch.cat([l,  current_features], 1)
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)
        print(fc_input.type())
        fake = self.fc(fc_input)


        return fake

    def feature_input(self, example_landmark, mfccs,emo_feature):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_eocder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_eocder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
      #      e_feature = self.emo_encoder(mfcc)

         #   current_feature = torch.cat([c_feature,e_feature],1)
         #   features = torch.cat([l,  c_feature], 1) #torch.Size([16, 768])
            lstm_input.append(c_feature)

        lstm_input = torch.stack(lstm_input, dim = 1)

        c_result = self.LSTM(lstm_input)

        fc_input = []
        for step_t in range(mfccs.size(1)):
      #      mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = emo_feature[:,step_t]
            c_part = c_result[:,step_t]

            current_features = torch.cat([c_part,e_feature], 1) #torch.Size([16, 768])
            features = torch.cat([l,  current_features], 1)
            fc_input.append(features)
        fc_input = torch.stack(fc_input, dim = 1)

        fake = self.fc(fc_input)


        return fake

    def update_network(self, loss_pca, loss_lm):

        loss = loss_pca + loss_lm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, example_landmark, landmark, mfccs):

        self.lm_encoder.train()

        self.con_encoder.train()
        self.emo_encoder.train()

        self.lstm.train()
        self.fc.train()


        output, loss_pca, loss_lm = self.process(example_landmark, landmark, mfccs)

        self.update_network(loss_pca, loss_lm )

        return output, loss_pca, loss_lm

    def val_func(self, example_landmark, landmark, mfccs):
        self.lm_encoder.eval()
        self.lstm.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()
        self.fc.eval()

        with torch.no_grad():
            output, loss_pca, loss_lm  = self.process(example_landmark, landmark, mfccs)

        return output, loss_pca, loss_lm

    def save_fig(self,data,output,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']


        return 0