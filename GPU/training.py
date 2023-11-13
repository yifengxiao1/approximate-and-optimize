import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
from pathlib import Path


class Linear(nn.Module):
    def __init__(self, x_len):
        super(Linear, self).__init__()
        self.x_len = x_len
        # self.scenario_len = scenario_len
        # self.dropout = nn.Dropout(0.1)
        self.fcx1 = nn.Sequential(nn.Linear(self.x_len, self.x_len), nn.ReLU())
        self.fcx2 = nn.Sequential(nn.Linear(self.x_len, 1), nn.ReLU())

        self.obj1 = nn.Sequential(nn.Linear(self.x_len + 256, self.x_len), nn.ReLU())
        self.obj2 = nn.Sequential(nn.Linear(self.x_len, 1), nn.ReLU())

        self.conv1 = nn.Sequential(
            # 输入[1,370,422]
            nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=3,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=0,        # 给图外边补上0
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((200,200))   # 经过池化 输出[16,32,32] 传入下一个卷积
        )
        self.conv2 = nn.Sequential(
            # 输入[16,32,32]
            nn.Conv2d(
                in_channels=16,    # 输入图片的高度
                out_channels=64,  # 输出图片的高度
                kernel_size=3,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=0,        # 给图外边补上0
            ),
            # 经过卷积层 输出[256,15,15] 传入池化层
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=9)    # 输出[64,22,22]
        )

        self.conv3 = nn.Sequential(
            # 输入[64,66,66]
            nn.Conv2d(
                in_channels=64,    # 输入图片的高度
                out_channels=256,  # 输出图片的高度
                kernel_size=3,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=0,        # 给图外边补上0
            ),
            # 经过卷积层 输出[256,20,20] 传入池化层
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=20)    # 输出[256,1,1]
        )


    def forward(self, x, xobj, eobj):
        e = torch.reshape(self.conv3(self.conv2(self.conv1(eobj))),(-1,256))
        obj = torch.cat([xobj, e], dim=1)
        outobj = self.obj2(self.obj1(obj))
        # scenario = self.scenario1(scenario)
        # scenario = torch.mean(scenario,dim=0)
        # scenario = self.scenario3(self.scenario2(scenario))
        # input = torch.cat([x, scenario], dim=0)
        outx = self.fcx2(self.fcx1(x))
        return outobj+outx



def test(problem_number):
    train_x,train_xobj,train_eobj, train_label, test_x,test_xobj,test_eobj, test_label = get_data(problem_number, j='test')
    test_label = np.array(test_label)
    eobj = torch.reshape(test_eobj,(-1,1,370,422))
    loss = []
    for j in range(10):
        E_model = torch.load(f'2_0/model2_{problem_number}_epoch{j}.pt',map_location=torch.device('cpu'))
        with torch.no_grad():
            prediction = np.array(E_model.forward(test_x,test_xobj,eobj))
        error_ratio = np.abs((prediction - test_label))/test_label
        loss.append(np.mean(error_ratio))
        print(np.mean(error_ratio))
    return loss

def get_data(problem_number,j, need_ind = False):
    if j=='test':
        p = Path(f'training_data_instance/problem2_{problem_number}_test.pkl')
    else:
        p = Path(f'training_data_instance/problem2_{problem_number}_{j+10}.pkl')
    f_save = open(p, 'rb')
    data = pkl.load(f_save)
    train_x = []
    train_xobj = []
    train_eobj = []
    train_data_ind = []  # use x-index version to reduce input size and see whether this works
    test_x = []
    test_xobj = []
    test_eobj = []
    test_data_ind = []
    train_label = []
    test_label = []
    train_size = len(data) * 0.9
    if need_ind:
        for i in range(len(data)):
            if i < train_size:
                train_data_ind.append(data[i]['x_index'])
                train_label.append(data[i]['obj'])
            else:
                test_data_ind.append(data[i]['x_index'])
                test_label.append(data[i]['obj'])
        return torch.tensor(train_data_ind).float(), torch.tensor(train_label).float(), torch.tensor(test_data_ind).float(), torch.tensor(test_label).float()
    else:
        for i in range(len(data)):
            if i < train_size:
                train_x.append(data[i]['x'])
                train_xobj.append(data[i]['xobj'])
                train_eobj.append(data[i]['eobj'])
                train_label.append(data[i]['obj'])
            else:
                test_x.append(data[i]['x'])
                test_xobj.append(data[i]['xobj'])
                test_eobj.append(data[i]['eobj'])
                test_label.append(data[i]['obj'])
        print('Gotten')
        return torch.tensor(train_x).float(),torch.tensor(train_xobj).float(),torch.tensor(train_eobj).float(), torch.tensor(train_label).float(), torch.tensor(test_x).float(),torch.tensor(test_xobj).float(),torch.tensor(test_eobj).float(), torch.tensor(test_label).float()
    



    
def train(problem_number, epoch=500, batch_size = 32, need_ind = False):
    train_x,train_xobj,train_eobj, train_label, test_x,test_xobj,test_eobj, test_label = get_data(problem_number,0, need_ind)
    input_len = train_x.size()[1]
    model = Linear(input_len)

       
    
    Loss =  torch.nn.MSELoss()


    if torch.cuda.is_available():
        model.cuda()
        # train_x.cuda()
        # train_xobj.cuda()
        # train_eobj.cuda()
        # train_label.cuda()
        # test_x.cuda()
        # test_xobj.cuda()
        # test_eobj.cuda()
        # test_label.cuda()
        # Loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    
    for k in range(10):
        train_x,train_xobj,train_eobj, train_label, test_x,test_xobj,test_eobj, test_label = get_data(problem_number,k, need_ind)
        for i in range(epoch):
            for j in range(round):
                indecs = np.random.choice(len(train_label), batch_size, replace=False)
                x = train_x[indecs].cuda()
                xobj = train_xobj[indecs].cuda()
                eobj = torch.reshape(train_eobj[indecs].cuda(),(-1,1,370,422))
                label = train_label[indecs].cuda()
                optimizer.zero_grad()
                output = model.forward(x, xobj, eobj).reshape(-1)
                # print(output)
                # penalty = 0
                # for name, param in model.named_parameters():
                #     if 'weight' in name:
                #         penalty += (torch.norm(param, 1) + torch.pow(torch.norm(param, 2), 2))
                loss = ( Loss(label, output) )  #/ batch_size 
                loss.backward()
                optimizer.step()
                    # index = np.random.choice(1000)
                    # try_x, try_scenario = x_and_its_scenario(index)
                    # print(E_model.forward(torch.from_numpy(try_x).float(),torch.from_numpy((try_scenario)).float()))
                    # print(labels[index])
                print(loss)
        torch.save(model,f'2_0/model2_{problem_number}_epoch{k}.pt')
    # if need_ind:
    #     torch.save(model,f'model/ind_model/ind_ model2_{problem_number}.pt')
    # else:
    #     torch.save(model,f'model/model_modified/model2_{problem_number}.pt')
    # torch.save(model,f'model2_{problem_number}_.pt')

if __name__=='__main__':
    # batch_size = 100
    # round = 500
    # epoch = 50
    # train(problem_number=0, epoch=epoch, batch_size=batch_size)
    print(test(0))