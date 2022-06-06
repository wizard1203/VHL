import torch.nn as nn
import torch.nn.functional as F




# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out



class SWD_CNN(nn.Module):
    def __init__(self, hidden_channel_list=[6, 16], hidden_dims=[120, 84], output_dim=10,
                input_channels=3):
        super(SWD_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channel_list[0], 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # middle_hidden_channel_list = hidden_channel_list[1:-1]
        layers = []
        for i in range(len(hidden_channel_list) - 2):
            index = i+1
            layers.append(nn.Conv2d(hidden_channel_list[index-1], hidden_channel_list[index],
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.conv_last = nn.Conv2d(hidden_channel_list[-2], hidden_channel_list[-1], 5)

        # self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.linear_input_dim = hidden_channel_list[-1] * 5 * 5
        self.linear = nn.Linear(self.linear_input_dim, output_dim)


    def forward(self, x):
        #out = self.conv1(x)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = self.conv2(out)
        #out = self.relu(out)
        #out = self.pool(out)
        #out = out.view(-1, 16 * 5 * 5)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.hidden_layers(x)
        x = self.pool(self.relu(self.conv_last(x)))
        x = x.view(-1, self.linear_input_dim)

        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.linear(x)

        return x






def build_SWD_CNN(model_name="swdcnn2", hidden_dims=[120, 84], output_dim=10,
                input_channels=3):
    if model_name == "swdcnn2":
        hidden_channel_list=[6, 16]
    elif model_name == "swdcnn3":
        hidden_channel_list=[6, 8, 16]
    elif model_name == "swdcnn4":
        hidden_channel_list=[6, 8, 8, 16]
    elif model_name == "swdcnn5":
        hidden_channel_list=[6, 8, 8, 8, 16]
    elif model_name == "swdcnn6":
        hidden_channel_list=[6, 8, 8, 8, 8, 16]
    elif model_name == "swdcnn7":
        hidden_channel_list=[6, 8, 8, 8, 8, 8, 16]
    elif model_name == "swdcnn8":
        hidden_channel_list=[6, 8, 8, 8, 8, 8, 8, 16]
    else:
        raise NotImplementedError

    model = SWD_CNN(hidden_channel_list, hidden_dims, output_dim, input_channels)

    return model














































