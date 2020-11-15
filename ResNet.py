'''
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
官方实现
'''
import torch
import torch.nn as nn #最常用的两个库
from torch.hub import load_state_dict_from_url #使用这个函数可以加载预训练的权重值

#训练好的预训练的权重值，不同层数权重值的路径
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

#封装一个3*3的卷积，in_planes为输入图像通道数，out_planes为卷积产生通道数,因为有bn层，所以不需要bias
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

#封装一个1*1的卷积,用于升降维，不需要填充
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#继承nn.model,与BottleNeck的不同在于有没有1x1的卷积层来降低计算量
class BasicBlock(nn.module):
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__() #父类初始化
        if norm_layer is None: #normalization layer，正则化层
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)  #这里还只是生成一个类，而调用传入的参数是不一样的
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)#inplace=True是指将计算得到的值覆盖之前的值
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        '''
        关于conv2d的调用
        输入x：
        [ batch_size, channels, height_1, width_1 ]
        batch_size: 一个batch中样例的个数	
        channels: 通道数，也就是当前层的深度	
        height_1: 图片的高	
        width_1: 图片的宽
        
        输出res:
        [ batch_size,output, height_3, width_3 ]
        batch_size	一个batch中样例的个数，同上	2
        output	输出的深度	8
        height_3　	卷积结果的高度	h1-h2+1 = 7-2+1 = 6
        weight_3	卷积结果的宽度	w1-w2+1 = 3-3+1 = 1   
        '''
        
    def forword(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        #out在处理时，若是stride=2，则相当于进行了下采样了
        if self.downsample is not None:#整体网络中有些位置尺寸会发生变化，需要下采样，因此x需要下采样保持同步，如图中虚线位置，保持x与未加x前的out(即fx)同步
            identity=  self.downsample(x)
        
        out += identity
        out = self. relu(out) #这个激活函数在融合后再调用
        
        return out
            
class BottleNeck(nn.module):
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        #
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        #x是个张量，out也是个张量,传入的图像？
    def forward(self, x):
        identity = x
        ###这调用的个啥
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        #下采样是因为图像大小变了
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
        
class ResNet(nn.module):
    
    #默认为image_net,因此网络分为1000类，block指用basicblock还是bottleneck，layers即每个stage中block需要重复多少次
    def __init__(self, block, layers, num_class=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d()
        self._norm_layer = norm_layer
        
        #图给的是64
        self.inplanes = 64
        
        #因为kernel_size为7，为了使卷积后图像大小不变，选用padding=3，因为输入的是彩图RBG三个通道，所以为3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(Kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) #此处layer对应stage
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #因为第一个stage处做了max pool，已经output_size/2了，就stride=1即可，不需要下采样
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  #全局平均池化
        self.fc = nn.Linear(512*block.expansion, num_class)#最终分为1000类，即1*1000矩阵 
        ###fc与linear学习
        
        #参数初始化，这里的modules()是父类的函数,返回的是迭代器iterator,将整个模型的所有构成由浅入深依次遍历出来
        for m in self.modules():#初始化学习
            #如果为卷积层，则用kaiming初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  #利用kaiming初始化搭配ReLU函数
            #如果为bn层，用0和1初始化
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):#会认为子类是一种父类类型，考虑继承关系。而type()不考虑
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    #实现中间的层(stage),重点 block传入的是一个类
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        
        #判断是否需要下采样,此处生成下采样函数 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride), #同时调整spatial(H x W)和channel两个方向
                    norm_layer(planes*block.expansion)
            )
            
        layers=[]
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer)) #第一个block单独处理，因为存在下采样的情况，之后不用传下采样的参数
        self.inplanes = planes * block.expansion   #记录layerN的channel变化，self.inplanes在这里发生变化
        for _ in range(1, blocks):  #从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)   #使用Sequential层组合blocks，形成stage
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flattern(x, 1) #降维，按行的顺序
        x = self.fc(x)
        
        return x
        
    '''
        全连接层（fully connected layers，FC）在整个卷积神经网络中起到“分类器”的作用。如果
    说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空间的话，全连接层则起到将学
    到的“分布式特征表示”映射到样本标记空间的作用。
    '''

#加载预训练的参数
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
        
    return model
        
def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained, progress, **kwargs)

model = resnet152(pretrain = True)
model.eval()