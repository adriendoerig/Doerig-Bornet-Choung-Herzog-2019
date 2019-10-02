import os
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from batch_maker import StimMaker

# load saved learning parameters
def load_checkpoint(model, optimizer=None, losslogger=None, fname='checkpoint.pt'):
    # note that input model and optimizer should be pre-defined!!
    start_epoch = 0
    if os.path.isfile(fname):
        print('=> Loading checkpoint' + fname)
        checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if losslogger is not None:
            losslogger = checkpoint['losslogger']
        print('=> loaded checkpoint ' + fname + ' epoch %3i'%start_epoch)
    else:
        print('=> no checkpoint found at' + fname)
    return model, optimizer, start_epoch, losslogger

# For occlusion analysis 
def get_batch_outputs_pytorch(imgs, models, backbone_model):
    import torch
    import torch.nn as nn
    
    ns = range(18)
    
    occlu_list = np.zeros((imgs.shape[0],len(models),2))
    
    if next(backbone_model.parameters()).is_cuda:
        imgs = imgs.to('cuda')
        
    with torch.no_grad():
        # from pre-trained network
        inputs = backbone_model(imgs,ns)

        for i,(n,model) in enumerate(zip(ns,models)):
            model.eval()
            occlu_list[:,i,:] = model(inputs[n]).cpu().numpy()

    return occlu_list


# Test models
def test_models(my_shapenet, models, n_blocks=10, n_samples=50, device='cpu', model_path='./', ns=None, subjs=None, fignm='results.svg',shapeMatrix=[],test_types=None):

    from make_plots import plotFig3a

    # create test set and test all trained models, then plot both train&test errors
    check_points = [f for f in os.listdir(model_path) if ((os.path.isfile(os.path.join(model_path,f)) & ('pt' in f)))]

    # If ns and sb are not specified, use all saved checkpoints
    up_ns = [lambda: False, lambda: True][ns is None]()
    up_sb = [lambda: False, lambda: True][subjs is None]()
    for cp in check_points:
        l,s = cp.split('_')[1], cp.split('_')[3].split('.')[0]
        if up_ns & (int(l) not in ns)   :
            ns.append(int(l))
            ns.sort()
        if up_sb & (int(s) not in subjs):
            subjs.append(int(s))
            subjs.sort()

    n_sub    = len(subjs)

    test_pcs    = np.zeros((n_sub, n_blocks, len(shapeMatrix)*len(ns)))

    # Test all trained networks (subjs)
    for s in subjs:

        # Load all model checkpoints (decoders from each layer)
        for i, model in enumerate(models):

            this_model_path = model_path + '/layer_' + str(ns[i]) + '_subject_' + str(s) + '.pt'
            model,_,_,_ = load_checkpoint(model, fname=this_model_path)

        print('Testing subject '+str(s)+'...')

        # Repeat testing for n_blocks and average to get the percent correct vernier discrimination performance
        for r in range(n_blocks):
            print('Block '+ str(r))

            with torch.no_grad():
                for j, shape in enumerate(shapeMatrix):
                    # generate data set on fly
                    t_i, t_t = make_dataset(btch_size=n_samples,shapeMatrix=[shape],test_types=test_types[j],device=device,type='test')
                    inputs = my_shapenet(t_i[0], ns)

                    for i, (n, model) in enumerate(zip(ns, models)):
                        model.eval()
                        t_o = model(inputs[n])
                        test_pcs[s,r,i*len(shapeMatrix)+j] = 100*float((t_o.argmax(1)==t_t[0]).sum())/t_t[0].shape[0]
                        
    # plot test result (pc)
    pc_means = test_pcs.mean(axis=1)

    # Save plot and data for plot to the model_path
    for i,n in enumerate(ns):
        print('Plot layer '+str(n)+' from '+str(i*len(shapeMatrix))+' to '+str((i+1)*len(shapeMatrix)))
        plotFig3a(pc_means.mean(axis=0)[i*len(shapeMatrix):(i+1)*len(shapeMatrix)], pc_means.std(axis=0)[i*len(shapeMatrix):(i+1)*len(shapeMatrix)], i, model_path=model_path)


# Generate train and test data sets
def make_dataset(btch_size=50, shapeMatrix=[], type='train', device='cpu', test_types=None):

    imgSize = (227, 227)
    shapeSize = 18
    barWidth = 1

    rufus = StimMaker(imgSize, shapeSize, barWidth)

    def transform_arrays(images, labels, device):
        images, labels = torch.Tensor(images).to(device), torch.Tensor(labels).to(device=device, dtype=torch.int64)
        images = images.permute(0, 3, 1, 2)  # Change to pytorch tensor dimension: batch x channel x height x width
        # normalize data
        mean, std = images.mean(), images.std()
        images.sub_(mean).div_(std)

        return images, labels

    if type is 'test':
        # Testing set
        if test_types is None:
            test_types = ['test']*len(shapeMatrix)
        ratios = [[0,0,0,1]]*len(shapeMatrix)

        batch_images, batch_labels = ([],[])
        for r,s in zip(ratios, shapeMatrix):
            imgs, labs = rufus.generate_Batch(btch_size, r, noiseLevel=0.1, shapeMatrix=s)
            imgs, labs = transform_arrays(imgs, labs, device)
            batch_images.append(imgs)
            batch_labels.append(labs)
        batch_labels.append(test_types)

    elif type is 'train':
        # Training set
        ratios = [0, 0, 1, 0]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
        batch_images, batch_labels = rufus.generate_Batch(btch_size, ratios, noiseLevel=0.1)
        batch_images, batch_labels = transform_arrays(batch_images, batch_labels, device)

    return batch_images, batch_labels


####################################################################################################################
    # Backbone model structure (Resnet50), and Decoder (MyClassifier) structure 
####################################################################################################################

# Load pretrained Shape biased network (https://github.com/rgeirhos/texture-vs-shape)
def load_model(model_name):

    model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }
    if model_name in model_urls.keys():
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
        model.load_state_dict(checkpoint["state_dict"])
    elif model_name == 'resnet50_ori':
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.DataParallel(model).cuda()

    return model

# Shapebiased network (Geirhos et al.)
class ShapeNet(nn.Module):

    def __init__(self, **kwargs):

        super(ShapeNet, self).__init__()

        # default options
        opt = {'model'      : 'resnet50_trained_on_SIN',
               'num_classes': 2}
        # update parameters
        for k,v in kwargs.items():
            opt[k] = v

        resnet = load_model(opt['model']).module
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, ns):

        ys = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_ = self.layer1[0].conv1(x)
        x_ = self.layer1[0].bn1(x_)
        x_ = self.layer1[0].conv2(x_)
        x_ = self.layer1[0].bn2(x_)
        x_ = self.layer1[0].conv3(x_)
        x_ = self.layer1[0].bn3(x_)
        x_ = self.layer1[0].relu(x_)
        ys.append(x_)

        x = self.layer1[0](x)
        ys.append(x)
        x = self.layer1[1](x)
        ys.append(x)
        x = self.layer1[2](x)
        ys.append(x)
        x = self.layer2[0](x)
        ys.append(x)
        x = self.layer2[1](x)
        ys.append(x)
        x = self.layer2[2](x)
        ys.append(x)
        x = self.layer2[3](x)
        ys.append(x)
        x = self.layer3[0](x)
        ys.append(x)
        x = self.layer3[1](x)
        ys.append(x)
        x = self.layer3[2](x)
        ys.append(x)
        x = self.layer3[3](x)
        ys.append(x)
        x = self.layer3[4](x)
        ys.append(x)        
        x = self.layer3[5](x)
        ys.append(x)
        x = self.layer4[0](x)
        ys.append(x)
        x = self.layer4[1](x)
        ys.append(x)
        x = self.layer4[2](x)
        ys.append(x)
        x = self.avgpool(x)
        ys.append(x)

        return ys

# Classifier with 1 hidden layer
class MyClassifier(nn.Module):

    # Classifier for vernier discrimination

    def __init__(self, input_size, n_hidden=512):

        super(MyClassifier, self).__init__()

        input_len = reduce(lambda a, b: a*b, input_size)
        self.input_size       = input_size
        self.input_len        = input_len

        self.late_classifiers = nn.Sequential(
            nn.BatchNorm1d(input_len),
            nn.Linear(input_len, n_hidden),
            nn.ELU(inplace=True),
            nn.Linear(n_hidden, 2),
            nn.Softmax())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.late_classifiers(x)

# Give the size of a layer given input size
def get_output_size(model, input_size=(3, 227, 227), device='CUDA', model_nm=None):
    with torch.no_grad():

        output_sizes = []
        x = torch.zeros(input_size).unsqueeze_(dim=0).to(device)

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        # output_sizes.append(x.size()[1:])
        x = model.maxpool(x)
        x_ = model.layer1[0].conv1(x)
        x_ = model.layer1[0].bn1(x_)
        # output_sizes.append(x_.size()[1:])
        x_ = model.layer1[0].conv2(x_)
        x_ = model.layer1[0].bn2(x_)
        # output_sizes.append(x_.size()[1:])
        x_ = model.layer1[0].conv3(x_)
        x_ = model.layer1[0].bn3(x_)
        x_ = model.layer1[0].relu(x_)
        output_sizes.append(x_.size()[1:])
        x = model.layer1[0](x)
        output_sizes.append(x.size()[1:])
        x = model.layer1[1](x)
        output_sizes.append(x.size()[1:])
        x = model.layer1[2](x)
        output_sizes.append(x.size()[1:])
        x = model.layer2[0](x)
        output_sizes.append(x.size()[1:])
        x = model.layer2[1](x)
        output_sizes.append(x.size()[1:])
        x = model.layer2[2](x)
        output_sizes.append(x.size()[1:])
        x = model.layer2[3](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[0](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[1](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[2](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[3](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[4](x)
        output_sizes.append(x.size()[1:])
        x = model.layer3[5](x)
        output_sizes.append(x.size()[1:])
        x = model.layer4[0](x)
        output_sizes.append(x.size()[1:])
        x = model.layer4[1](x)
        output_sizes.append(x.size()[1:])
        x = model.layer4[2](x)
        output_sizes.append(x.size()[1:])
        x = model.avgpool(x)
        output_sizes.append(x.size()[1:])

    return output_sizes

