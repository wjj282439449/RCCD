from collections import OrderedDict
from .layers import *
from .layers import DANetModule
import copy
from .convnext import getEncoder
from .resnet import resnet18
import torch.nn.functional as F
from einops.einops import rearrange
__all__ = ['UCCD', 'get_UCCD',]



class UCCD(nn.Module):
    def __init__(self, encoder, args, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.encoderName = args.backboneName

        self.isAttentionLayer = args.embeddableAttention["isActivateLayer"]
        self.isAttentionConcat = args.embeddableAttention["isAttentionConcat"]
        self.AttentionModule = globals()[args.embeddableAttention["name"]]
        #NonLocal2D    DANetModule

        self.isTemporalAttention = args.interactiveAttention["isActivateLayer"]
        self.SpatiotemporalAttentionModule = globals()[args.interactiveAttention["name"]]
        #SpatiotemporalAttentionBase  SpatiotemporalAttentionFull  SpatiotemporalAttentionFullNotWeightShared

        self.isCBAM = args.CBAM["isActivateLayer"]
        self.isCBAMconcat = args.CBAM["isAttentionConcat"]
        self.CBAModule = globals()[args.CBAM["name"]]#Easy to extend to more CBAM-like attention mechanisms.

        self.isDeeplySupEncoder = args.deeplySupervisedEncoder["isActivateLayer"]
        self.deeplySupEncoder = globals()[args.deeplySupervisedEncoder["name"]]

        self.isDeeplySupDecoder = args.deeplySupervisedDecoder["isActivateLayer"]
        self.deeplySupDecoder = globals()[args.deeplySupervisedDecoder["name"]]

        self.decoderBlock = globals()[args.decoderBlock["name"]]

        if "resnet" in self.encoderName:
            self.encoderNameScale = 2
        else:
            self.encoderNameScale = 4
        self.isFeatureFusion = args.detectorHead["isFeatureFusion"]
        self.isConcatInput = args.detectorHead["isConcatInput"]
        self.outChannels = len(args.listOfCategoryNames)

        self.deeplySupervisedFeatures = []
        self.isBackboneFeaturesSimilarity = args.isBackboneFeaturesSimilarity
        if self.isBackboneFeaturesSimilarity:
            self.backboneFeaturesGram1 = []
            self.backboneFeaturesGram2 = []
        self.downScale = [16, 8, 4, 0]
        self.stageNumber = 4
        self.backbone_depth = {
                                'tiny': [3, 3, 9, 3], 
                                'small': [3, 3, 27, 3],
                                'base': [3, 3, 27, 3], 
                                'large': [3, 3, 27, 3],
                                'xlarge': [3, 3, 27, 3],
                                "resnet18": [2, 2, 2, 2]
                            }
        self.backbone_dims = {
                                'tiny': [96, 192, 384, 768], 
                                'small': [96, 192, 384, 768],
                                'base': [128, 256, 512, 1024], 
                                'large': [192, 384, 768, 1536],
                                'xlarge': [256, 512, 1024, 2048],
                                "resnet18": [64, 128, 256, 512]
                            }
        self.size_dict = {
                            'tiny': [24, 96, 192, 384, 768], 
                            'small': [24, 96, 192, 384, 768],
                            'base': [32, 128, 256, 512, 1024], 
                            'large': [48, 192, 384, 768, 1536],
                            'xlarge': [64, 256, 512, 1024, 2048],
                            "resnet18": [32, 64, 128, 256, 512]
                        }
        # source constructure
        #init attention module
        self.deeplySupEncoderModules = []
        self.deeplySupDecoderModules = []
        self.CBAMs = []
        self.TemporalAttentions = []
        self.embeddableAttentions = []
        self.ChangeSqueezeConv = []
        # module sequence,  F.interpolate、TemporalAttention、AdversialSupervised、concat feature、conv、

        for index in range(self.stageNumber):
            if index == 0:

                if self.isCBAM[index] > 0:
                    if self.isCBAMconcat:
                        self.CBAMs.append(self.CBAModule(self.n_channels*2, **args.CBAM["params"]))
                    else:
                        self.CBAMs.append(self.CBAModule(self.n_channels, **args.CBAM["params"]))
                if self.isTemporalAttention[index] > 0:
                    self.TemporalAttentions.append(self.SpatiotemporalAttentionModule(self.size_change[index],))
                if self.isAttentionLayer[index] > 0:
                    if self.isAttentionConcat:
                        self.embeddableAttentions.append(self.AttentionModule(self.n_channels*2))
                    else:
                        self.embeddableAttentions.append(self.AttentionModule(self.n_channels))
                self.ChangeSqueezeConv.append(self.decoderBlock(self.n_channels*2, self.n_channels))
                if self.isDeeplySupEncoder[index] > 0:
                    self.deeplySupEncoderModules.append(self.deeplySupEncoder(self.n_channels*2, **args.deeplySupervisedEncoder["params"]))
                if self.isDeeplySupDecoder[index] > 0:
                    self.deeplySupDecoderModules.append(self.deeplySupDecoder(self.n_channels, **args.deeplySupervisedDecoder["params"]))
            else:
                if self.isCBAM[index] > 0:
                    if self.isCBAMconcat:
                        self.CBAMs.append(self.CBAModule(self.size_change[index]*2, **args.CBAM["params"]))
                    else:
                        self.CBAMs.append(self.CBAModule(self.size_change[index], **args.CBAM["params"]))
                if self.isTemporalAttention[index] > 0:
                    self.TemporalAttentions.append(self.SpatiotemporalAttentionModule(self.size_change[index],))
                if self.isAttentionLayer[index] > 0:
                    if self.isAttentionConcat:
                        self.embeddableAttentions.append(self.AttentionModule(self.size_change[index]*2))
                    else:
                        self.embeddableAttentions.append(self.AttentionModule(self.size_change[index]))
                self.ChangeSqueezeConv.append(self.decoderBlock(self.size_change[index]*4, self.size_change[index]))
                if self.isDeeplySupEncoder[index] > 0:
                    self.deeplySupEncoderModules.append(self.deeplySupEncoder(self.size_change[index]*2, **args.deeplySupervisedEncoder["params"]))
                if self.isDeeplySupDecoder[index] > 0:
                    self.deeplySupDecoderModules.append(self.deeplySupDecoder(self.size_change[index], **args.deeplySupervisedDecoder["params"]))

        self.CBAMs = nn.ModuleList(self.CBAMs)
        self.TemporalAttentions = nn.ModuleList(self.TemporalAttentions)
        self.embeddableAttentions = nn.ModuleList(self.embeddableAttentions)
        self.ChangeSqueezeConv = nn.ModuleList(self.ChangeSqueezeConv)
        self.deeplySupEncoderModules = nn.ModuleList(self.deeplySupEncoderModules)
        self.deeplySupDecoderModules = nn.ModuleList(self.deeplySupDecoderModules)
        if self.isFeatureFusion == True:
            self.ChangeFinalSqueezeConv = self.decoderBlock(sum(self.size_change[:-1]), self.size_change[-1]*self.encoderNameScale)
        else:
            self.ChangeFinalSqueezeConv = self.decoderBlock(self.size_change[-2], self.size_change[-1]*self.encoderNameScale)
            # self.ChangeFinalSqueezeConv = self.decoderBlock(self.size_change[-2], self.size_change[-1])

        if self.isConcatInput:
            self.ChangeFinalConv = nn.Sequential(self.decoderBlock(self.size_change[-1]*self.encoderNameScale + args.inputChannel*2, self.size_change[-1], **args.decoderBlock["params"]), nn.Conv2d(self.size_change[-1], self.outChannels, kernel_size=1))
        else:
            self.ChangeFinalConv = nn.Sequential(self.decoderBlock(self.size_change[-1]*self.encoderNameScale, self.size_change[-1], **args.decoderBlock["params"]), nn.Conv2d(self.size_change[-1], self.outChannels, kernel_size=1))

        # self.softmax = nn.Softmax(dim=1)
        self.register_hook(self.encoder)
        self.backboneFeatures = []

    def register_hook(self, backbone):
        if "resnet" in self.encoderName:
            def hook(module, input, output):
                self.backboneFeatures.append(output)
            depth = self.backbone_depth[self.encoderName]
            for index, depth_num in enumerate(depth):
                getattr(backbone, "layer"+str(index+1)).register_forward_hook(hook)
        else:
            def hook(module, input, output):
                self.backboneFeatures.append(output)
            depth = self.backbone_depth[self.encoderName]
            for index, depth_num in enumerate(depth):
                backbone.stages[index][depth_num-1].register_forward_hook(hook)

    @property
    def n_channels(self):
        return self.backbone_dims[self.encoderName][-1]

    @property
    def size_change(self):
        size_dict =  copy.deepcopy(self.size_dict)
        size_dict = size_dict[self.encoderName][::-1]
        return size_dict

    def forward(self, x1, x2):
        input_1 = x1
        input_2 = x2
        _ = self.encoder(x1)
        _ = self.encoder(x2)
        # print(_)
        blocks1 = self.backboneFeatures[0:self.stageNumber]
        blocks2 = self.backboneFeatures[self.stageNumber:]
        self.deeplySupervisedFeatures = []
        if self.isBackboneFeaturesSimilarity:
            self.backboneFeaturesGram1 = []
            self.backboneFeaturesGram2 = []
        self.backboneFeatures = []

        FusionFeatures = []
        change = None
        for stage in range(self.stageNumber):
            moduleIdx = stage
            eff_last_1 = blocks1.pop()#.permute(0, 3, 1, 2) 
            eff_last_2 = blocks2.pop()#.permute(0, 3, 1, 2)

            if self.isBackboneFeaturesSimilarity:
                bs, ch, h, w = eff_last_1.shape
                tempSimilarity1 = eff_last_1.reshape(eff_last_1.shape[0], eff_last_1.shape[1], -1)
                tempSimilarity2 = eff_last_2.reshape(eff_last_2.shape[0], eff_last_2.shape[1], -1)
                gram1 = tempSimilarity1 @ tempSimilarity1.permute(0, 2, 1)/(ch*h*w)
                self.backboneFeaturesGram1.append(gram1)
                gram2 = tempSimilarity2 @ tempSimilarity2.permute(0, 2, 1)/(ch*h*w)
                self.backboneFeaturesGram2.append(gram2)

            if self.isDeeplySupEncoder[stage] > 0:
                moduleIdx = self.isDeeplySupEncoder[stage] - 1
                tempDeepFeature = self.deeplySupEncoderModules[moduleIdx](torch.cat([eff_last_1, eff_last_2], dim=1))
                tempDeepFeature = F.interpolate(tempDeepFeature, size=(256,256), mode='bilinear', align_corners=True)
                self.deeplySupervisedFeatures.append(tempDeepFeature)

            if self.isTemporalAttention[stage] > 0:
                moduleRealIdx = self.isTemporalAttention[stage] - 1
                eff_last_1, eff_last_2 = self.TemporalAttentions[moduleRealIdx](eff_last_1, eff_last_2)

            if self.isAttentionLayer[stage] > 0:
                moduleIdx = self.isAttentionLayer[stage] - 1
                if self.isAttentionConcat:
                    eff_last = self.embeddableAttentions[moduleIdx](torch.cat([eff_last_1, eff_last_2], dim=1))
                    sliceNum = eff_last.shape[1]//2
                    eff_last_1, eff_last_2 = eff_last[:,0:sliceNum], eff_last[:,sliceNum:]
                else:
                    eff_last_1, eff_last_2 = self.embeddableAttentions[moduleIdx](eff_last_1), self.embeddableAttentions[moduleIdx](eff_last_2)

            if self.isCBAM[stage] > 0:
                moduleIdx = self.isCBAM[stage] - 1
                if self.isCBAMconcat:
                    eff_last = self.CBAMs[moduleIdx](torch.cat([eff_last_1, eff_last_2], dim=1))
                    sliceNum = eff_last.shape[1]//2
                    eff_last_1, eff_last_2 = eff_last[:,0:sliceNum], eff_last[:,sliceNum:]
                else:
                    eff_last_1, eff_last_2 = self.CBAMs[moduleIdx](eff_last_1), self.CBAMs[moduleIdx](eff_last_2)
            
            if stage == 0:
                change = torch.cat([eff_last_1, eff_last_2], dim=1)
            else:
                change = torch.cat([eff_last_1, eff_last_2, change], dim=1)

            if stage == self.stageNumber-1:
                change = self.ChangeSqueezeConv[stage](change)
                if self.isDeeplySupDecoder[stage] > 0:
                    moduleIdx = self.isDeeplySupDecoder[stage] - 1
                    tempDeepFeature = self.deeplySupDecoderModules[moduleIdx](change)
                    tempDeepFeature = F.interpolate(tempDeepFeature, size=(256,256), mode='bilinear', align_corners=True)
                    self.deeplySupervisedFeatures.append(tempDeepFeature)
                FusionFeatures.append(change)
            else:
                change = self.ChangeSqueezeConv[stage](change)    
                if self.isDeeplySupDecoder[stage] > 0:
                    moduleIdx = self.isDeeplySupDecoder[stage] - 1
                    tempDeepFeature = self.deeplySupDecoderModules[moduleIdx](change)
                    tempDeepFeature = F.interpolate(tempDeepFeature, size=(256,256), mode='bilinear', align_corners=True)
                    self.deeplySupervisedFeatures.append(tempDeepFeature)
                FusionFeatures.append(change)
                change = F.interpolate(change, scale_factor=2., mode='bilinear', align_corners=True)
            
        if self.isFeatureFusion == True:
            for index, down in enumerate(self.downScale):
                FusionFeatures[index] = F.interpolate(FusionFeatures[index], scale_factor=2**(self.stageNumber-index-1), mode='bilinear', align_corners=True)
            fusion = torch.cat(FusionFeatures, dim=1)
        else:
            fusion = change

        change = self.ChangeFinalSqueezeConv(fusion)
        change = F.interpolate(change, scale_factor=self.encoderNameScale, mode='bilinear', align_corners=True)
        change = self.ChangeFinalConv(change)

        return change

def getUCCD(args, **kwargs):
    print("is pretrained: ", args.isBackbonePretrained)
    encoder = getEncoder(pretrained=args.isBackbonePretrained, backbone_scale=args.backboneName, classes=len(args.listOfCategoryNames), in_22k=args.backboneTrainedIn22k, resolution=args.backboneTrainedResolution, **kwargs)
    #To set the parameters of the backbone network as non-trainable.
    if args.backboneTrainable == False:
        for name, value in encoder.named_parameters():
            value.requires_grad = False
    model = UCCD(encoder, args, **kwargs)    
    return model