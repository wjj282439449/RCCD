from collections import OrderedDict
from .layers import *
from .layers import DANetModule
import copy
from .convnext import getEncoder
from .resnet import resnet18
import torch.nn.functional as F
from einops.einops import rearrange
__all__ = ['UCCD', 'get_UCCD',]



class Extractor(nn.Module):
    def __init__(self, encoder, args, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.encoderName = args.backboneName

        # self.isAttentionLayer = args.embeddableAttention["isActivateLayer"]
        # self.isAttentionConcat = args.embeddableAttention["isAttentionConcat"]
        # self.AttentionModule = globals()[args.embeddableAttention["name"]]
        #NonLocal2D    DANetModule

        # self.isTemporalAttention = args.interactiveAttention["isActivateLayer"]
        # self.SpatiotemporalAttentionModule = globals()[args.interactiveAttention["name"]]
        #SpatiotemporalAttentionBase  SpatiotemporalAttentionFull  SpatiotemporalAttentionFullNotWeightShared

        # self.isCBAM = args.CBAM["isActivateLayer"]
        # self.isCBAMconcat = args.CBAM["isAttentionConcat"]
        # self.CBAModule = globals()[args.CBAM["name"]]#Easy to extend to more CBAM-like attention mechanisms.

        # self.isDeeplySupEncoder = args.deeplySupervisedEncoder["isActivateLayer"]
        # self.deeplySupEncoder = globals()[args.deeplySupervisedEncoder["name"]]

        # self.isDeeplySupDecoder = args.deeplySupervisedDecoder["isActivateLayer"]
        # self.deeplySupDecoder = globals()[args.deeplySupervisedDecoder["name"]]

        # self.decoderBlock = globals()[args.decoderBlock["name"]]
        self.stageNumber = 4
        if "resnet" in self.encoderName:
            self.encoderNameScale = 2
        else:
            self.encoderNameScale = 4
        # self.isFeatureFusion = args.detectorHead["isFeatureFusion"]
        # self.isConcatInput = args.detectorHead["isConcatInput"]
        # self.outChannels = len(args.listOfCategoryNames)

        self.deeplySupervisedFeatures = []
        # self.isBackboneFeaturesSimilarity = args.isBackboneFeaturesSimilarity
        # if self.isBackboneFeaturesSimilarity:
        #     self.backboneFeaturesGram1 = []
        #     self.backboneFeaturesGram2 = []
        self.backbone_depth = {
                                'tiny': [3, 3, 9, 3], 
                                'small': [3, 3, 27, 3],
                                'base': [3, 3, 27, 3], 
                                'large': [3, 3, 27, 3],
                                'xlarge': [3, 3, 27, 3],
                                "resnet18": [2, 2, 2, 2],
                                "resnet34": [3, 4, 6, 3],
                                "resnet50": [3, 4, 6, 3],
                                "resnet101": [3, 4, 23, 3],
                                "resnet152": [3, 8, 36, 3]
                            }
        self.backbone_dims = {
                                'tiny': [96, 192, 384, 768], 
                                'small': [96, 192, 384, 768],
                                'base': [128, 256, 512, 1024], 
                                'large': [192, 384, 768, 1536],
                                'xlarge': [256, 512, 1024, 2048],
                                "resnet18": [64, 128, 256, 512],
                                "resnet34": [64, 128, 256, 512],
                                "resnet50": [256, 512, 1024, 2048],
                                "resnet101": [256, 512, 1024, 2048],
                                "resnet152": [256, 512, 1024, 2048]
                            }
        self.size_dict = {
                            'tiny': [24, 96, 192, 384, 768], 
                            'small': [24, 96, 192, 384, 768],
                            'base': [32, 128, 256, 512, 1024], 
                            'large': [48, 192, 384, 768, 1536],
                            'xlarge': [64, 256, 512, 1024, 2048],
                            "resnet18": [32, 64, 128, 256, 512],
                            "resnet34": [32, 64, 128, 256, 512],
                            "resnet50": [128, 256, 512, 1024, 2048],
                            "resnet101": [128, 256, 512, 1024, 2048],
                            "resnet152": [128, 256, 512, 1024, 2048]
                        }
        # # source constructure

        # self.softmax = nn.Softmax(dim=1)
        # self.register_hook(self.encoder)
        # self.backboneFeatures = []

    # def register_hook(self, backbone):
    #     if "resnet" in self.encoderName:
    #         def hook(module, input, output):
    #             self.backboneFeatures.append(output)
    #         depth = self.backbone_depth[self.encoderName]
    #         for index, depth_num in enumerate(depth):
    #             getattr(backbone, "layer"+str(index+1)).register_forward_hook(hook)
    #     else:
    #         def hook(module, input, output):
    #             self.backboneFeatures.append(output)
    #         depth = self.backbone_depth[self.encoderName]
    #         for index, depth_num in enumerate(depth):
    #             backbone.stages[index][depth_num-1].register_forward_hook(hook)

    @property
    def n_channels(self):
        return self.backbone_dims[self.encoderName][-1]

    @property
    def size_change(self):
        size_dict =  copy.deepcopy(self.size_dict)
        size_dict = size_dict[self.encoderName][::-1]
        return size_dict


    def similarity(self, a, b, similarityType="cosine"):
        if similarityType == "l1":
            pass
        elif similarityType == "l2":
            pass
        elif similarityType == "cosine":
            return nn.functional.cosine_similarity(a, b, dim=1)
        else:
            return nn.functional.cosine_similarity(a, b, dim=1)


    def forward(self, x1, x2, mask=None):
        # input_1 = x1
        # input_2 = x2
        # self.backboneFeatures = []
        # _ = self.encoder(x1)
        # _ = self.encoder(x2)
        # print(_)
        blocks1 = self.encoder(x1)
        blocks2 = self.encoder(x2)
        # self.deeplySupervisedFeatures = []
        self.backboneFeaturesGram1 = []
        self.backboneFeaturesGram2 = []
        # self.backboneFeatures = []

        similarityIndex = None
        gramSimilarityIndex = None
        for stage in range(self.stageNumber):
            eff_last_1 = blocks1.pop()#.permute(0, 3, 1, 2) 
            eff_last_2 = blocks2.pop()#.permute(0, 3, 1, 2)
            # print(eff_last_1.shape)
            # print("eff_last_2", eff_last_2.sum())
            bs, ch, h, w = eff_last_1.shape
            if mask != None:
                # print("stage",stage)
                # print(mask.unsqueeze(dim=1).shape)
                maskInterpolate = F.interpolate(mask.unsqueeze(dim=1), size=(h,w), mode='nearest')#, align_corners=True)
                newMask = []
                for eachMask in maskInterpolate.squeeze(dim=1):
                    newMask.append(eachMask.repeat(ch,1,1))
                maskInterpolate = torch.stack(newMask)
                # print(maskInterpolate.shape)
                # print(maskInterpolate.sum())
                eff_last_1[maskInterpolate == 1] = 0
                eff_last_2[maskInterpolate == 1] = 0
                # print("@@@@@@@@@@@@")
            # print("eff_last_2_masked", eff_last_2.sum())
            # print(eff_last_1.reshape(bs, -1).shape)
            # print(eff_last_2.reshape(bs, -1).shape)
            singleFeatureSimilarity = self.similarity(eff_last_1.reshape(bs, -1), eff_last_2.reshape(bs, -1))#.mean(dim=1)
            if stage == 0:
                similarityIndex = singleFeatureSimilarity
            else:
                similarityIndex = similarityIndex + singleFeatureSimilarity
            # if self.isBackboneFeaturesSimilarity:
            bs, ch, h, w = eff_last_1.shape
            tempSimilarity1 = eff_last_1.reshape(eff_last_1.shape[0], eff_last_1.shape[1], -1)
            tempSimilarity2 = eff_last_2.reshape(eff_last_2.shape[0], eff_last_2.shape[1], -1)
            gram1 = tempSimilarity1 @ tempSimilarity1.permute(0, 2, 1)/(ch*h*w)
            # self.backboneFeaturesGram1.append(gram1)
            gram2 = tempSimilarity2 @ tempSimilarity2.permute(0, 2, 1)/(ch*h*w)
            # self.backboneFeaturesGram2.append(gram2)
            singleGramFeatureSimilarity = self.similarity(gram1.reshape(bs, -1), gram2.reshape(bs, -1))#.mean(dim=1)
            if stage == 0:
                gramSimilarityIndex = singleGramFeatureSimilarity
            else:
                gramSimilarityIndex = gramSimilarityIndex + singleGramFeatureSimilarity
        # print()
        similarityIndex = similarityIndex / self.stageNumber
        gramSimilarityIndex = gramSimilarityIndex / self.stageNumber
        # print("similarityIndex", similarityIndex)
        # print("gramSimilarityIndex", gramSimilarityIndex)
        # if mask != None:
        #     print("@@@@@@@@@@@@")
                
        return similarityIndex.unsqueeze(dim=1), gramSimilarityIndex.unsqueeze(dim=1)

def getExtractor(args, **kwargs):
    print("is pretrained: ", args.isBackbonePretrained)
    encoder = getEncoder(pretrained=args.isBackbonePretrained, backbone_scale=args.backboneName, classes=len(args.listOfCategoryNames), in_22k=args.backboneTrainedIn22k, resolution=args.backboneTrainedResolution, **kwargs)
    #To set the parameters of the backbone network as non-trainable.
    if args.backboneTrainable == False:
        for name, value in encoder.named_parameters():
            value.requires_grad = False
    model = Extractor(encoder, args, **kwargs)    
    return model