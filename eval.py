import os
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import L1Loss, BCELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Adam, Adadelta, Adagrad, Adamax, ASGD, Rprop, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from models.uccd import getUCCD
from models.loss import UCCDloss
from utils.dataloader import WSIDataset
import numpy as np
import random
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from multiprocessing import Process
import importlib
import copy
from torchmetrics import ConfusionMatrix
from PIL import Image

#determine pytorch version 
# torchVersion = torch.__version__
# torchHasCuda = True if "cu" in torchVersion else False
# torchIsTwoPointZero = True if "2." == torchVersion[0:2] else False
ROOTDIR = {
    "LEVIR-CD+": r"datasets/LEVIR-CD+",
    # "S2Looking": r"datasets/S2Looking",
    # "SYSU-CD": r"../../datasets/SYSU-CD",
    # "BANDON": r"../../datasets/BANDON",
    # "DSIFN-CD": r"../../datasets/DSIFN-CD",
}
saveDatasetName = "-".join(ROOTDIR.keys())
result_dir = "./data/results/"+saveDatasetName
model_path = r"./data/checkpoints/LEVIR-CD+_RCCD-base_test/UCCD-tiny_epoch-189_trainF1-93.44_valF1-86.49_unchange-change.pth"
# model_path = r"./data/checkpoints/S2Looking_RCCD-base_test/UCCD-tiny_epoch-77_trainF1-81.13_valF1-65.98_unchange-change.pth"

def processingExtendArgs(args):
    #Some keys require auto generation during processing.
    args.accumulateSteps = args.computeBatchSize//args.gpuBatchSize
    args.num_classes = len(args.listOfCategoryNames)
    #dataset path setting, highlight one and close other
    #train-LEVIR-CD_val-LEVIR-CD
    args.saveDatasetName = "_".join([("-".join([stage,] + list(args.dataset[stage]["params"]["root_dir"].keys()))) for stage in args.modelStage])
    args.saveDir = os.path.join(args.checkpointPath, args.saveDatasetName)
    args.resultDir = os.path.join(args.checkpointPath, args.saveDatasetName)
    # args.modelStage.sort()
    args.plotMetrics.sort()
    return args

def loadConfig(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def parserConfigurator():
    parser = argparse.ArgumentParser(description='UCCD source code')
    #setting the arguments
    parser.add_argument('-c', '--configPath', required=True, type=str, help="Path to the configuration file")
    #parsing args
    args = parser.parse_args()
    configSource = loadConfig(args.configPath)
    config = {}
    #parse the first level
    for key,value in configSource.items():
        config.update(value)
    #Update the configuration with the values from the command-line arguments, giving them higher priority.
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    args.__dict__.update(**config)
    args = processingExtendArgs(args)
    return args

def setupRandomSeed(args):
    random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)
    torch.cuda.manual_seed(args.randomSeed)
    torch.cuda.manual_seed_all(args.randomSeed)
    np.random.seed(args.randomSeed)
    torch.backends.cudnn.deterministic = args.cudnnDeterministic
    torch.backends.cudnn.benchmark = args.cudnnBenchmark
    torch.set_float32_matmul_precision(args.float32MatmulPrecision)

def dataloaderCreator(root_dir, mode="train", args=None):
    dataset = WSIDataset(root_dir=root_dir, mode=mode, args=args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.gpuBatchSize, **args.dataloader[mode]["params"])
    return dataset, dataloader

def save_checkpoints(model, step, args):
    if os.path.exists(args.saveDir) == False: 
        os.makedirs(args.saveDir)
    filename = args.modelName + "-" + args.backboneName + "_" + str(step) + "_" + "-".join(args.listOfCategoryNames) + ".pth"
    torch.save(model.state_dict(), os.path.join(args.saveDir, filename))
    print("        Save checkpoint {} to {}. \n".format(step.split("_")[0], filename))

def train(model, index, trainDataLoader, criterion, args, optimizer):
    optimizer.zero_grad()
    mode = "train"
    showRowLen = args.num_classes + 1
    #metrics setting
    # args.epochCalculateMetrics = ["TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "IoU", "TNR", "F1", "Kappa", "Loss"]
    epochMetrics = {key: [0.0]*showRowLen for key in args.epochCalculateMetrics}
    if args.num_classes == 1:
        torchConfusionMatrix = ConfusionMatrix(task="binary", num_classes=args.num_classes).to(device)
    elif args.num_classes >=2:
        torchConfusionMatrix = ConfusionMatrix(task="multiclass", num_classes=args.num_classes).to(device)
    # model = torch.compile(model)
    model.train()
    accumulateCounter = 0
    for batch_idx, (img1, img2, label1, label2, labelChange, dir) in tqdm(enumerate(trainDataLoader)):  
        # 
        img1, img2, label1, label2, labelChange = img1.float(), img2.float(), label1.float(), label2.float(), labelChange.float()       
        if args.useCuda:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labelChange = labelChange.to(device)
            
        output_change = model(img1, img2)
        loss = criterionCalculator(output_change, labelChange, model.deeplySupervisedFeatures, criterion)
        (loss/args.accumulateSteps).backward()

        if (accumulateCounter+1)%args.accumulateSteps == 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulateCounter = 0
        else:
            accumulateCounter = accumulateCounter + 1 

        output_change = F.softmax(output_change, dim=1)
        _, predicted_labels = torch.max(output_change.detach(), dim=1)
        torchConfusionMatrix.update(predicted_labels.detach(), labelChange.detach())
        epochMetrics["Loss"][args.num_classes] += loss.detach().cpu().item()
    epochMetrics["Loss"] = [round(epochMetrics["Loss"][args.num_classes] / len(trainDataLoader), 4)] * (args.num_classes+1)
    calculateMatrix(epochMetrics, torchConfusionMatrix.compute().cpu().numpy(), args.meanMetrics)
    printTable(epochMetrics, args.epochsDisplayMetrics, args.listOfCategoryNames, mode)
    returnMetrics = {key: epochMetrics[key][-2] for key in args.plotMetrics}
    torchConfusionMatrix.reset()
    return returnMetrics

def validate(model, index, valDataLoader, criterion, args):
    mode = "val"
    showRowLen = args.num_classes + 1
    # args.epochCalculateMetrics = ["TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "IoU", "TNR", "F1", "Kappa", "Loss"]
    epochMetrics = {key: [0]*showRowLen for key in args.epochCalculateMetrics}
    if args.num_classes == 1:
        torchConfusionMatrix = ConfusionMatrix(task="binary", num_classes=args.num_classes).to(device)
    elif args.num_classes >=2:
        torchConfusionMatrix = ConfusionMatrix(task="multiclass", num_classes=args.num_classes).to(device)
    dir_fold = args.resultDir + os.sep + str(index) 
    # model = torch.compile(model)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, label1, label2, labelChange, dir) in tqdm(enumerate(valDataLoader)): 
            img1, img2, labelChange = img1.float(), img2.float(), labelChange.float()
            if args.useCuda:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labelChange = labelChange.to(device)

            # reset gradients
            output_change = model(img1, img2)
            # loss = criterionCalculator(output_change, labelChange, model.deeplySupervisedFeatures, criterion)

            output_change = F.softmax(output_change, dim=1)
            _, predicted_labels = torch.max(output_change.detach(), dim=1)
            torchConfusionMatrix.update(predicted_labels.detach(), labelChange.detach())
            # epochMetrics["Loss"][args.num_classes] += loss.detach().cpu().item()
            result_change = [result_dir + os.sep + i + '.png' for i in dir]
            
            # result_change = [args.resultDir + os.sep + str(index) + os.sep + i + '_change.png' for i in dir]
            output_change = output_change[:,1,:,:].detach().float()
            output_change[output_change>=0.5] = 255
            output_change[output_change<0.5] = 0
            output_change = output_change.cpu().numpy()
            for i in range(output_change.shape[0]):
                temp_output_change = Image.fromarray(output_change[i])
                temp_output_change.convert("RGB").save(result_change[i])
        # epochMetrics["Loss"] = [round(epochMetrics["Loss"][args.num_classes] / len(valDataLoader), 4)] * (args.num_classes+1)
        calculateMatrix(epochMetrics, torchConfusionMatrix.compute().cpu().numpy(), args.meanMetrics)
        printTable(epochMetrics, args.epochsDisplayMetrics, args.listOfCategoryNames, mode)
    returnMetrics = {key: epochMetrics[key][-2] for key in args.plotMetrics}
    torchConfusionMatrix.reset()
    return returnMetrics

def criterionCalculator(pred, label, deeplySupervisedFeatures, criterion):
    count = 0
    loss = 0.0
    for name, value in criterion.items():
        count += 1
        loss_temp = 0.0
        func = value["instantiation"]
        factor = value["coefficient"]
        if name == "focalloss":
            loss_temp = func(pred, label.long())#.detach()
        elif name == "uccdloss" and deeplySupervisedFeatures != []:
            batchSize = label.size(0)
            for eachLayer in  deeplySupervisedFeatures:
                unchange = func(eachLayer[:,0], 1-label.view(batchSize, -1).mean(dim=1)).mean()
                change = func(eachLayer[:,1], label.view(batchSize, -1).mean(dim=1)).mean()
                loss_temp += (unchange+change)/2
            loss_temp = loss_temp/len(deeplySupervisedFeatures)
        elif name == "segloss":
            loss_temp = func(pred,label.long()).mean()
        elif name == "diceloss":
            loss_temp = func(pred[:,1,:,:], label).mean()#.detach()
        loss += factor * loss_temp
    return  loss/count

def confusionMatrixColletor(torchConfusionMatrix, metrics, pred, label, threshold=0.5):
    # for i in range(len(label)):
    print("", torchConfusionMatrix.update(pred, label))
    print(torchConfusionMatrix.compute())

    # singlePred = (pred >= threshold).byte()
    # singleLabel = (label > threshold).byte()
    # plus = singlePred + singleLabel
    # FN = (singlePred < singleLabel).sum()
    # FP = (singlePred > singleLabel).sum()
    # TP = (plus == 2).sum()
    # TN = (plus == 0).sum()
    # metrics["TN"] = metrics["TN"]+TN.cpu().item()
    # metrics["FP"] = metrics["FP"]+FP.cpu().item()
    # metrics["FN"] = metrics["FN"]+FN.cpu().item()
    # metrics["TP"] = metrics["TP"]+TP.cpu().item()
    # return 

def calculateMatrix(metrics, torchCM, meanMetrics):
    nClass = len(metrics["Acc"])-1
    metrics["Acc"] = [round(np.diag(torchCM).sum() / np.sum(torchCM)*100, 3)] * len(metrics["Acc"])
    total = torchCM.sum()
    observed_accuracy = np.diag(torchCM).sum() / total
    expected_accuracy = (torchCM.sum(axis=0) * torchCM.sum(axis=1)).sum() / (total**2)
    metrics["Kappa"] = [round((observed_accuracy - expected_accuracy) / (1 - expected_accuracy) * 100, 3)] * len(metrics["Acc"])

    precision = np.diag(torchCM) / np.sum(torchCM, axis=0)
    recall = np.diag(torchCM) / np.sum(torchCM, axis=1)
    metrics["Pre"][0:nClass] = list(precision)
    metrics["Rec"][0:nClass] = list(recall)
    metrics["F1"][0:nClass] = list(2 * (precision * recall) / (precision + recall))
    metrics["IoU"][0:nClass] = list(np.diag(torchCM) / (np.sum(torchCM, axis=1) + np.sum(torchCM, axis=0) - np.diag(torchCM)))

    for eachMean in meanMetrics:
        metrics[eachMean][-1] = sum(metrics[eachMean][0:nClass])/nClass
        metrics[eachMean] = [round(i*100, 3) for i in metrics[eachMean]]
    # metrics["TNR"][0:nClass] = round(list(recall), 3)
    # for i in range(len(metrics["Acc"])-1):
    #     TN = metrics["TN"][i]
    #     FP = metrics["FP"][i]
    #     FN = metrics["FN"][i]
    #     TP = metrics["TP"][i]
    #     metrics["Acc"][i] = (TP+TN)/(TP+TN+FP+FN)
    #     metrics["Pre"][i] = round(TP/(TP+FP+0.0001)*100, 3)
    #     metrics["Rec"][i] = round(TP/(TP+FN+0.0001)*100, 3)
    #     metrics["IoU"][i] = round(TP/(TP+FP+FN)*100, 3)
    #     metrics["TNR"][i] = round(TN/(TN+FP)*100, 3)
    #     metrics["F1"][i] = round(2*TP/(2*TP+FP+FN)*100, 3)
    #     Pe = ((TP+FP)*(TP+FN)+(TN+FN)*(TN+FP))/(TP+FP+TN+FN)/(TP+FP+TN+FN)
    #     metrics["Kappa"][i] = round((metrics["Acc"][i]-Pe)/(1-Pe)*100, 3)
    #     metrics["Acc"][i] = round(metrics["Acc"][i]*100, 3)
    # for k,v in metrics.items():
    #     metrics[k][-1] = sum(metrics[k][0:-1])/(len(metrics["Acc"])-1)

def printTable(metrics, displayMetrics, labelName, mode):
    labelNameCopy = labelName + ["average"]
    table = PrettyTable([mode,] + displayMetrics)
    for i in range(len(metrics["Acc"])):
        row = [labelNameCopy[i]] + [metrics[key][i] for key in displayMetrics]
        table.add_row(row)
    print(table)

def saveMetricsPlot(metrics, args):
    stageName = list(metrics.keys())
    stageName.sort()
    epochs = len(list(metrics.values())[0])

    #draw plot
    x = range(epochs)
    len_stage = 1
    len_metric = len(args.plotMetrics)
    fig, axs = plt.subplots(len_metric, len_stage, dpi=600, figsize=(10,10))#
    for index, each_subplot in enumerate(args.plotMetrics):
        if each_subplot == "Loss":
            axs[index].set_ylim(0, 0.1)
        else:
            axs[index].set_ylim(0, 100)
        color = ['b','r','g','c','m','y','k','w']
        for each_stage in args.modelStage:
            axs[index].plot(x, metrics[each_subplot + " " + each_stage], label=each_stage)
        axs[index].set_title(each_subplot)
        axs[index].legend()
    fig.savefig(args.saveDir + os.sep + "results.png", bbox_inches="tight")
    plt.close("all")

def creatPlotProcess(metric_record, args):
    plotProcess = Process(target=saveMetricsPlot, args=(metric_record, args))
    plotProcess.start()
    plotProcess.join()
 
def criterionCreator(args):
    criterion = copy.deepcopy(args.criterion)
    losses = criterion.keys()
    for loss in losses:
        criterion[loss]["instantiation"] = globals()[criterion[loss]["name"]]().to(device)
    return criterion

def main():

    args = parserConfigurator()
    setupRandomSeed(args)
    
    global device 
    device = torch.device(args.modelRunDevice) if torch.cuda.is_available() else torch.device("cpu")
        
    #backbone support type: ConvNeXt-'tiny','small','base',and 'resnet18'
    model = getUCCD(args)#.cuda()
    modelParams = filter(lambda p: p.requires_grad, model.parameters())

    print(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    model.zero_grad()
    
    optimizer = globals()[args.optimizer["name"]](modelParams, **args.optimizer["params"])
        
    valDataset, valDataloader = dataloaderCreator(**args.dataset["val"]["params"], args=args)

    #loss function of the composite
    criterion = criterionCreator(args)

    metric_record = {(args.plotMetrics[i//len(args.modelStage)]+" "+args.modelStage[i%len(args.modelStage)]):[] for i in range(len(args.modelStage)*len(args.plotMetrics))}
    
    # for i in range(args.startEpoch, args.totalEpochs):

    val_avg_metric = validate(model, 90, valDataloader, criterion, args)
    for key in val_avg_metric.keys():
        metric_record[key+" val"].append(val_avg_metric[key])


if __name__ == "__main__":
    main()
