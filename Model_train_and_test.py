import torch
from torch import nn, optim
import torchvision.transforms as transforms
from Model.Representation_constraint import Create_Data_List, Image_Custom_Dataset_with_Data_Augmentation, \
    Train_one_epoch, Evaluate, Show_Summary
from Model.Model import SupCon_Inception_v3
import numpy as np
import pickle
import os
import Config
import warnings
import argparse



def CrossAge_main(modelName, device, saveRecordFunction, supCon_Function, consist_Function):
    print(f"modelName: {modelName}, device: {device}, saveRecordFunction: {saveRecordFunction}, "
          f"supCon_Function:{supCon_Function}, consist_Function: {consist_Function}")

    # 预处理部分
    crossAgeKeyList = Config.num_k

    termMetricList = []
    pretermMetricList = []
    allMetricList = []
    for i, testKey in enumerate(range(crossAgeKeyList)):
        testDirList = Create_Data_List(f"{Config.cross_validation_dir}/Term_Fold_{i}_TrainSet.txt") + \
                      Create_Data_List(f"{Config.cross_validation_dir}/Preterm_Fold_{i}_TestSet.txt")

        testTermDirList = [item for item in testDirList if "Term" in item]
        testPretermDirList = [item for item in testDirList if "Preterm" in item]

        test_termDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testTermDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_pretermDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testPretermDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_allDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        trainTermDirList = Create_Data_List(f"{Config.cross_validation_dir}/Term_Fold_{i}_TrainSet.txt")
        trainPretermDirList = Create_Data_List(f"{Config.cross_validation_dir}/Preterm_Fold_{i}_TrainSet.txt")
        trainDirList = trainTermDirList + trainPretermDirList

        # train dataset
        train_dataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(trainDirList, True),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Create Model
        model = SupCon_Inception_v3().to(device)

        #
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

        trainRecorder = []
        termRecorder = []
        pretermRecorder = []
        allRecorder = []
        for epoch in range(1, Config.epoch + 1):
            train_all_loss, train_task_loss, train_supcon_loss, train_consist_loss, train_metric = \
                    Train_one_epoch(model, epoch, device, optimizer, train_dataloader, supCon_Function=supCon_Function, consist_Function=consist_Function)

            term_loss, term_metric = Evaluate(model, device, test_termDataloader)
            preterm_loss, preterm_metric = Evaluate(model, device, test_pretermDataloader)
            all_loss, all_metric = Evaluate(model, device, test_allDataloader)

            print(f"epoch: {epoch}/{Config.epoch}")
            print(f"Train: All loss: {train_all_loss:.6f}, task_loss: {train_task_loss:.6f}, supcon_loss: {train_supcon_loss:.6f}, consist_loss: {train_consist_loss:.6f}")
            print(f"Test : All loss: {all_loss:.6f}")

            print(f"Train   ACC: {train_metric[1]:.6f}, recall: {train_metric[2]:.6f}, precis: {train_metric[3]:.6f}, specif: {train_metric[4]:.6f}, f1: {train_metric[5]:.6f}")
            print(f"Term T: ACC: {term_metric[1]:.6f}, recall: {term_metric[2]:.6f}, precis: {term_metric[3]:.6f}, specif: {term_metric[4]:.6f}, f1: {term_metric[5]:.6f}")
            print(f"Pret T: ACC: {preterm_metric[1]:.6f}, recall: {preterm_metric[2]:.6f}, precis: {preterm_metric[3]:.6f}, specif: {preterm_metric[4]:.6f}, f1: {preterm_metric[5]:.6f}")
            print(f"All  T: ACC: {all_metric[1]:.6f}, recall: {all_metric[2]:.6f}, precis: {all_metric[3]:.6f}, specif: {all_metric[4]:.6f}, f1: {all_metric[5]:.6f}\n")

            trainRecorder.append((train_task_loss, train_metric))
            termRecorder.append((term_loss, term_metric))
            pretermRecorder.append((preterm_loss, preterm_metric))
            allRecorder.append((all_loss, all_metric))

            if epoch % 5 == 0:
                torch.save(
                    model.state_dict(),
                    f"Weights/CrossAge_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Epoch_{epoch}_Fold_{i}.pth"
                )

        # save weights
        torch.save(
            model.state_dict(),
            f"Weights/CrossAge_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Fold_{i}_Final.pth"
        )

        if saveRecordFunction:
            saveFile = f"CrossAge_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Fold_{i}.pickle"
            recorder = {
                "train": trainRecorder,
                "term": termRecorder,
                "preterm": pretermRecorder,
                "all": allRecorder
            }
            with open(saveFile, "wb") as file:
                pickle.dump(recorder, file)

        # test
        term_loss, term_metric = Evaluate(model, device, test_termDataloader)
        preterm_loss, preterm_metric = Evaluate(model, device, test_pretermDataloader)
        all_loss, all_metric = Evaluate(model, device, test_allDataloader)

        termMetricList.append(term_metric)
        pretermMetricList.append(preterm_metric)
        allMetricList.append(all_metric)

    print("\n")
    print("Summary")
    print("CrossAge")
    Show_Summary("term", termMetricList)
    Show_Summary("pret", pretermMetricList)
    Show_Summary("all ", allMetricList)



def Cross_Scene_Devices_main(keyList, modelName, device, saveRecordFunction, supCon_Function, consist_Function):
    print(f"modelName: {modelName}, device: {device}, saveRecordFunction: {saveRecordFunction}, "
          f"supCon_Function:{supCon_Function}, consist_Function: {consist_Function}")

    if "Nanfang" in keyList:
        saveKey = "CrossScene"
    else:
        saveKey = "CrossDevices"

    termMetricList = []
    pretermMetricList = []
    allMetricList = []
    for i, testKey in enumerate(keyList):
        testDirList = Create_Data_List(f"{Config.cross_validation_dir}/{testKey}_Fold_{i}_TrainSet.txt") + \
                      Create_Data_List(f"{Config.cross_validation_dir}/{testKey}_Fold_{i}_TestSet.txt")

        testTermDirList = [item for item in testDirList if "Term" in item]
        testPretermDirList = [item for item in testDirList if "Preterm" in item]

        test_termDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testTermDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_pretermDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testPretermDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_allDataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(testDirList, False),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4
        )

        trainDirList = []
        for key in keyList:
            if key == testKey:
                continue

            trainDirList += Create_Data_List(f"{Config.cross_validation_dir}/{testKey}_Fold_{i}_TrainSet.txt") + \
                            Create_Data_List(f"{Config.cross_validation_dir}/{testKey}_Fold_{i}_TestSet.txt")

        # train dataset
        train_dataloader = torch.utils.data.DataLoader(
            Image_Custom_Dataset_with_Data_Augmentation(trainDirList, True),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=1
        )

        # Create Model
        model = SupCon_Inception_v3().to(device)

        #
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

        trainRecorder = []
        termRecorder = []
        pretermRecorder = []
        allRecorder = []
        for epoch in range(1, Config.epoch + 1):
            train_all_loss, train_task_loss, train_supcon_loss, train_consist_loss, train_metric = \
                    Train_one_epoch(model, epoch, device, optimizer, train_dataloader, supCon_Function=supCon_Function, consist_Function=consist_Function)

            term_loss, term_metric = Evaluate(model, device, test_termDataloader)
            preterm_loss, preterm_metric = Evaluate(model, device, test_pretermDataloader)
            all_loss, all_metric = Evaluate(model, device, test_allDataloader)

            print(f"epoch: {epoch}/{Config.epoch}")
            print(f"Train: All loss: {train_all_loss:.6f}, task_loss: {train_task_loss:.6f}, supcon_loss: {train_supcon_loss:.6f}, consist_loss: {train_consist_loss:.6f}")
            print(f"Test : All loss: {all_loss:.6f}")

            print(f"Train   ACC: {train_metric[1]:.6f}, recall: {train_metric[2]:.6f}, precis: {train_metric[3]:.6f}, specif: {train_metric[4]:.6f}, f1: {train_metric[5]:.6f}")
            print(f"Term T: ACC: {term_metric[1]:.6f}, recall: {term_metric[2]:.6f}, precis: {term_metric[3]:.6f}, specif: {term_metric[4]:.6f}, f1: {term_metric[5]:.6f}")
            print(f"Pret T: ACC: {preterm_metric[1]:.6f}, recall: {preterm_metric[2]:.6f}, precis: {preterm_metric[3]:.6f}, specif: {preterm_metric[4]:.6f}, f1: {preterm_metric[5]:.6f}")
            print(f"All  T: ACC: {all_metric[1]:.6f}, recall: {all_metric[2]:.6f}, precis: {all_metric[3]:.6f}, specif: {all_metric[4]:.6f}, f1: {all_metric[5]:.6f}\n")

            trainRecorder.append((train_task_loss, train_metric))
            termRecorder.append((term_loss, term_metric))
            pretermRecorder.append((preterm_loss, preterm_metric))
            allRecorder.append((all_loss, all_metric))

            if epoch % 5 == 0:
                torch.save(
                    model.state_dict(),
                    f"Weights/{saveKey}_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Epoch_{epoch}_Fold_{i}.pth"
                )

        # save weights
        torch.save(
            model.state_dict(),
            f"Weights/{saveKey}_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Fold_{i}_Final.pth"
        )

        if saveRecordFunction:
            saveFile = f"{saveKey}_supCon({supCon_Function})_consist({consist_Function})_KeyMixed_ModelName({modelName})_Fold_{i}.pickle"
            recorder = {
                "train": trainRecorder,
                "term": termRecorder,
                "preterm": pretermRecorder,
                "all": allRecorder
            }
            with open(saveFile, "wb") as file:
                pickle.dump(recorder, file)

        # test
        term_loss, term_metric = Evaluate(model, device, test_termDataloader)
        preterm_loss, preterm_metric = Evaluate(model, device, test_pretermDataloader)
        all_loss, all_metric = Evaluate(model, device, test_allDataloader)

        termMetricList.append(term_metric)
        pretermMetricList.append(preterm_metric)
        allMetricList.append(all_metric)

    print("\n")
    print("Summary")
    print(f"{keyList}")
    Show_Summary("term", termMetricList)
    Show_Summary("pret", pretermMetricList)
    Show_Summary("all ", allMetricList)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some variables.')
    parser.add_argument('--GPU', type=str, default="0")
    parser.add_argument('--supCon', action='store_true', default=True)
    parser.add_argument('--consist', action='store_true', default=True)
    args = parser.parse_args()

    # Setting
    saveRecordFunction = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    supCon_Function = args.supCon
    consist_Function = args.consist
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelName = "SupCon_Inception_v3"

    # senceKeyList = ["Nanfang", "Baoan", "Sanyuan", "SZZ"]
    # Cross_Scene_Devices_main(senceKeyList, modelName, device, saveRecordFunction, supCon_Function, consist_Function)

    # deviceKeyList = ["T40", "IDS", "C48M"]
    # Cross_Scene_Devices_main(deviceKeyList, modelName, device, saveRecordFunction, supCon_Function, consist_Function)

    CrossAge_main(modelName, device, saveRecordFunction, supCon_Function, consist_Function)




