import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from PIL import Image
import Config
from torchvision import transforms
from collections import Counter
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report


######################################################################
def Create_Data_List(filename):
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            data_list.append(line.strip())

    return data_list



class Image_Custom_Dataset_with_Data_Augmentation(Dataset):
    def __init__(self, data_list, augmentation_aug=False, i=False):
        self.data_list = data_list
        self.augmentation_aug = augmentation_aug

        self.to_Tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.weak_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.strong_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 缩放图像大小到224*224
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(20),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.2, 1.2), value=0, inplace=False)
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        info = self.data_list[index]
        image_path, _, label = info.split("\t")

        image = Image.open(image_path).convert("RGB")
        if self.augmentation_aug:
            o_image = self.to_Tensor(image)
            w_image = self.weak_transform(image)
            s_image = self.strong_transform(image)

            return o_image, int(label), w_image, s_image

        else:
            image = self.to_Tensor(image)

            return image, int(label)


######################################################################
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



######################################################################
def Performance_Metric(y_true, y_pred):
    report = classification_report(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    confu_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = confu_matrix.ravel()
    spe_1 = tn / (tn + fp)
    spe_2 = tp / (fn + tp)
    specificity = (spe_1 + spe_2) / 2

    return (report, acc, recall, precision, specificity, f1, confu_matrix)



def Show_Summary(key, recorder):
    acc, recall, precis, specif, f1 = [], [], [], [], []
    for i, result in enumerate(recorder):
        acc.append(result[1])
        recall.append(result[2])
        precis.append(result[3])
        specif.append(result[4])
        f1.append(result[5])

    print(f"{key} Mean : ACC {np.mean(acc):.6f}, recall {np.mean(recall):.6f}, "
          f"precision {np.mean(precis):.6f}, specificity {np.mean(specif):.6f}, f1 {np.mean(f1):.6f}")
    print(f"ACC   :  {acc}")
    print(f"recall: {recall}")
    print(f"precis: {precis}")
    print(f"specif: {specif}")
    print(f"f1:     {f1}")
    print()



######################################################################
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



######################################################################
def Train_one_epoch(model, epoch, device, optimizer, data_loader, threshold = 0.7, supCon_Function=False, consist_Function=False):
    model.train()
    optimizer.zero_grad()

    all_loss_sum = AverageMeter()
    task_loss_sum = AverageMeter()
    supcon_loss_sum = AverageMeter()
    consist_loss_sum = AverageMeter()
    train_acc_sum, all_n = 0.0, 0
    all_prediction, all_labels = [], []

    task_criterion = nn.CrossEntropyLoss().to(device)
    consistency_criterion = nn.MSELoss().to(device)
    supCon_criterion = SupConLoss(temperature=0.07).to(device)

    for batch, (data, labels, w_data, s_data) in enumerate(data_loader):
        o_data, labels, w_data, s_data = \
            data.float().to(device), labels.long().to(device), w_data.float().to(device), s_data.float().to(device)

        num_batch = labels.shape[0]

        data = torch.cat((o_data, w_data, s_data), dim=0)
        data = torch.cat((data, data), dim=0)
        outputDir = model(data)

        if supCon_Function:
            temp_labels = torch.cat((labels, labels, labels), dim=0)

            supCon_feature = outputDir["supCon_output"]
            f1, f2 = torch.split(supCon_feature, [int(num_batch*3), int(num_batch*3)], dim=0)
            supCon_feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            supCon_loss = Config.supCon_loss * supCon_criterion(supCon_feature, temp_labels) # 0.1
            supcon_loss_sum.update(supCon_loss.item(), int(num_batch*3))

        if consist_Function:
            embedding = outputDir["embedding"]
            embedding, _ = torch.split(embedding, [int(num_batch*3), int(num_batch*3)], dim=0)
            o_embedding, w_embedding, s_embedding = torch.split(embedding, [num_batch, num_batch, num_batch], dim=0)

            logits = outputDir["output"]
            logits, _ = torch.split(logits, [int(num_batch * 3), int(num_batch * 3)], dim=0)
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            confidence, _ = torch.max(softmax_logits, dim=1)
            _, _, s_confidence = torch.split(confidence, [num_batch, num_batch, num_batch], dim=0)

            s_mask = s_confidence > threshold

            o_2_w_loss = consistency_criterion(o_embedding[s_mask], w_embedding[s_mask])
            o_2_s_loss = consistency_criterion(o_embedding[s_mask], s_embedding[s_mask])
            w_2_s_loss = consistency_criterion(w_embedding[s_mask], s_embedding[s_mask])

            consist_loss = Config.consist_loss * (o_2_w_loss + o_2_s_loss + w_2_s_loss) # 0.1
            consist_loss_sum.update(consist_loss.item(), int(num_batch*3))

        labels = torch.cat((labels, labels, labels), dim=0)
        task_output = outputDir["output"]
        task_output, _ = torch.split(task_output, [int(num_batch*3), int(num_batch*3)], dim=0)
        task_loss = task_criterion(task_output, labels)
        task_loss_sum.update(task_loss.item(), int(num_batch*3))

        loss = task_loss
        if supCon_Function:
            loss += supCon_loss
        if consist_Function:
            loss += consist_loss
        all_loss_sum.update(loss.item(), int(num_batch*3))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计
        train_acc_sum += (task_output.argmax(dim=1) == labels).sum().cpu().item()
        all_n += int(num_batch*3)

        _, prediction = torch.max(task_output.data, dim=1)
        all_prediction.extend(prediction.to('cpu'))
        all_labels.extend(labels.to('cpu'))

    all_loss = all_loss_sum.avg
    task_loss = task_loss_sum.avg
    supcon_loss = supcon_loss_sum.avg
    consist_loss = consist_loss_sum.avg
    metrics = Performance_Metric(all_labels, all_prediction)

    return all_loss, task_loss, supcon_loss, consist_loss, metrics




@torch.no_grad()
def Evaluate(model, device, data_loader):
    # 训练模式
    model.eval()

    task_criterion = nn.CrossEntropyLoss().to(device)

    # Metrix
    loss_sum = AverageMeter()
    test_acc_sum, all_n = 0.0, 0
    all_prediction, all_labels = [], []
    for batch, (data, labels) in enumerate(data_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        num_batch = labels.shape[0]

        outputs = model(data)["output"]

        loss = task_criterion(outputs, labels)
        loss_sum.update(loss.item(), num_batch)

        # Metrix Computing
        test_acc_sum += (outputs.argmax(dim=1) == labels).sum().cpu().item()
        all_n += len(labels)

        _, prediction = torch.max(outputs.data, dim=1)
        all_prediction.extend(prediction.to('cpu'))
        all_labels.extend(labels.to('cpu'))

    loss = loss_sum.avg
    metrics = Performance_Metric(all_labels, all_prediction)

    return loss, metrics





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testSetDirList = Create_Data_List(f"{Config.cross_validation_dir}/Preterm_Fold_0_TestSet.txt")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_loader = torch.utils.data.DataLoader(
        Image_Custom_Dataset_with_Data_Augmentation(testSetDirList, transform),
        batch_size=1,
        shuffle=True
    )

    for batch, (data, labels, w_data, s_data) in enumerate(data_loader):
        o_data, labels, w_data, s_data = \
            data.float().to(device), labels.long().to(device), w_data.float().to(device), s_data.float().to(device)

        print(o_data.shape)





