from dataLoader import make_loader
import torch as t
from torch import nn, optim
from loss import Loss
from model import CenterNet
import os


def train_epoch(model, criterion, train_loader, current_epoch, device_ids, optimizer, log_step, epoch):
    model.train()
    steps = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(device_ids[0])
        l_train_cuda = l_train.cuda(device_ids[0])
        train_output = model(d_train_cuda)
        train_loss, train_heatmap_loss, train_size_loss, train_offset_loss = criterion(train_output, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if current_step % log_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_heatmap_loss:%.5f, train_size_loss:%.5f, train_offset_loss:%.5f, train_loss:%.5f" % (current_epoch, epoch, current_step, steps, train_heatmap_loss.item(), train_size_loss.item(), train_offset_loss.item(), train_loss.item()))
        current_step += 1
    print("saving epoch model......")
    t.save(model.module.state_dict(), "epoch.pth")
    return model


def valid_epoch(model, criterion, valid_loader, current_epoch, device_ids):
    global best_valid_loss
    model.eval()
    steps = len(valid_loader)
    accum_loss = 0
    accum_heatmap_loss = 0
    accum_size_loss = 0
    accum_offset_loss = 0
    for d_valid, l_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(device_ids[0])
        l_valid_cuda = l_valid.cuda(device_ids[0])
        with t.no_grad():
            valid_output = model(d_valid_cuda)
            valid_loss, valid_heatmap_loss, valid_size_loss, valid_offset_loss = criterion(valid_output, l_valid_cuda)
            accum_loss += valid_loss.item()
            accum_heatmap_loss += valid_heatmap_loss.item()
            accum_size_loss += valid_size_loss.item()
            accum_offset_loss += valid_offset_loss.item()
    avg_loss = accum_loss / steps
    avg_heatmap_loss = accum_heatmap_loss / steps
    avg_size_loss = accum_size_loss / steps
    avg_offset_loss = accum_offset_loss / steps
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        print("saving best model......")
        t.save(model.module.state_dict(), "best.pth")
    print("##########valid epoch:%d##########" % (current_epoch,))
    print("valid_heatmap_loss:%.5f, valid_size_loss:%.5f, valid_offset_loss:%.5f, valid_loss:%.5f" % (avg_heatmap_loss, avg_size_loss, avg_offset_loss, avg_loss))
    return model


def main(num_classes, class_names, lamda_size, lamda_off, alpha, beta, init_lr, final_lr, epoch, batch_size, data_root_dir, img_size, sigma_iou_thresh, num_workers, R, log_step, CUDA_VISIBLE_DEVICES):
    global  best_valid_loss
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
    best_valid_loss = float("inf")
    model = CenterNet(num_classes)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    criterion = Loss(lamda_size, lamda_off, num_classes, alpha, beta).cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr)
    lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=final_lr)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(data_root_dir, True, img_size ,num_classes, class_names, R, sigma_iou_thresh, batch_size, num_workers)
        valid_loader = make_loader(data_root_dir, False, img_size, num_classes, class_names, R, sigma_iou_thresh, batch_size, num_workers)
        model = train_epoch(model, criterion, train_loader, current_epoch, device_ids, optimizer, log_step, epoch)
        model = valid_epoch(model, criterion, valid_loader, current_epoch, device_ids)
        lr_sch.step()


if __name__ == "__main__":
    num_classes = 20
    class_names = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
    lamda_size = 0.1  # weight of size loss
    lamda_off = 1  # weight of offset loss
    alpha = 2  # focal loss parameter
    beta = 4  # focal loss parameter
    init_lr = 0.001
    final_lr = 0.00001
    epoch = 500
    batch_size = 8
    data_root_dir = r"F:\data\VOCdevkit\VOC2012\voc"
    img_size = (512, 512)
    sigma_iou_thresh = 0.7  # for getting gauss standard dev
    num_workers = 1
    R = 4
    log_step = 10
    CUDA_VISIBLE_DEVICES = "0"
    main(num_classes, class_names, lamda_size, lamda_off, alpha, beta, init_lr, final_lr, epoch, batch_size, data_root_dir, img_size, sigma_iou_thresh, num_workers, R, log_step, CUDA_VISIBLE_DEVICES)