
import os
import time
import argparse
import logging
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.backends.cudnn as cudnn
import gc
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append('/home/phd-level/Desktop/SiamEVM_REV/models/')
sys.path.append('/home/phd-level/Desktop/SiamEVM_REV/models/utils')
from custom import Custom
from iou import compute_iou_torch

cudnn.benchmark = True

# Set PyTorch CUDA memory allocator configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_random_seed(42)

def init_log(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger('global')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def mask_to_bbox(mask):
    mask = mask.squeeze().cpu().numpy()
    if mask.max() == 0:
        return [0, 0, 0, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]

def compute_iou_torch(box1, box2):
    x1, y1, x2, y2 = box1.unbind(-1)
    x1_, y1_, x2_, y2_ = box2.unbind(-1)
    inter_x1 = torch.max(x1, x1_)
    inter_y1 = torch.max(y1, y1_)
    inter_x2 = torch.min(x2, x2_)
    inter_y2 = torch.min(y2, y2_)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou.clamp(max=1.0)

def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    target = target.contiguous().view(target.size(0), pred.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth).clamp(min=1e-6)
    return 1 - dice.mean()

def focal_loss(pred, target, alpha=0.75, gamma=2.0):  # Increased alpha for small objects
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    target = target.contiguous().view(target.size(0), pred.size(1), -1)
    bce = nn.BCELoss(reduction='none')(pred, target)
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()

def boundary_loss(pred, target, smooth=1):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + smooth)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + smooth)
    return F.l1_loss(pred_grad, target_grad)  # 修正：移除 .cpu()

def giou_loss(box1, box2):
    x1, y1, x2, y2 = box1.unbind(-1)
    x1_, y1_, x2_, y2_ = box2.unbind(-1)
    inter_x1 = torch.max(x1, x1_)
    inter_y1 = torch.max(y1, y1_)
    inter_x2 = torch.min(x2, x2_)
    inter_y2 = torch.min(y2, y2_)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    c_x1 = torch.min(x1, x1_)
    c_y1 = torch.min(y1, y1_)
    c_x2 = torch.max(x2, x2_)
    c_y2 = torch.max(y2, y2_)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    giou = iou - (c_area - union_area) / c_area.clamp(min=1e-6)
    return (1 - giou).mean()

def visualize_predict(template_img, search_img, pred_bbox, pred_mask, gt_bbox, gt_mask, epoch, batch_idx, save_dir='visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=template_img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=template_img.device).view(3, 1, 1)
    template_img = template_img.mul(std).add(mean).permute(1, 2, 0).cpu().numpy()
    search_img = search_img.mul(std).add(mean).permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.detach().cpu().squeeze().numpy()
    gt_mask = gt_mask.cpu().squeeze().numpy()
    pred_bbox = pred_bbox.detach().cpu().numpy()
    gt_bbox = gt_bbox.cpu().numpy()

    pred_mask = (pred_mask * 255).astype(np.uint8) / 255.0
    gt_mask = (gt_mask * 255).astype(np.uint8) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(template_img)
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(search_img)
    axes[1].set_title('Search Image')
    axes[1].axis('off')
    axes[2].imshow(search_img)
    x1, y1, x2, y2 = pred_bbox
    axes[2].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', label='Predict Bbox')
    pred_mask_vis = np.zeros_like(search_img)
    pred_mask_vis[..., 0] = pred_mask
    axes[2].imshow(pred_mask_vis, alpha=0.5)
    axes[2].set_title('Prediction (Bbox and Mask)')
    axes[2].legend()
    axes[2].axis('off')
    axes[3].imshow(search_img)
    x1, y1, x2, y2 = gt_bbox
    axes[3].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', label='Ground Truth Bbox')
    gt_mask_vis = np.zeros_like(search_img)
    gt_mask_vis[..., 1] = gt_mask
    axes[3].imshow(gt_mask_vis, alpha=0.4)
    axes[3].set_title('Ground Truth (Bbox and Mask)')
    axes[3].legend()
    axes[3].axis('off')
    save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved visualization to {save_path}')

def custom_collate_fn(batch):
    template_imgs = torch.stack([item['template_img'].contiguous() for item in batch], dim=0)
    search_imgs = torch.stack([item['search_img'].contiguous() for item in batch], dim=0)
    template_masks = torch.stack([item['template_mask'].contiguous() for item in batch], dim=0)
    search_masks = torch.stack([item['search_mask'].contiguous() for item in batch], dim=0)
    template_bboxes = torch.stack([item['template_bbox'].contiguous() for item in batch], dim=0)
    search_bboxes = torch.stack([item['search_bbox'].contiguous() for item in batch], dim=0)
    video_ids = [item['video_id'] for item in batch]
    search_frames = [item['search_frame'] for item in batch]
    return {
        'template_img': template_imgs,
        'search_img': search_imgs,
        'template_mask': template_masks,
        'search_mask': search_masks,
        'template_bbox': template_bboxes,
        'search_bbox': search_bboxes,
        'video_id': video_ids,
        'search_frame': search_frames
    }
class CustomDataset(Dataset):
    def __init__(self, base_path, split='train', template_transform=None, search_transform=None, cache_images=False):
        self.base_path = base_path
        self.split = split
        self.template_transform = template_transform
        self.search_transform = search_transform
        self.videos = []
        self.cache_images = cache_images
        self.image_cache = {}
        self.load_data()

    def load_data(self):
        split_file = 'train.txt' if self.split == 'trainval' else 'val.txt'
        annotation_path = os.path.join(self.base_path, 'ImageSets/2017', split_file)
        with open(annotation_path, 'r') as f:
            video_names = f.read().strip().split('\n')
        
        skipped_samples = 0
        for video_id in video_names:
            video_path = os.path.join(self.base_path, 'JPEGImages/480p', video_id)
            frames = sorted(os.listdir(video_path))
            for i in range(1, len(frames)):
                template_mask_path = os.path.join(
                    self.base_path, 'Annotations/480p', video_id, frames[0].replace('.jpg', '.png')
                )
                search_mask_path = os.path.join(
                    self.base_path, 'Annotations/480p', video_id, frames[i].replace('.jpg', '.png')
                )
                
                try:
                    template_mask = Image.open(template_mask_path).convert('L')
                    search_mask = Image.open(search_mask_path).convert('L')
                    template_mask_np = np.array(template_mask)
                    search_mask_np = np.array(search_mask)
                    
                    if template_mask_np.max() == 0 or search_mask_np.max() == 0:
                        skipped_samples += 1
                        continue
                    
                    self.videos.append({
                        'video_id': video_id,
                        'template_frame': frames[0],
                        'search_frame': frames[i]
                    })
                except Exception as e:
                    logging.warning(f"Failed to load mask for video {video_id}, frame {frames[i]}: {e}")
                    skipped_samples += 1
                    continue
        
        logging.info(f"Loaded {len(self.videos)} valid samples, skipped {skipped_samples} samples with empty masks")
        
        if self.cache_images:
            logging.info("Caching images to memory...")
            for video in self.videos:
                template_img_path = os.path.join(self.base_path, 'JPEGImages/480p', video['video_id'], video['template_frame'])
                search_img_path = os.path.join(self.base_path, 'JPEGImages/480p', video['video_id'], video['search_frame'])
                try:
                    self.image_cache[template_img_path] = Image.open(template_img_path).convert('RGB')
                    self.image_cache[search_img_path] = Image.open(search_img_path).convert('RGB')
                except Exception as e:
                    logging.warning(f"Failed to cache image {template_img_path} or {search_img_path}: {e}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        template_img_path = os.path.join(self.base_path, 'JPEGImages/480p', video['video_id'], video['template_frame'])
        search_img_path = os.path.join(self.base_path, 'JPEGImages/480p', video['video_id'], video['search_frame'])
        template_mask_path = os.path.join(self.base_path, 'Annotations/480p', video['video_id'], video['template_frame'].replace('.jpg', '.png'))
        search_mask_path = os.path.join(self.base_path, 'Annotations/480p', video['video_id'], video['search_frame'].replace('.jpg', '.png'))
        
        try:
            if self.cache_images and template_img_path in self.image_cache and search_img_path in self.image_cache:
                template_img = self.image_cache[template_img_path]
                search_img = self.image_cache[search_img_path]
            else:
                template_img = Image.open(template_img_path).convert('RGB')
                search_img = Image.open(search_img_path).convert('RGB')
            template_mask = Image.open(template_mask_path).convert('L')
            search_mask = Image.open(search_mask_path).convert('L')
        except Exception as e:
            logging.warning(f"Error loading data at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.videos) - 1))

        if self.template_transform:
            template_img = self.template_transform(template_img)
        if self.search_transform:
            search_img = self.search_transform(search_img)

        template_mask = transforms.ToTensor()(template_mask)
        search_mask = transforms.ToTensor()(search_mask)
        template_mask = F.interpolate(template_mask.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)
        search_mask = F.interpolate(search_mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        if template_mask.max() == 0 or search_mask.max() == 0:
            logging.warning(f"Empty mask detected at index {idx}, returning random valid sample")
            return self.__getitem__(random.randint(0, len(self.videos) - 1))

        template_bbox = mask_to_bbox(template_mask)
        search_bbox = mask_to_bbox(search_mask)

        template_bbox = torch.tensor(template_bbox, dtype=torch.float32)
        search_bbox = torch.tensor(search_bbox, dtype=torch.float32)

        return {
            'template_img': template_img.contiguous(),
            'search_img': search_img.contiguous(),
            'template_mask': template_mask.contiguous(),
            'search_mask': search_mask.contiguous(),
            'template_bbox': template_bbox.contiguous(),
            'search_bbox': search_bbox.contiguous(),
            'video_id': video['video_id'],
            'search_frame': video['search_frame']
        }

def build_data_loader(args):
    template_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Changed to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    search_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Changed to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(
        base_path=args.base_path,
        split='train',
        template_transform=template_transform,
        search_transform=search_transform,
        cache_images=args.cache_images
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_dataset = CustomDataset(
        base_path=args.base_path,
        split='val',
        template_transform=template_transform,
        search_transform=search_transform,
        cache_images=args.cache_images
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader

def build_lr_scheduler(optimizer, args):
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma
    )
    return scheduler

def save_checkpoint(state, args, is_best, epoch, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(args.resume), f'best_{epoch}.pth')
        torch.save(state, best_filename)
        logging.info(f'Saved best model to {best_filename}')
    if epoch % args.checkpoint_interval == 0:
        periodic_filename = os.path.join(os.path.dirname(args.resume), f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, periodic_filename)
        logging.info(f'Saved periodic checkpoint to {periodic_filename}')

def is_valid_number(x):
    return not (torch.isnan(x) or torch.isinf(x))

def get_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6  # MB
        reserved = torch.cuda.memory_reserved() / 1e6  # MB
        return allocated, reserved
    return None, None

def train(model, train_loader, optimizer, epoch, tb_writer, args, fine_tune_backbone=False):
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    bbox_losses = AverageMeter()
    mask_losses = AverageMeter()
    iou_losses = AverageMeter()
    iou_score_losses = AverageMeter()

    # 新增：用於存儲歷史權重（EMA）
    weight_history = {
        'bbox': args.bbox_weight,
        'mask': args.mask_weight,
        'iou': args.iou_weight,
        'iou_score': args.iou_score_weight
    }
    ema_alpha = 0.9  # EMA 平滑係數
    min_weight = 0.5  # 最小權重
    max_weight = 10.0  # 最大權重

    model.train()
    if not fine_tune_backbone:
        model.template_features.eval()
        model.search_features.eval()
    end = time.time()

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        template_img = data['template_img'].cuda()
        search_img = data['search_img'].cuda()
        search_mask = data['search_mask'].cuda()
        search_bbox = data['search_bbox'].cuda()

        if search_mask.max() == 0:
            logging.warning(f"Batch {i} contains empty search mask, skipping")
            continue

        model.template(template_img)
        iou_scores, obj_scores, pred_bbox, pred_mask = model.track_mask(search_img)

        pred_mask = pred_mask.squeeze(1)
        search_mask = search_mask.squeeze(1)

        best_indices = torch.argmax(iou_scores, dim=0)
        iou_pred = iou_scores[best_indices, torch.arange(iou_scores.size(1))]

        # 計算損失
        bbox_loss = nn.SmoothL1Loss()(pred_bbox, search_bbox)
        mask_loss = (0.3 * focal_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1), alpha=0.75, gamma=2.0) +
                     0.4 * dice_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1)) +
                     0.3 * boundary_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1)))
        iou_true = compute_iou_torch(pred_bbox, search_bbox)
        iou_score_loss = nn.MSELoss()(iou_pred, iou_true)
        iou_loss = giou_loss(pred_bbox, search_bbox)

        # 小目標加權
        small_obj_mask = ((search_bbox[..., 2] - search_bbox[..., 0]) * (search_bbox[..., 3] - search_bbox[..., 1])) < (0.1 * 256 * 256)
        small_obj_weight = torch.where(small_obj_mask, 2.0, 1.0)
        bbox_loss = (bbox_loss * small_obj_weight).mean()
        mask_loss = (mask_loss * small_obj_weight).mean()

        # 自適應權重調整
        losses_values = [bbox_loss.item(), mask_loss.item(), iou_loss.item(), iou_score_loss.item()]
        if all(v > 0 for v in losses_values):
            # 計算相對損失（正規化）
            mean_loss = sum(losses_values) / len(losses_values)
            relative_losses = [v / mean_loss for v in losses_values]
            
            # 根據訓練階段調整最大變動範圍
            epoch_progress = epoch / args.epochs
            max_change = 0.2 if epoch_progress < 0.2 else 0.5  # 早期限制調整幅度
            
            # 計算新權重（結合用戶指定權重和相對損失）
            init_weights = [args.bbox_weight, args.mask_weight, args.iou_weight, args.iou_score_weight]
            new_weights = []
            for init_w, rel_loss in zip(init_weights, relative_losses):
                adaptive_w = init_w / rel_loss  # 損失越大，權重越小
                # 限制變動範圍
                adaptive_w = min(max(adaptive_w, init_w * (1 - max_change)), init_w * (1 + max_change))
                # 應用 EMA 平滑
                key = ['bbox', 'mask', 'iou', 'iou_score'][len(new_weights)]
                weight_history[key] = ema_alpha * weight_history[key] + (1 - ema_alpha) * adaptive_w
                # 限制權重範圍
                smoothed_w = min(max(weight_history[key], min_weight), max_weight)
                new_weights.append(smoothed_w)
        else:
            # 如果損失值無效，使用初始權重
            new_weights = [args.bbox_weight, args.mask_weight, args.iou_weight, args.iou_score_weight]

        # 總損失
        loss = (new_weights[0] * bbox_loss + new_weights[1] * mask_loss +
                new_weights[2] * iou_loss + new_weights[3] * iou_score_loss) / args.accumulate_steps

        if not is_valid_number(loss):
            logging.warning(f"Batch {i} loss contains NaN or Inf, skipping")
            continue

        losses.update(loss.item() * args.accumulate_steps, search_img.size(0))
        bbox_losses.update(bbox_loss.item(), search_img.size(0))
        mask_losses.update(mask_loss.item(), search_img.size(0))
        iou_losses.update(iou_loss.item(), search_img.size(0))
        iou_score_losses.update(iou_score_loss.item(), search_img.size(0))

        loss.backward()

        if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(train_loader):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            allocated, reserved = get_memory_stats()
            logging.info(f'Batch {i}: GPU memory allocated={allocated:.2f}MB, reserved={reserved:.2f}MB')

        if i % args.vis_freq == 0:
            visualize_predict(
                template_img=template_img[0],
                search_img=search_img[0],
                pred_bbox=pred_bbox[0],
                pred_mask=pred_mask[0],
                gt_bbox=search_bbox[0],
                gt_mask=search_mask[0],
                epoch=epoch,
                batch_idx=i,
                save_dir='visualizations'
            )

        if i % args.print_freq == 0:
            global_step = epoch * len(train_loader) + i
            tb_writer.add_scalar('Loss/train', losses.avg, global_step)
            tb_writer.add_scalar('Bbox_Loss/train', bbox_losses.avg, global_step)
            tb_writer.add_scalar('Mask_Loss/train', mask_losses.avg, global_step)
            tb_writer.add_scalar('IoU_Loss/train', iou_losses.avg, global_step)
            tb_writer.add_scalar('IoU_Score_Loss/train', iou_score_losses.avg, global_step)
            # 新增：記錄權重
            tb_writer.add_scalar('Weight/bbox', new_weights[0], global_step)
            tb_writer.add_scalar('Weight/mask', new_weights[1], global_step)
            tb_writer.add_scalar('Weight/iou', new_weights[2], global_step)
            tb_writer.add_scalar('Weight/iou_score', new_weights[3], global_step)

            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Accumulation Steps: {args.accumulate_steps}\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Bbox Loss {bbox_losses.val:.4f} ({bbox_losses.avg:.4f})\t'
                f'Mask Loss {mask_losses.val:.4f} ({mask_losses.avg:.4f})\t'
                f'IoU Loss {iou_losses.val:.4f} ({iou_losses.avg:.4f})\t'
                f'IoU Score Loss {iou_score_losses.val:.4f} ({iou_score_losses.avg:.4f})\t'
                f'Weights [bbox: {new_weights[0]:.3f}, mask: {new_weights[1]:.3f}, iou: {new_weights[2]:.3f}, iou_score: {new_weights[3]:.3f}]'
            )

    torch.cuda.empty_cache()
    return losses.avg

def validate(model, val_loader, epoch, tb_writer, args):
    losses = AverageMeter()
    bbox_losses = AverageMeter()
    mask_losses = AverageMeter()
    iou_losses = AverageMeter()
    iou_score_losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            template_img = data['template_img'].cuda()
            search_img = data['search_img'].cuda()
            search_mask = data['search_mask'].cuda()
            search_bbox = data['search_bbox'].cuda()

            if search_mask.max() == 0:
                logging.warning(f"Validation batch {i} contains empty search mask, skipping")
                continue

            model.template(template_img)
            iou_scores, obj_scores, pred_bbox, pred_mask = model.track_mask(search_img)

            pred_mask = pred_mask.squeeze(1)
            search_mask = search_mask.squeeze(1)

            best_indices = torch.argmax(iou_scores, dim=0)
            iou_pred = iou_scores[best_indices, torch.arange(iou_scores.size(1))]

            bbox_loss = nn.SmoothL1Loss()(pred_bbox, search_bbox)
            mask_loss = (0.3 * focal_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1), alpha=0.75, gamma=2.0) +
                         0.4 * dice_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1)) +
                         0.3 * boundary_loss(pred_mask.unsqueeze(1), search_mask.unsqueeze(1)))
            iou_true = compute_iou_torch(pred_bbox, search_bbox)
            iou_score_loss = nn.MSELoss()(iou_pred, iou_true)
            iou_loss = giou_loss(pred_bbox, search_bbox)

            # 小目標加權
            small_obj_mask = ((search_bbox[..., 2] - search_bbox[..., 0]) * (search_bbox[..., 3] - search_bbox[..., 1])) < (0.1 * 256 * 256)
            small_obj_weight = torch.where(small_obj_mask, 2.0, 1.0)
            bbox_loss = (bbox_loss * small_obj_weight).mean()
            mask_loss = (mask_loss * small_obj_weight).mean()

            # 自適應權重（與訓練一致）
            losses_values = [bbox_loss.item(), mask_loss.item(), iou_loss.item(), iou_score_loss.item()]
            if all(v > 0 for v in losses_values):
                mean_loss = sum(losses_values) / len(losses_values)
                relative_losses = [v / mean_loss for v in losses_values]
                init_weights = [args.bbox_weight, args.mask_weight, args.iou_weight, args.iou_score_weight]
                new_weights = [init_w / rel_loss for init_w, rel_loss in zip(init_weights, relative_losses)]
                new_weights = [min(max(w, 0.5), 10.0) for w in new_weights]  # 限制範圍
            else:
                new_weights = [args.bbox_weight, args.mask_weight, args.iou_weight, args.iou_score_weight]

            loss = (new_weights[0] * bbox_loss + new_weights[1] * mask_loss +
                    new_weights[2] * iou_loss + new_weights[3] * iou_score_loss)

            losses.update(loss.item(), search_img.size(0))
            bbox_losses.update(bbox_loss.item(), search_img.size(0))
            mask_losses.update(mask_loss.item(), search_img.size(0))
            iou_losses.update(iou_loss.item(), search_img.size(0))
            iou_score_losses.update(iou_score_loss.item(), search_img.size(0))

            if i % 100 == 0:
                torch.cuda.empty_cache()

    tb_writer.add_scalar('Loss/val', losses.avg, epoch)
    tb_writer.add_scalar('Bbox_Loss/val', bbox_losses.avg, epoch)
    tb_writer.add_scalar('Mask_Loss/val', mask_losses.avg, epoch)
    tb_writer.add_scalar('IoU_Loss/val', iou_losses.avg, epoch)
    tb_writer.add_scalar('IoU_Score_Loss/val', iou_score_losses.avg, epoch)
    # 新增：記錄驗證權重
    tb_writer.add_scalar('Weight/bbox_val', new_weights[0], epoch)
    tb_writer.add_scalar('Weight/mask_val', new_weights[1], epoch)
    tb_writer.add_scalar('Weight/iou_val', new_weights[2], epoch)
    tb_writer.add_scalar('Weight/iou_score_val', new_weights[3], epoch)

    logging.info(
        f'Validation Epoch {epoch}: '
        f'Loss {losses.avg:.4f} '
        f'Bbox Loss {bbox_losses.avg:.4f} '
        f'Mask Loss {mask_losses.avg:.4f} '
        f'IoU Loss {iou_losses.avg:.4f} '
        f'IoU Score Loss {iou_score_losses.avg:.4f} '
        f'Weights [bbox: {new_weights[0]:.3f}, mask: {new_weights[1]:.3f}, iou: {new_weights[2]:.3f}, iou_score: {new_weights[3]:.3f}]'
    )

    return losses.avg

def main():
    parser = argparse.ArgumentParser(description='Train SiamEVM model on DAVIS 2017')
    parser.add_argument('--arch', default='Custom', type=str, help='Model architecture')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--accumulate_steps', default=4, type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', default=300, type=int, help='Total training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_step', default=10, type=int, help='Learning rate step size')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--clip', default=5.0, type=float, help='Gradient clipping value')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--print_freq', default=50, type=int, help='Log frequency')
    parser.add_argument('--vis_freq', default=40, type=int, help='Visualization frequency')
    parser.add_argument('--checkpoint_interval', default=5, type=int, help='Checkpoint saving interval')
    parser.add_argument('--cache_images', action='store_true', help='Cache images to memory')
    parser.add_argument('--base_path', default='/home/phd-level/Desktop/SiamEVM_REV/train/datasets/DAVIS', type=str, help='DAVIS 2017 dataset root directory')
    parser.add_argument('--pretrained', default='/home/phd-level/Desktop/SiamEVM_REV/models/VSSMBackbone/vssmsmall_dp03_ckpt_epoch_238.pth', type=str, help='Pretrained model path')
    parser.add_argument('--resume', default='/home/phd-level/Desktop/SiamEVM_REV/best_19.pth', type=str, help='Resume checkpoint path')
    parser.add_argument('--bbox_weight', default=3.5, type=float, help='Bbox loss weight')
    parser.add_argument('--mask_weight', default=5.0, type=float, help='Mask loss weight')
    parser.add_argument('--iou_weight', default=3.0, type=float, help='IoU loss weight')
    parser.add_argument('--iou_score_weight', default=2.0, type=float, help='IoU score loss weight')
    parser.add_argument('--backbone_finetune_epoch', default=20, type=int, help='Epoch to start finetuning backbone')
    args = parser.parse_args()

    log_dir = '/home/phd-level/Desktop/SiamEVM_REV/train/logs'
    runs_dir = '/home/phd-level/Desktop/SiamEVM_REV/train/runs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'train_davis_2017_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logger = init_log(log_file)
    logger.info(f'Arguments: {args}')

    tb_writer = SummaryWriter(log_dir=runs_dir)

    train_loader, val_loader = build_data_loader(args)

    model = Custom(pretrain=args.pretrained != '').cuda()
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f'Loaded pretrained model from {args.pretrained}')

    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        logger.info(f'Resumed training from {args.resume}, starting epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        fine_tune_backbone = epoch >= args.backbone_finetune_epoch
        params = []
        for name, param in model.named_parameters():
            if 'mask_decoder' in name or 'template_proj' in name or 'corr_module' in name:
                param.requires_grad = True
                params.append({'params': param, 'lr': args.lr})
            elif fine_tune_backbone and ('template_features.features.layers.2' in name or 'search_features.features.layers.2' in name):
                param.requires_grad = True
                params.append({'params': param, 'lr': args.lr * 0.1})
            else:
                param.requires_grad = False
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
        
        lr_scheduler = build_lr_scheduler(optimizer, args)

        train_loss = train(model, train_loader, optimizer, epoch, tb_writer, args, fine_tune_backbone)
        val_loss = validate(model, val_loader, epoch, tb_writer, args)
        lr_scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_loss': best_loss
        }, args, is_best, epoch, filename=f'/home/phd-level/Desktop/SiamEVM_REV/train/checkpoints/checkpoint_DAVIS_{epoch}.pth')

        logger.info(f'Epoch {epoch} ended, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, best_loss: {best_loss:.4f}')

    tb_writer.close()

if __name__ == '__main__':
    main()
