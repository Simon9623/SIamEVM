# train_LaSOT.py

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

# <<< MODIFIED: This function is still needed for the LaSOT pseudo-masks
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

# <<< MODIFIED: These mask-related losses are no longer used but kept for code integrity
def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    target = target.contiguous().view(target.size(0), pred.size(1), -1)
    intersection = (pred * target).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth).clamp(min=1e-6)
    return 1 - dice.mean()

def focal_loss(pred, target, alpha=0.75, gamma=2.0):
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
    return F.l1_loss(pred_grad, target_grad)

def giou_loss(box1, box2):
    x1, y1, x2, y2 = box1.unbind(-1)
    x1_, y1_, x2_, y2_ = box2.unbind(-1)
    inter_x1 = torch.max(x1, x1_)
    inter_y1 = torch.max(y1, y1_)
    inter_x2 = torch.min(x2, x2_)
    inter_y2 = torch.min(y2, y2_)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2 - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    c_x1 = torch.min(x1, x1_)
    c_y1 = torch.min(y1, y1_)
    c_x2 = torch.max(x2, x2_)
    c_y2 = torch.max(y2, y2_)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    giou = iou - (c_area - union_area) / c_area.clamp(min=1e-6)
    return (1 - giou).mean()

def visualize_predict(template_img, search_img, pred_bbox, pred_mask, gt_bbox, gt_mask, epoch, batch_idx, save_dir='visualizations_LaSOT'):
    # Visualization remains largely the same, but gt_mask will be a rectangle
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
    axes[3].set_title('Ground Truth (Bbox and Pseudo Mask)')
    axes[3].legend()
    axes[3].axis('off')
    save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved visualization to {save_path}')

def custom_collate_fn(batch):
    # This function can remain unchanged
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

# <<< NEW: Helper function to create a pseudo-mask from a bounding box
def bbox_to_mask(bbox, width, height):
    """Generates a rectangular mask from a bounding box [x, y, w, h]"""
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    x, y, w, h = map(int, bbox)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 > x1 and y2 > y1:
        mask[:, y1:y2, x1:x2] = 1.0
    return mask

# <<< NEW: Dataset class for LaSOT
class LaSOTDataset(Dataset):
    def __init__(self, base_path, split='train', template_transform=None, search_transform=None, cache_images=False):
        self.base_path = base_path
        self.split = split
        self.template_transform = template_transform
        self.search_transform = search_transform
        self.samples = []
        self.cache_images = cache_images
        self.image_cache = {}
        self.load_data()

    def load_data(self):
        # The official LaSOT split file is 'testing_set.txt' for the extension
        # Here we assume a file named 'lasot_ext_train.txt' exists, listing video categories
        split_file = 'lasot_ext_train.txt' # You may need to create this file
        list_path = os.path.join(self.base_path, split_file)
        with open(list_path, 'r') as f:
            video_categories = f.read().strip().split('\n')

        skipped_samples = 0
        for category in video_categories:
            category_path = os.path.join(self.base_path, category)
            for video_name in sorted(os.listdir(category_path)):
                video_path = os.path.join(category_path, video_name)
                if not os.path.isdir(video_path):
                    continue

                gt_path = os.path.join(video_path, 'groundtruth.txt')
                img_path = os.path.join(video_path, 'img')

                try:
                    ground_truth = np.loadtxt(gt_path, delimiter=',')
                    frames = sorted(os.listdir(img_path))
                    
                    if len(frames) <= 1:
                        skipped_samples += len(frames)
                        continue

                    # Use first frame as template, subsequent frames as search
                    for i in range(1, len(frames)):
                        # Check if bbox has valid size
                        if ground_truth[0][2] <= 0 or ground_truth[0][3] <= 0 or \
                           ground_truth[i][2] <= 0 or ground_truth[i][3] <= 0:
                            skipped_samples += 1
                            continue

                        self.samples.append({
                            'video_name': video_name,
                            'template_frame_path': os.path.join(img_path, frames[0]),
                            'search_frame_path': os.path.join(img_path, frames[i]),
                            'template_bbox': ground_truth[0], # [x, y, w, h]
                            'search_bbox': ground_truth[i],   # [x, y, w, h]
                        })
                except Exception as e:
                    logging.warning(f"Could not process video {video_name}: {e}")
                    skipped_samples += 1

        logging.info(f"Loaded {len(self.samples)} valid samples from LaSOT, skipped {skipped_samples} samples")

        if self.cache_images:
            # Caching logic remains the same
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        template_img_path = sample['template_frame_path']
        search_img_path = sample['search_frame_path']
        
        try:
            template_img = Image.open(template_img_path).convert('RGB')
            search_img = Image.open(search_img_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Error loading images at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        w_t, h_t = template_img.size
        w_s, h_s = search_img.size
        
        template_bbox_xywh = sample['template_bbox']
        search_bbox_xywh = sample['search_bbox']
        
        # Generate pseudo-masks from bounding boxes
        template_mask = bbox_to_mask(template_bbox_xywh, w_t, h_t)
        search_mask = bbox_to_mask(search_bbox_xywh, w_s, h_s)
        
        if self.template_transform:
            template_img = self.template_transform(template_img)
        if self.search_transform:
            search_img = self.search_transform(search_img)

        # The mask tensors here are pseudo-masks, resized to match image input sizes
        template_mask_resized = F.interpolate(template_mask.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)
        search_mask_resized = F.interpolate(search_mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        
        # Convert bbox from [x,y,w,h] to [x1,y1,x2,y2] for loss calculation
        x, y, w, h = search_bbox_xywh
        search_bbox_xyxy = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        
        # We also need the template bbox for the model's template method
        x_t, y_t, w_t, h_t = template_bbox_xywh
        template_bbox_xyxy = torch.tensor([x_t, y_t, x_t + w_t, y_t + h_t], dtype=torch.float32)

        return {
            'template_img': template_img.contiguous(),
            'search_img': search_img.contiguous(),
            'template_mask': template_mask_resized.contiguous(),
            'search_mask': search_mask_resized.contiguous(),
            'template_bbox': template_bbox_xyxy.contiguous(), # Not used in loss but kept for consistency
            'search_bbox': search_bbox_xyxy.contiguous(),
            'video_id': sample['video_name'],
            'search_frame': os.path.basename(search_img_path)
        }

# <<< NEW: Data loader builder for LaSOT
def build_LaSOT_data_loader(args):
    template_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    search_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LaSOTDataset(
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

    # LaSOT does not have a standard validation set in the same format,
    # you might need to create one or skip validation during this phase.
    # For simplicity, we return None for val_loader.
    val_loader = None 

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
        # <<< MODIFIED: Save path for LaSOT fine-tuning
        best_filename = os.path.join(os.path.dirname(args.resume), f'best_lasot_{epoch}.pth')
        torch.save(state, best_filename)
        logging.info(f'Saved best model to {best_filename}')
    if epoch % args.checkpoint_interval == 0:
        periodic_filename = os.path.join(os.path.dirname(args.resume), f'checkpoint_lasot_epoch_{epoch}.pth')
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
    # This function needs significant changes to the loss calculation
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    bbox_losses = AverageMeter()
    # mask_losses = AverageMeter() # <<< MODIFIED: No longer tracking mask loss
    iou_losses = AverageMeter()
    iou_score_losses = AverageMeter()

    model.train()
    if not fine_tune_backbone:
        model.template_features.eval()
        model.search_features.eval()
    end = time.time()

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        template_img = data['template_img'].cuda()
        search_img = data['search_img'].cuda()
        # search_mask is a pseudo-mask, not used in loss
        search_bbox = data['search_bbox'].cuda()

        model.template(template_img)
        iou_scores, obj_scores, pred_bbox, pred_mask = model.track_mask(search_img)

        # The predicted mask is still generated for the memory bank mechanism
        pred_mask = pred_mask.squeeze(1)

        best_indices = torch.argmax(iou_scores, dim=0)
        iou_pred = iou_scores[best_indices, torch.arange(iou_scores.size(1))]

        # <<< MODIFIED: Loss Calculation for LaSOT
        bbox_loss = nn.SmoothL1Loss()(pred_bbox, search_bbox)
        
        # Mask loss is now zero as we don't have ground truth masks
        mask_loss = torch.tensor(0.0, device=pred_bbox.device)
        
        iou_true = compute_iou_torch(pred_bbox, search_bbox)
        iou_score_loss = nn.MSELoss()(iou_pred, iou_true)
        iou_loss = giou_loss(pred_bbox, search_bbox)

        small_obj_mask = ((search_bbox[..., 2] - search_bbox[..., 0]) * (search_bbox[..., 3] - search_bbox[..., 1])) < (0.1 * 256 * 256)
        small_obj_weight = torch.where(small_obj_mask, 2.0, 1.0)
        bbox_loss = (bbox_loss * small_obj_weight).mean()

        # <<< MODIFIED: Adaptive weights no longer include mask_loss
        init_weights = [args.bbox_weight, args.iou_weight, args.iou_score_weight]
        losses_values = [bbox_loss.item(), iou_loss.item(), iou_score_loss.item()]
        
        # Using a simpler weighting scheme for fine-tuning
        # Here we just use the fixed weights from args
        new_weights = [args.bbox_weight, 0.0, args.iou_weight, args.iou_score_weight]

        # Total loss calculation without mask_loss
        loss = (new_weights[0] * bbox_loss +
                new_weights[2] * iou_loss +
                new_weights[3] * iou_score_loss) / args.accumulate_steps

        if not is_valid_number(loss):
            logging.warning(f"Batch {i} loss contains NaN or Inf, skipping")
            continue

        losses.update(loss.item() * args.accumulate_steps, search_img.size(0))
        bbox_losses.update(bbox_loss.item(), search_img.size(0))
        # mask_losses.update(mask_loss.item(), search_img.size(0)) # No mask loss
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
        
        if i % args.vis_freq == 0:
            visualize_predict(
                template_img=template_img[0],
                search_img=search_img[0],
                pred_bbox=pred_bbox[0],
                pred_mask=pred_mask[0],
                gt_bbox=search_bbox[0],
                gt_mask=data['search_mask'][0], # Visualize the pseudo-mask
                epoch=epoch,
                batch_idx=i,
                save_dir='visualizations_LaSOT'
            )

        if i % args.print_freq == 0:
            global_step = epoch * len(train_loader) + i
            tb_writer.add_scalar('Loss/train', losses.avg, global_step)
            tb_writer.add_scalar('Bbox_Loss/train', bbox_losses.avg, global_step)
            tb_writer.add_scalar('IoU_Loss/train', iou_losses.avg, global_step)
            tb_writer.add_scalar('IoU_Score_Loss/train', iou_score_losses.avg, global_step)
            tb_writer.add_scalar('Weight/bbox', new_weights[0], global_step)
            tb_writer.add_scalar('Weight/iou', new_weights[2], global_step)
            tb_writer.add_scalar('Weight/iou_score', new_weights[3], global_step)

            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.avg:.3f}\t'
                f'Loss {losses.avg:.4f}\t'
                f'Bbox Loss {bbox_losses.avg:.4f}\t'
                f'IoU Loss {iou_losses.avg:.4f}\t'
                f'IoU Score Loss {iou_score_losses.avg:.4f}\t'
                f'Weights [bbox: {new_weights[0]:.3f}, iou: {new_weights[2]:.3f}, iou_score: {new_weights[3]:.3f}]'
            )

    torch.cuda.empty_cache()
    return losses.avg

# <<< MODIFIED: Validation is simplified as we don't have a standard val set for LaSOT
def validate(model, val_loader, epoch, tb_writer, args):
    if val_loader is None:
        logging.info("No validation loader provided, skipping validation.")
        return 0.0 # Return a default value for loss
    # If you create a validation set for LaSOT, the logic would be similar to train()
    # but with torch.no_grad() and without optimizer steps.
    # For now, we keep it simple.
    return 0.0

def main():
    # <<< MODIFIED: Argument defaults for LaSOT
    parser = argparse.ArgumentParser(description='Finetune SiamEVM model on LaSOT-ext')
    parser.add_argument('--arch', default='Custom', type=str, help='Model architecture')
    parser.add_argument('--batch', default=8, type=int, help='Batch size')
    parser.add_argument('--accumulate_steps', default=2, type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', default=50, type=int, help='Total training epochs for finetuning')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate for finetuning')
    parser.add_argument('--lr_step', default=15, type=int, help='Learning rate step size')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--clip', default=5.0, type=float, help='Gradient clipping value')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--print_freq', default=100, type=int, help='Log frequency')
    parser.add_argument('--vis_freq', default=500, type=int, help='Visualization frequency')
    parser.add_argument('--checkpoint_interval', default=1, type=int, help='Checkpoint saving interval')
    parser.add_argument('--cache_images', action='store_true', help='Cache images to memory')
    parser.add_argument('--base_path', default='/path/to/your/LaSOT_dataset/', type=str, help='LaSOT dataset root directory')
    # Pretrained path is not used here as we resume from a fully trained model
    parser.add_argument('--pretrained', default='', type=str, help='Pretrained backbone path (not used when resuming)')
    parser.add_argument('--resume', default='/home/phd-level/Desktop/SiamEVM_REV/best_19.pth', type=str, help='Resume from DAVIS-trained checkpoint')
    parser.add_argument('--bbox_weight', default=3.5, type=float, help='Bbox loss weight')
    parser.add_argument('--mask_weight', default=0.0, type=float, help='Mask loss weight (MUST be 0 for LaSOT)')
    parser.add_argument('--iou_weight', default=3.0, type=float, help='IoU loss weight')
    parser.add_argument('--iou_score_weight', default=2.0, type=float, help='IoU score loss weight')
    parser.add_argument('--backbone_finetune_epoch', default=0, type=int, help='Finetune backbone from the start')
    args = parser.parse_args()

    # <<< MODIFIED: Log and runs directory for LaSOT
    log_dir = '/home/phd-level/Desktop/SiamEVM_REV/train/logs_lasot'
    runs_dir = '/home/phd-level/Desktop/SiamEVM_REV/train/runs_lasot'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'train_lasot_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logger = init_log(log_file)
    logger.info(f'Arguments: {args}')

    tb_writer = SummaryWriter(log_dir=runs_dir)

    # <<< MODIFIED: Use the new LaSOT data loader
    train_loader, val_loader = build_LaSOT_data_loader(args)

    model = Custom(pretrain=False).cuda() # Pretraining is handled by resume
    
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        logger.info(f'Loading model from {args.resume} to finetune on LaSOT...')
        checkpoint_data = torch.load(args.resume)
        model.load_state_dict(checkpoint_data['state_dict'])
        # start_epoch = checkpoint['epoch'] # Better to start from epoch 0 for finetuning
        # best_loss = checkpoint['best_loss'] # Reset best_loss for the new task
        logger.info(f'Successfully loaded model from {args.resume}.')
    else:
        logger.error("A checkpoint from DAVIS training is required to finetune on LaSOT. Please provide a --resume path.")
        return

    for epoch in range(start_epoch, args.epochs):
        fine_tune_backbone = epoch >= args.backbone_finetune_epoch
        
        # <<< MODIFIED: Parameter freezing logic
        params_to_train = []
        for name, param in model.named_parameters():
            # Freeze the entire mask decoder
            if 'mask_decoder' in name:
                param.requires_grad = False
            # Train the correlation module and projections
            elif 'template_proj' in name or 'search_proj' in name or 'corr_module' in name:
                param.requires_grad = True
                params_to_train.append(param)
            # Optionally finetune parts of the backbone
            elif fine_tune_backbone and ('features.layers.2' in name or 'features.layers.3' in name):
                param.requires_grad = True
                # Use a smaller learning rate for backbone
                # The optimizer will handle this if we group params
            else:
                param.requires_grad = False
        
        # Group parameters for different learning rates
        optimizer_params = [
            {
                "params": [p for n, p in model.named_parameters() if ("features.layers" in n) and p.requires_grad],
                "lr": args.lr * 0.1
            },
            {
                "params": [p for n, p in model.named_parameters() if not ("features.layers" in n) and p.requires_grad],
                "lr": args.lr
            },
        ]

        optimizer = optim.AdamW(optimizer_params, lr=args.lr, weight_decay=1e-4)
        lr_scheduler = build_lr_scheduler(optimizer, args)

        train_loss = train(model, train_loader, optimizer, epoch, tb_writer, args, fine_tune_backbone)
        
        # Validation is optional for this phase
        val_loss = validate(model, val_loader, epoch, tb_writer, args)
        
        lr_scheduler.step()

        # Checkpointing based on training loss if validation is skipped
        current_loss = train_loss
        is_best = current_loss < best_loss
        best_loss = min(current_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_loss': best_loss
        }, args, is_best, epoch, filename=f'/home/phd-level/Desktop/SiamEVM_REV/train/checkpoints_LaSOT/checkpoint_LaSOT_{epoch}.pth')

        logger.info(f'Epoch {epoch} ended, train_loss: {train_loss:.4f}, best_loss_so_far: {best_loss:.4f}')

    tb_writer.close()

if __name__ == '__main__':
    main()