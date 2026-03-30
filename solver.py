import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from evaluation import get_DSC, get_IoU, get_HD95


class DiceLoss(nn.Module):
    """Soft Dice loss operating on logits (applies sigmoid internally)."""

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        smooth = 1.0
        pflat = probs.view(probs.size(0), -1)
        tflat = targets.view(targets.size(0), -1)
        inter = (pflat * tflat).sum(dim=1)
        dice = (2.0 * inter + smooth) / (pflat.sum(dim=1) + tflat.sum(dim=1) + smooth)
        return (1.0 - dice).mean()


class Solver:
    def __init__(self, config, train_loader, val_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model_type = config.model_type
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.image_size = config.image_size
        self.t = config.t

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs

        self.model_path = config.model_path
        self.result_path = config.result_path

        self.device = torch.device(
            f'cuda:{config.cuda_idx}' if torch.cuda.is_available() else 'cpu'
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.build_model()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def build_model(self):
        builders = {
            'U_Net':      lambda: U_Net(img_ch=self.img_ch, output_ch=self.output_ch),
            'R2U_Net':    lambda: R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t),
            'AttU_Net':   lambda: AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch),
            'R2AttU_Net': lambda: R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t),
        }
        self.model = builders[self.model_type]()
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr, (self.beta1, self.beta2)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        best_path = os.path.join(self.model_path, 'best_model.pth')
        best_dsc = 0.0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            # --- train one epoch ---
            self.model.train()
            epoch_loss = 0.0
            train_dsc = 0.0
            train_iou = 0.0
            n_samples = 0

            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.bce_loss(logits, masks) + self.dice_loss(logits, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    bs = images.size(0)
                    epoch_loss += loss.item() * bs
                    for i in range(bs):
                        train_dsc += get_DSC(probs[i], masks[i]).item()
                        train_iou += get_IoU(probs[i], masks[i]).item()
                    n_samples += bs

            self.scheduler.step()
            epoch_loss /= n_samples
            train_dsc /= n_samples
            train_iou /= n_samples

            # --- validate ---
            val_dsc, val_iou = self._validate()

            lr_now = self.optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1:3d}/{self.num_epochs}] '
                f'lr={lr_now:.2e}  Loss={epoch_loss:.4f}  |  '
                f'Train DSC={train_dsc:.4f} IoU={train_iou:.4f}  |  '
                f'Val DSC={val_dsc:.4f} IoU={val_iou:.4f}'
            )

            if val_dsc > best_dsc:
                best_dsc = val_dsc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), best_path)
                print(f'  >> New best (Val DSC={best_dsc:.4f})')

        print(f'\nTraining complete. Best epoch={best_epoch}, '
              f'Best Val DSC={best_dsc:.4f}')
        print('Running test on best model ...\n')
        self.test()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_dsc, total_iou, n = 0.0, 0.0, 0
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            probs = torch.sigmoid(self.model(images))
            bs = images.size(0)
            for i in range(bs):
                total_dsc += get_DSC(probs[i], masks[i]).item()
                total_iou += get_IoU(probs[i], masks[i]).item()
            n += bs
        return total_dsc / n, total_iou / n

    # ------------------------------------------------------------------
    # Testing  (DSC, IoU, HD95, FPS, FLOPs, VRAM Peak)
    # ------------------------------------------------------------------
    def test(self):
        best_path = os.path.join(self.model_path, 'best_model.pth')
        if not os.path.isfile(best_path):
            print(f'ERROR: {best_path} not found. Train first.')
            return
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device)
        )
        self.model.eval()

        # ---- FLOPs (computed once) ----
        flops_g = self._compute_flops()

        # ---- GPU warm-up & VRAM baseline ----
        with torch.no_grad():
            dummy = torch.randn(
                1, self.img_ch, self.image_size, self.image_size,
                device=self.device,
            )
            for _ in range(20):
                self.model(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

        # ---- Inference loop ----
        results = []
        total_time = 0.0

        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = self.model(images)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                total_time += t1 - t0

                probs = torch.sigmoid(logits)
                for i in range(images.size(0)):
                    dsc_val = get_DSC(probs[i], masks[i]).item()
                    iou_val = get_IoU(probs[i], masks[i]).item()
                    hd95_val = get_HD95(probs[i], masks[i])
                    results.append({
                        'DSC': dsc_val, 'IoU': iou_val, 'HD95': hd95_val,
                    })

        # ---- Aggregate ----
        n = len(results)
        fps = n / total_time if total_time > 0 else 0.0
        vram_peak_mb = 0.0
        if self.device.type == 'cuda':
            vram_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        avg_dsc = np.mean([r['DSC'] for r in results])
        avg_iou = np.mean([r['IoU'] for r in results])
        hd95_finite = [r['HD95'] for r in results if np.isfinite(r['HD95'])]
        avg_hd95 = np.mean(hd95_finite) if hd95_finite else float('nan')

        # ---- Print ----
        sep = '=' * 55
        print(sep)
        print(f'  Test Results: {self.model_type}  on  {self.config.dataset}')
        print(sep)
        print(f'  DSC        : {avg_dsc:.4f}')
        print(f'  IoU        : {avg_iou:.4f}')
        print(f'  HD95       : {avg_hd95:.4f}')
        print(f'  FPS        : {fps:.2f}')
        print(f'  FLOPs      : {flops_g:.4f} G')
        print(f'  VRAM Peak  : {vram_peak_mb:.2f} MB')
        print(sep)

        # ---- Save per-sample CSV ----
        per_csv = os.path.join(self.result_path, 'per_sample.csv')
        with open(per_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['DSC', 'IoU', 'HD95'])
            w.writeheader()
            w.writerows(results)

        # ---- Save summary CSV ----
        summary_csv = os.path.join(self.result_path, 'summary.csv')
        with open(summary_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'Model', 'Dataset', 'DSC', 'IoU', 'HD95',
                'FPS', 'FLOPs_G', 'VRAM_Peak_MB',
            ])
            w.writerow([
                self.model_type, self.config.dataset,
                f'{avg_dsc:.4f}', f'{avg_iou:.4f}', f'{avg_hd95:.4f}',
                f'{fps:.2f}', f'{flops_g:.4f}', f'{vram_peak_mb:.2f}',
            ])

        print(f'  Results saved to {self.result_path}/\n')

    # ------------------------------------------------------------------
    # FLOPs (via thop)
    # ------------------------------------------------------------------
    def _compute_flops(self):
        try:
            from thop import profile
            dummy = torch.randn(
                1, self.img_ch, self.image_size, self.image_size,
                device=self.device,
            )
            flops, _ = profile(self.model, inputs=(dummy,), verbose=False)
            return flops / 1e9
        except ImportError:
            print('WARNING: thop not installed – FLOPs skipped.  '
                  'Install with:  pip install thop')
            return 0.0
