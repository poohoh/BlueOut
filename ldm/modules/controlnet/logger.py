import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        # Group per-epoch to align with EpochImageLogger
        # Allow an epoch offset (e.g., when resuming from a previous run)
        try:
            epoch_offset = int(os.environ.get("EPOCH_OFFSET", "0"))
        except Exception:
            epoch_offset = 0
        effective_epoch = current_epoch + epoch_offset
        epoch_dir = f"epoch_{effective_epoch + 1:04d}"
        root = os.path.join(save_dir, "image_log", split, epoch_dir)
        for key in sorted(images.keys()):
            grid = torchvision.utils.make_grid(images[key], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "b-{:06}_{}_gs-{:06}_e-{:06}.png".format(
                batch_idx, key, global_step, effective_epoch
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # choose index: batch_idx or global_step
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **(self.log_images_kwargs or {}))

            for key in sorted(images.keys()):
                N = min(images[key].shape[0], self.max_images)
                images[key] = images[key][:N]
                if isinstance(images[key], torch.Tensor):
                    images[key] = images[key].detach().cpu()
                    if self.clamp:
                        images[key] = torch.clamp(images[key], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        # respect log_first_step=False and avoid logging at step 0
        if not self.log_first_step and check_idx == 0:
            return False
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.disabled:
            return
        # Avoid any logging work on non-zero ranks
        try:
            if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
                return
        except Exception:
            pass
        self.log_img(pl_module, batch, batch_idx, split="train")


class EpochImageLogger(Callback):
    """
    Minimal epoch-end image saver (PNG only).

    - Saves up to `max_images` from the last seen batch each epoch
    - Groups outputs under save_dir/image_log/<split>/epoch_XXXX/
    - Does NOT write to TensorBoard — PNG files only
    """

    def __init__(self, max_images: int = 5, clamp: bool = True, rescale: bool = True,
                 disabled: bool = False, log_images_kwargs=None, split: str = "train",
                 save_text_prompts: bool = False, prompts_key: str = "global_prompt",
                 prompts_filename: str = "global_prompts.txt"):
        super().__init__()
        self.max_images = max_images
        self.clamp = clamp
        self.rescale = rescale
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.split = split
        self.save_text_prompts = save_text_prompts
        self.prompts_key = prompts_key
        self.prompts_filename = prompts_filename

        # cached batch for this epoch
        self._last_batch = None
        self._last_batch_idx = None

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        # group by epoch subfolder
        try:
            epoch_offset = int(os.environ.get("EPOCH_OFFSET", "0"))
        except Exception:
            epoch_offset = 0
        effective_epoch = current_epoch + epoch_offset
        root = os.path.join(save_dir, "image_log", split, f"epoch_{effective_epoch + 1:04d}")
        for idx, key in enumerate(sorted(images.keys()), start=1):
            grid = torchvision.utils.make_grid(images[key], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "b-{:06}_{:02d}_{}_gs-{:06}_e-{:06}.png".format(
                batch_idx, idx, key, global_step, effective_epoch
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # keep last seen batch for this epoch
        if self.disabled:
            return
        self._last_batch = batch
        self._last_batch_idx = batch_idx

    def on_train_epoch_end(self, trainer, pl_module):
        if self.disabled or self._last_batch is None:
            return
        # Only rank 0 performs image logging to avoid accessing logger on other ranks
        try:
            if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
                return
        except Exception:
            return

        switched_to_eval = False
        try:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
                switched_to_eval = True

            with torch.no_grad():
                # ask model to assemble loggable images; cap to max_images
                images = pl_module.log_images(self._last_batch, N=self.max_images, split=self.split, **self.log_images_kwargs)

                if not isinstance(images, dict) or len(images) == 0:
                    return

                # truncate to max_images, clamp and move to cpu for saving
                for k in list(images.keys()):
                    try:
                        N = min(images[k].shape[0], self.max_images)
                        images[k] = images[k][:N]
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)
                    except Exception:
                        # drop problematic entry
                        images.pop(k, None)

            if images:
                self.log_local(pl_module.logger.save_dir, self.split, images,
                               pl_module.global_step, pl_module.current_epoch, self._last_batch_idx or 0)

                # Optionally save text prompts as a .txt file alongside images
                try:
                    if self.save_text_prompts and isinstance(self._last_batch, dict):
                        prompts = self._last_batch.get(self.prompts_key)
                        if isinstance(prompts, (list, tuple)) and len(prompts) > 0:
                            N = min(len(prompts), self.max_images)
                            try:
                                epoch_offset = int(os.environ.get("EPOCH_OFFSET", "0"))
                            except Exception:
                                epoch_offset = 0
                            effective_epoch = pl_module.current_epoch + epoch_offset
                            root = os.path.join(pl_module.logger.save_dir, "image_log", self.split,
                                                f"epoch_{effective_epoch + 1:04d}")
                            os.makedirs(root, exist_ok=True)
                            path = os.path.join(root, self.prompts_filename)
                            with open(path, 'w', encoding='utf-8') as f:
                                for i in range(N):
                                    f.write(str(prompts[i]).strip() + "\n")
                except Exception as _e:
                    pass
        except Exception as e:
            # non-fatal logging failure
            try:
                rank = getattr(trainer, "global_rank", 0)
            except Exception:
                rank = 0
            if rank == 0:
                print(f"[Warn][EpochImageLogger] Failed to log images at epoch {trainer.current_epoch}: {e}")
        finally:
            if switched_to_eval:
                pl_module.train()
