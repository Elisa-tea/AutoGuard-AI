import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from src.models.GANmigru3bw.generator import Generator
from src.models.GANmigru3bw.discriminator import Discriminator


def to_uint8(img_tensor):
    img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.size(1) == 1:
        img_tensor = nn.functional.interpolate(
            img_tensor.float(), size=(75, 75), mode='bilinear', align_corners=False
        )
    if img_tensor.size(0) == 1:
        img_tensor = img_tensor.squeeze(0)
    img_tensor = (img_tensor * 255).to(torch.uint8)
    return img_tensor


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0)
    return torch.stack(batch, 0)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, expected_size=(1, 16, 305)):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform
        self.expected_size = expected_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")  # grayscale
                if img.size != (self.expected_size[2], self.expected_size[1]):
                    print(f"Skipping unexpected size image: {img_path} with size {img.size}")
                    return None
                if self.transform:
                    img = self.transform(img)
                return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None


def gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10.0):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    output, _ = netD(interpolates)

    grads = torch.autograd.grad(
        outputs=output,
        inputs=interpolates,
        grad_outputs=torch.ones_like(output, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty


def to_3channel(img_tensor):
    # Kept here (unused now) in case you re-enable FID later
    if img_tensor.size(1) == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    return img_tensor


def train_gan(
    data_root,
    output_dir,
    num_epochs=100,
    batch_size=64,
    lr=0.0001,
    image_size=(16, 305),
    nz=100,
    use_subset=False,
    subset_ratio=1,
    critic_iters=5,
    save_every=5,
    update_fid_per_batch=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_dataset = CustomImageDataset(
        data_root, transform=transform, expected_size=(1, image_size[0], image_size[1])
    )

    if use_subset:
        random.seed(42)
        subset_size = int(subset_ratio * len(full_dataset))
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        dataset = Subset(full_dataset, subset_indices)
        print(f"Using a subset: {subset_size} images out of {len(full_dataset)}")
    else:
        dataset = full_dataset
        print(f"Using full dataset: {len(dataset)} images")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_skip_none
    )

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate schedulers
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # fid_metric = FrechetInceptionDistance(feature=64).to(device)

    os.makedirs(output_dir, exist_ok=True)

    scalerD = torch.cuda.amp.GradScaler()
    scalerG = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        netG.train()
        netD.train()
        for i, data in enumerate(dataloader, 0):
            if data.size(0) == 0:
                print(f"Skipping empty batch at iteration {i}")
                continue

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # Discriminator
            for _ in range(critic_iters):
                netD.zero_grad()
                with torch.cuda.amp.autocast():
                    output_real, features_real = netD(real_cpu)
                    D_real = output_real.mean()

                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    output_fake, features_fake = netD(fake.detach())
                    D_fake = output_fake.mean()

                    gp = gradient_penalty(netD, real_cpu, fake.detach(), device)

                    errD = D_fake - D_real + gp

                scalerD.scale(errD).backward()
                scalerD.step(optimizerD)
                scalerD.update()

            # Generator
            netG.zero_grad()
            with torch.cuda.amp.autocast():
                output_fake, features_fake = netD(fake)
                errG_adv = -output_fake.mean()

                mean_fake = features_fake.mean(dim=0, keepdim=True)
                mean_real = features_real.detach().mean(dim=0, keepdim=True)

                if mean_fake.shape != mean_real.shape:
                    min_h = min(mean_fake.shape[-2], mean_real.shape[-2])
                    min_w = min(mean_fake.shape[-1], mean_real.shape[-1])
                    mean_fake = mean_fake[..., :min_h, :min_w]
                    mean_real = mean_real[..., :min_h, :min_w]

                feat_loss = nn.functional.mse_loss(mean_fake, mean_real)

                errG = errG_adv + 0.3 * feat_loss

            scalerG.scale(errG).backward()
            scalerG.step(optimizerG)
            scalerG.update()

            """
             if update_fid_per_batch:
                 with torch.no_grad():
                    fake_uint8 = to_uint8(fake).to(device)
                     real_uint8 = to_uint8(real_cpu).to(device)
            
                    fake_uint8_3ch = to_3channel(fake_uint8)
                    real_uint8_3ch = to_3channel(real_uint8)
                
                        fid_metric.update(fake_uint8_3ch, real=False)
                    fid_metric.update(real_uint8_3ch, real=True)
            """

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}][Batch {i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(real): {D_real.item():.4f} D(fake): {D_fake.item():.4f}"
                )

       
        schedulerD.step()
        schedulerG.step()

        
        with torch.no_grad():
            fake_images = netG(fixed_noise)
            save_image(
                (fake_images + 1) / 2,
                os.path.join(output_dir, f"epoch_{epoch:03d}.png"),
                nrow=8,
                normalize=True,
            )

        # checkpoints
        if epoch % save_every == 0 or epoch == num_epochs:
            torch.save(netG.state_dict(), os.path.join(output_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(output_dir, f"netD_epoch_{epoch}.pth"))

