import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.GANmigru3bw.train_gan import train_gan  # updato for your local setup

if __name__ == "__main__":
    # Paths relative to project root.
    # Update these paths locally before running the code.

    data_root = "outputs/imagesGIDS/shell_ag/aug_gan"

    output_dir = "outputs/gan_training_new/CH15_15_bw2/aug_gan"


    train_gan(
        data_root=data_root,
        output_dir=output_dir,
        num_epochs=100,
        batch_size=128,
        lr=0.0001,
        image_size=(16, 305),
        nz=100,
        use_subset=False,
        subset_ratio=0.1,
        critic_iters=5,
        save_every=5,
        update_fid_per_batch=False,
    )
