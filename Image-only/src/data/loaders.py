from torch.utils.data import DataLoader
from src.data.dataset import AQIMultiDataset
from src.data.sampler import NightAwareWeightedSampler

def build_loaders(train_csv, val_csv, test_csv, batch_size=16):

    train_ds = AQIMultiDataset(train_csv, split="train")
    val_ds   = AQIMultiDataset(val_csv, split="val")
    test_ds  = AQIMultiDataset(test_csv, split="test")

    train_sampler = NightAwareWeightedSampler(train_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
