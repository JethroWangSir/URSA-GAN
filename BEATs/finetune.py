import os
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset import HAT_ESC
from BEATs import BEATs, BEATsConfig

NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.1
NUM_TRG_CLASSES = 40

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.squeeze(1)
            _, outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pretrained_model_path = 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    checkpoint = torch.load(pretrained_model_path)

    # print('Keys in checkpoint['model']:')
    # for key in checkpoint['model'].keys():
    #     print(key)

    if 'predictor.weight' in checkpoint['model']:
        checkpoint['model'].pop('predictor.weight')
    if 'predictor.bias' in checkpoint['model']:
        checkpoint['model'].pop('predictor.bias')

    cfg = BEATsConfig(
        {
            **checkpoint['cfg'],
            'predictor_class': NUM_TRG_CLASSES
        }
    )
    netB = BEATs(cfg).to(device)
    netB.load_state_dict(checkpoint['model'], strict=False)

    trainset = HAT_ESC('/share/nas169/jethrowang/URSA-GAN/data/BEATs')
    trainset_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(netB.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)

    num_training_steps = len(trainset_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_PROPORTION)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(NUM_EPOCHS):
        netB.train()
        running_loss = 0.0
        progress_bar = tqdm(trainset_dataloader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]', unit='batch')

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.squeeze(1)

            optimizer.zero_grad()
            _, outputs, _ = netB(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            scheduler.step()

            running_loss += loss.item() * inputs.size(0)

            current_lr = optimizer.param_groups[0]['lr']

        epoch_loss = running_loss / len(trainset)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {epoch_loss:.4f}')

        valset = HAT_ESC('/share/nas169/jethrowang/URSA-GAN/data/BEATs')
        valset_dataloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
        val_accuracy = validate(netB, valset_dataloader, device)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Accuracy: {val_accuracy:.4f}')

    checkpoint = {
        'cfg': cfg.__dict__,
        'model': netB.state_dict()
    }

    torch.save(checkpoint, f'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2_trg_noisy_{NUM_EPOCHS}epochs.pt')
    print(f'Checkpoint saved successfully.')

if __name__ == '__main__':
    main()
