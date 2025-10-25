import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from fcdiffusion.dataset import TrainDataset
from fcdiffusion.logger import ImageLogger
from fcdiffusion.model import create_model, load_state_dict
import torch
torch.cuda.set_device(0)

# Configs
resume_path = 'path/to/your/init ckpt'
batch_size = 2
logger_freq = 2000
learning_rate = 1e-5
sd_locked = True
val_every_n_train_steps = 1600


model = create_model('configs/model_config.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
control_mode = model.control_mode
logger_root_path = 'fcdiffusion_' + control_mode + '_img_logs_gmea'
checkpoint_path = 'fcdiffusion_' + control_mode + '_checkpoint_gmea'

with open("model_arch_origin",'w') as f:
    f.write(f"{model}")


dataset = TrainDataset('./datasets/training_data.json', cache_size=100)

val_split = 0.05
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=batch_size, shuffle=False)

logger = ImageLogger(root_path=logger_root_path, batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs/' + checkpoint_path,
    filename='best-model-{epoch}-{step}-{val_loss:.4f}',
    monitor='val/loss',
    mode='min',
    save_top_k=1,
    every_n_train_steps=val_every_n_train_steps,
    save_last=True, 

)

trainer = pl.Trainer(
    gpus='0', 
    precision=32, 
    callbacks=[logger, checkpoint_callback],
    val_check_interval=val_every_n_train_steps,
)


trainer.fit(model, train_dataloader, val_dataloader)