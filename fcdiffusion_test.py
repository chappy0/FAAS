


from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
torch.cuda.set_device(0)

import os
os.environ['CURL_CA_BUNDLE'] = ''


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=True):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


yaml_file_path = "configs/model_config.yaml"

ckpt_file_path = 'path/to/your/checkpoint' 

config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_model_from_config(config, ckpt_file_path, device)

model.eval()


import os  
  
def is_image_file(filename):  
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  
  
def is_text_file(filename):  
    return filename.lower().endswith('.txt')  
  
def traverse_images_and_texts(directory):  
    image_files = []  
    text_contents = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if is_image_file(file):  
                image_files.append(os.path.join(root, file))  
            elif is_text_file(file):
                file_path = os.path.join(root, file)  
                with open(file_path, 'r', encoding='utf-8') as f:  
                    content = f.read()  
                    text_contents.append((content))  
    return image_files, text_contents

def save_image(image, path, rescale=True):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    image = torch.clamp(image, -1, 1)

    if rescale:
        image = (image + 1.0) / 2.0
    
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    Image.fromarray(image).save(path)


directory_path ='path/to/your/source images' 
image_files, text_contents = traverse_images_and_texts(directory_path)  

test_res_num = 1

repath = "path/to/your/reconstruction images"     
if not os.path.exists(repath):
    os.makedirs(repath)
for image_file,text_content in zip(image_files,text_contents):  

    test_img_path,target_prompt = image_file,text_content
    _,reconstruction_img =  os.path.split(test_img_path) 
    reconstruction_img = repath + "/e_" + reconstruction_img 


    dataset = TestDataset(test_img_path, target_prompt, test_res_num)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=True)
    
        
    for step, batch in enumerate(dataloader):

        log = model.log_images(batch, ddim_steps=50)  
        if step == 0:
            reconstruction = log['reconstruction'].squeeze()
            reconstruction = reconstruction.permute(1, 2, 0)
            reconstruction = torch.clamp(reconstruction, -1, 1)
            # Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
            Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)

            # save_image(reconstruction, reconstruction_img)
        

        sample = log['samples'].squeeze()
        # save_image(sample, reconstruction_img)

        sample = sample.permute(1, 2, 0)
        sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
        Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)





