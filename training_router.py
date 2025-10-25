import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from tqdm import tqdm

from fcdiffusion.model import create_model, load_state_dict
from fcdiffusion.router import PromptComplexityRouter

class PromptComplexityDataset(Dataset):
    def __init__(self, json_file):
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        score = torch.tensor(item["complexity_score"], dtype=torch.float32)
        return prompt, score

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading and freezing FCDiffusion model for text encoding...")
    fcd_model = create_model(args.fcd_config).to(device)
    fcd_model.load_state_dict(load_state_dict(args.fcd_ckpt, location='cpu'))
    fcd_model.eval()
    for param in fcd_model.parameters():
        param.requires_grad = False


    router = PromptComplexityRouter(embedding_dim=args.embedding_dim).to(device)

    dataset = PromptComplexityDataset(args.router_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    optimizer = optim.AdamW(router.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("\n--- Starting Router Training ---")
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0
        

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for prompts, scores in progress_bar:
            scores = scores.to(device).unsqueeze(1)

            with torch.no_grad():

                prompt_embeddings = fcd_model.get_learned_conditioning(list(prompts)).to(device)
            predicted_scores = router(prompt_embeddings)

            loss = criterion(predicted_scores, scores)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.6f}")

    torch.save(router.state_dict(), args.save_path)
    print(f"\nTraining finished. Router model saved to '{args.save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Prompt Complexity Router.")
    parser.add_argument('--fcd_config', type=str, default='configs/model_config.yaml', help='Path to the FCDiffusion model config file.')
    parser.add_argument('--fcd_ckpt', type=str, required=True, help='Path to your trained FCDiffusion checkpoint (e.g., FCDiffusion_ini.ckpt).')
    
    parser.add_argument('--router_data', type=str, default='data_route_training.json', help='Path to the preprocessed router training data.')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Dimension of text embeddings (1024 for SD 2.1, 768 for SD 1.5).')
    parser.add_argument('--save_path', type=str, default='models/prompt_router.ckpt', help='Path to save the trained router model.')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    
    args = parser.parse_args()
    train(args)
