import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from process_puz import df

class QADataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = str(self.df.iloc[idx]['Clue'])
        answer = str(self.df.iloc[idx]['Ans'])

        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=512,  # Adjust max length as needed
            return_tensors='pt',
            truncation=True,
            padding = 'max_length'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor([0]),  # Placeholder, update as needed
            'end_positions': torch.tensor([1])  # Placeholder, update as needed
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = QADataset(tokenizer, df)

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

num_epochs = 3  # Adjust as needed
total_steps = len(dataset) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

save_checkpoint_path = "crossbert.pth"  # Change the path as needed
save_interval = 10  # Adjust as needed, e.g., save every 100 batches

batch_count = 0

for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_count += 1
        if batch_count % save_interval == 0:
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'batch_count': batch_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }, save_checkpoint_path)
