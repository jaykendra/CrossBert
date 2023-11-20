from finetune import *

checkpoint = torch.load(save_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
# You might also want to load optimizer and scheduler states if needed
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
model.eval()  # Set the model to evaluation mode
