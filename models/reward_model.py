

# TODO: Rule-Based Reward Function (Math)
# This is a placeholder for the math reward model.
# The reward function is based on the countdown function which takes both the format and the correctness of the answer
# https://github.com/kanishkg/cognitive-behaviors/blob/main/verl/utils/reward_score/countdown.py#L59


# -------------------------------
# This is the reward model for preference-based RLHF (RLOO).
# A Siamese DistilBert model will be used to compute the reward score.

import torch
from transformers import DistilBertTokenizer
from reward_DistilBERT import SiameseRewardModel

def load_reward_model(model_path, device=None):
    """
    Load a pre-trained SiameseRewardModel
    
    Args:
        model_path (str): Path to the saved model weights
        device (torch.device, optional): Device to load the model on. 
                                        Defaults to cuda if available, else cpu.
    
    Returns:
        SiameseRewardModel: Loaded and initialized model
    """
    # Determine device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SiameseRewardModel().to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    
    return model

def score_responses(model, tokenizer, prompts, responses, device=None, max_length=512):
    """
    Score a batch of [prompt, response] pairs
    
    Args:
        model (SiameseRewardModel): Loaded reward model
        tokenizer (DistilBertTokenizer): Tokenizer for encoding
        prompts (list): List of prompts
        responses (list): List of corresponding responses
        device (torch.device, optional): Device to run inference on
        max_length (int, optional): Maximum sequence length
    
    Returns:
        torch.Tensor: Reward scores for each [prompt, response] pair
    """
    # Determine device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare inputs
    texts = [f"{prompt} [SEP] {response}" for prompt, response in zip(prompts, responses)]
    
    # Tokenize
    tokens = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    ).to(device)
    
    # Score responses
    with torch.no_grad():
        rewards = model.forward(**tokens)
    
    return rewards

def main():
    """
    Example usage of loading and scoring responses
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = load_reward_model('reward_model.pth', device)
    
    # Example prompts and responses
    prompts = [
        'How do I cook pasta?',
        'What is machine learning?'
    ]
    responses = [
        'Boil water, add salt, cook pasta for 8-10 minutes until al dente.',
        'Machine learning is a subset of AI where algorithms learn from data.'
    ]
    
    # Score responses
    scores = score_responses(model, tokenizer, prompts, responses, device)
    
    # Print scores
    for prompt, response, score in zip(prompts, responses, scores):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Score: {score.item()}\n")

if __name__ == "__main__":
    main()