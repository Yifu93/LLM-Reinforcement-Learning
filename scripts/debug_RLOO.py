from scripts.dataloader import get_ultrafeedback_dataset

data_path = "./data/ultrafeedback_binarized/train_gen"

print("Loading dataset...")
dataset = get_ultrafeedback_dataset(data_path)
print(dataset[0])
# print(dataset.features)

# def wrap_reward_model(reward_model, reward_tokenizer, device):
#     reward_model.to(device)
#     reward_model.eval()

#     def compute_reward(flat_list):

#         prompt_response_pairs = flat_list
#         print(f"Total elements: {len(prompt_response_pairs)}")
#         for i in range(0, len(prompt_response_pairs), 2):
#             print(f"\nPair {i // 2}:")
#             print(f"  Prompt   : {prompt_response_pairs[i][:80]}...")
#             print(f"  Response : {prompt_response_pairs[i+1][:80]}...")


#         assert len(flat_list) % 2 == 0, "List length must be even"
#         num_pairs = len(flat_list) // 2
#         batch_size = BATCH_SIZE
#         k = NUM_GEN_PER_PROMPT
#         assert num_pairs == batch_size * k, f"Expected {batch_size*k} pairs but got {num_pairs}"

#         # Rebuild prompt-response pairs in order
#         prompt_response_pairs = [
#             (flat_list[i], flat_list[i + 1]) for i in range(0, len(flat_list), 2)
#         ]

#         prompts, responses = zip(*prompt_response_pairs)

#         with torch.no_grad():
#             inputs = reward_tokenizer(
#                 list(prompts),
#                 list(responses),
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True
#             ).to(device)

#             outputs = reward_model(**inputs)
#             rewards = outputs.logits.squeeze(-1) 
#             print(f"Computed rewards: {rewards}")
#         return rewards

#     return compute_reward