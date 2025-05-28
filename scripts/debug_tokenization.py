from scripts.dataloader import select_from_smoltalk, tokenize_SmolTalk_sft_batch

# First, convert the synthetic dataset to the expected format
synthetic_dataset = [
    {
        "messages": [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do you make an omelette?"},
            {"role": "assistant", "content": "Beat some eggs, pour into a pan, cook until firm, and fold."}
        ]
    },
]

# Apply select_from_smoltalk to extract prompt-response pairs
extracted_examples = [select_from_smoltalk(example) for example in synthetic_dataset]

# Convert to batch dictionary format
batched_input = {
    "prompt": [e["prompt"] for e in extracted_examples],
    "response": [e["response"] for e in extracted_examples],
}

# Tokenize and print the tokenized results
tokenized_dataset = tokenize_SmolTalk_sft_batch(batched_input)

print("\nTokenized dataset:")
for i in range(len(batched_input["prompt"])):
    print(f"\nExample {i + 1}:")
    print("Prompt:", batched_input["prompt"][i])
    print("Response:", batched_input["response"][i])
    print("Input IDs:", tokenized_dataset["input_ids"][i])
    print("Attention Mask:", tokenized_dataset["attention_mask"][i])
    print("Position IDs:", tokenized_dataset["position_ids"][i])
    print("Labels:", tokenized_dataset["labels"][i])
