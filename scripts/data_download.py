from datasets import load_dataset

# Download the dataset from Hugging Face Hub
# -- Preference Dataset --
# (1) SmolTalk (Dataset for SFT): https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
# (2) UltraFeedback (Dataset for DPO and RLOO): https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
# -- Verifier-Based Dataset --
# (3) WarmStart (Dataset for SFT): https://huggingface.co/datasets/Asap7772/cog_behav_all_strategies
# (4) PromptsDataset from TinyZero (RLOO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
# (5) On-Policy Preference Dataset (the same as PromptsDataset, DPO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4


# Load dataset once from HF Hub
smoltalk_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
# smoltalk_dataset.save_to_disk("./data/smoltalk")

ultrafeedback_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
# ultrafeedback_dataset.save_to_disk("./data/ultrafeedback_binarized")

warmstart_dataset = load_dataset("Asap7772/cog_behav_all_strategies")
# warmstart_dataset.save_to_disk("./data/warmstart")

Countdown_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
# print(Countdowndataset["train"][0])

