from functions.countdown import compute_score

def score_countdown_math(reply: str, target: int, nums: list[int]) -> float:
    gt = {"target": target, "numbers": nums}
    return compute_score(reply, gt)

# test
prompt = "<|im_start|>assistant<think>\nLet me analyze this step by step:\n\n1. First, let's try to get close to 51 using two numbers:\n- 93 - 43 = 50 (very close to 51!)\n- 93/3 would give us exactly 31\n\n2. Since we need to reach exactly 51, and we found that 93-43=50 is very close, let's see if we can incorporate 1 somehow.\n\n3. Looking at 93-43=50, we could try to add 1 to it:\n- 50+1=51\nThis works! We can write this as (93-43)+1\n\n4. Let's verify:\n93-43 = 50\n50+1 = 51\n\nThis solution uses all numbers exactly once and reaches our target of 51.\n</think>\n\n<answer>(93-43)+1</answer>"
target = 51
nums = [93, 43, 1]

print(score_countdown_math(prompt, target, nums))