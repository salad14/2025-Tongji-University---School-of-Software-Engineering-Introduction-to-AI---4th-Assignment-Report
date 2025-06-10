import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time

model_path = "/mnt/data/DeepSeek-R1-0528-Qwen3-8B" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 加载模型和分词器 ---
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True 
)
print("Tokenizer loaded successfully.")

print(f"Loading model from {model_path}...")
# 将模型移动到GPU并设置为评估模式，以优化推理性能
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True, 
    torch_dtype="auto"
).to(device).eval() 
print("Model loaded successfully.")

test_questions = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    """领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。""",
]

# --- 循环进行问答测试 ---
print(f"\n{'='*60}")
print(f"Starting DeepSeek-R1-0528-Qwen3-8B Inference Tests with {len(test_questions)} questions.")
print(f"{'='*60}\n")

for i, prompt_text in enumerate(test_questions):
    print(f"\n{'#'*10} Test Case {i+1}/{len(test_questions)} {'#'*10}")
    print(f"Question: {prompt_text}")
    print("-" * 30)
    print("DeepSeek-R1-0528-Qwen3-8B's Answer (Streaming Output):")
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    model_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 生成回答
    generated_ids = model.generate(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=300, 
        do_sample=True,      
        temperature=0.7,     
        top_p=0.8,           
        eos_token_id=tokenizer.eos_token_id 
    )
    
    print("\n" + "-" * 30)
    print(f"End of Answer for Test Case {i+1}.")
    print(f"{'#'*10} End of Test Case {i+1} {'#'*10}\n")
    
    time.sleep(3) 

print(f"\n{'='*60}")
print("All DeepSeek-R1-0528-Qwen3-8B Inference Tests Completed!")
print(f"{'='*60}\n")