from transformers import AutoModelForCausalLM, AutoTokenizer
import time

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "./Qwen1.5-1.8B-Chat",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained("./Qwen1.5-1.8B-Chat")

print("模型加载完成！")


def get_response(input_prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    model_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return model_response


def get_input():
    # 用户输入文章
    print("请输入文章（输入空行结束）：")

    # 从用户输入中读取多行文章
    lines = []
    while True:
        line = input()  # 读取一行输入
        if not line.strip():  # 如果输入空行，结束输入
            break
        lines.append(line)

    # 将输入内容拼接成完整的文章
    content = "\n".join(lines)

    return content


prompt = "你好！"
while prompt != "<end>":
    response = get_response(prompt)
    print(response)
    prompt = get_input()
    print(f"prompt: {prompt}")
    print("==========promot_end==========\n")


print("对话结束")
