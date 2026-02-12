import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT = "Qwen/Qwen2.5-0.5B-Instruct"



def load_model():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    return tokenizer, model


@torch.inference_mode()
def generate_continuation(tokenizer, model, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a text autocomplete engine. "
                "Continue the user's text naturally and coherently. "
                "Do NOT repeat the user's text. "
                "Do NOT add role labels like 'assistant'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=50,       
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for no_copy_words in ["assistant", "Assistant:", "Assistant", "USER:", "SYSTEM:"]:
        continuation = continuation.replace(no_copy_words, " ")

    return " ".join(continuation.split()).strip()



def main():
    print(f"Loading model: {CHECKPOINT}")
    tokenizer, model = load_model()
    print("Ready. Type a sentence and press Enter. Type 'q' to quit.\n")

    while True:
        s = input("You: ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            break
        if not s:
            continue

        extra = generate_continuation(tokenizer, model, s)
        if extra:
            print(f"Model: {s} {extra}\n")
        else:
            print(f"Model: {s}\n")


if __name__ == "__main__":
    main()
