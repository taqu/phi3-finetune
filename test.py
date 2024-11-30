import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, PeftConfig

#model_id = "microsoft/Phi-3.5-mini-instruct"
lora_id = "./phi-3.5-mini-instruct-LoRA/"

last_checkpoint = get_last_checkpoint(lora_id)
config = PeftConfig.from_pretrained(last_checkpoint)
print(config.base_model_name_or_path)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, last_checkpoint)
#for name, module in model.named_modules():
#    print(name)

message = """
<|user|>Create a doxygen comment for the following C++ Function.

void* 
partition(
    void* a,
    int (*cmp)(void const*, void const*),
    size_t sz,
    size_t n
) {
    char* const base = a;
    if (n <= 1) return base + sz;
    char* lo = base;
    char* hi = &base[sz * (n - 1)];
    char* m  = lo + sz * ((hi - lo) / sz / 2);
    if (cmp(lo, m) > 0) {
        swap(lo, m, sz);
    }
    if (cmp(m, hi) > 0) {
        swap(m, hi, sz);
        if (cmp(lo, m) > 0) {
            swap(lo, m, sz);
        }
    }
    while (1) {
        while (cmp(lo, m) < 0) lo += sz;
        while (cmp(m, hi) < 0) hi -= sz;
        if (lo >= hi) return hi + sz;
        swap(lo, hi, sz);
        if (lo == m) {
            m = hi;
        } else if (hi == m) {
            m = lo;
        }
        lo += sz;
        hi -= sz;
    }
}
<|end|>
<|assistant|>"""

input_ids = tokenizer.encode(
    message,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id
]

outputs = model.generate(
    input_ids,
    max_new_tokens=4096,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

