from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
if __name__ in "__main__":
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    fp16 = True
    bf16 = False
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    cp_model = AutoModelForCausalLM.from_pretrained(
        "/mnt/c/Users/thddm/Documents/model/kkt_od_inst/",
        device_map="cpu",
        quantization_config=bnb_config, 
        torch_dtype=compute_dtype
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b",
        device_map="cpu",
        quantization_config=bnb_config, 
        torch_dtype=compute_dtype
    )
    base_model.resize_token_embeddings(256003)
    inst_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="cpu",
        quantization_config=bnb_config, 
        torch_dtype=compute_dtype
    )
    inst_model.resize_token_embeddings(256003)

   

    for k, v in base_model.state_dict().items():
        if v.shape != cp_model.state_dict()[k].shape:
            print(v.shape, cp_model.state_dict()[k].shape)
    
    raise()
    for k, v in cp_model.state_dict().items():
        print(k, v.shape)
        break
    skip_layers = ["model.embed_tokens.weight", "lm_head.weight"]

    for k, v in cp_model.state_dict().items():
        if (k in skip_layers) or ("layernorm" in k):
            continue

        chat_vector = inst_model.state_dict()[k] - base_model.state_dict()[k]
        new_value = cp_model.state_dict()[k] + chat_vector
        v.copy_(new_value)

    for k, v in cp_model.state_dict().items():
        print(k, v)
        break
    cp_model.save_pretrained("/mnt/c/Users/thddm/Documents/model/kkt-simpo-chatvector", safe_serialization=False)