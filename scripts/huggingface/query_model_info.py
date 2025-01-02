from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rich import print


def get_full_model_info(model_name, tokenizer_name=None):
    """
    Load a model and tokenizer, create an NLP pipeline, and extract model info.
    
    Args:
        model_name (str): The name or path of the model.
        tokenizer_name (str): The name or path of the tokenizer (optional, defaults to model_name).
    
    Returns:
        dict: Model properties (including NLP pipeline, tokenizer, and model).
    """
    tokenizer_name = tokenizer_name or model_name

    print(f"Loading Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading Model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    print(f"Creating NLP pipeline for {model_name}")
    nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    # Extract model properties
    model_properties = {
        "nlp_pipeline": nlp_pipeline,  # for use in text processing
        "model": model,               
        "tokenizer": tokenizer,       
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "base_model": model.config.architectures if hasattr(model.config, "architectures") else "Unknown",
        "model_type": model.config.model_type if hasattr(model.config, "model_type") else "Unknown",
        "num_labels": model.config.num_labels if hasattr(model.config, "num_labels") else "Unknown",
        "tokenizers_max_token_length": tokenizer.model_max_length,
        "supported_entities": sorted(
            set(label.split("-")[-1] for label in model.config.id2label.values() if label != "O")
        ),
    }

    # Print model properties
    print("\n[bold underline]Model Properties[/bold underline]")
    for prop, value in model_properties.items():
        if prop not in ["nlp_pipeline", "model", "tokenizer", "supported_entities"]:
            print(f"{prop.replace('_', ' ').capitalize()}: {value}")

    print("\n[bold underline]Supported Entities by the Model:[/bold underline]")
    for entity in model_properties["supported_entities"]:
        print(f"• {entity}")
    
    

    return model_properties
