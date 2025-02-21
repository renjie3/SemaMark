from transformers import AutoTokenizer, AutoModelForCausalLM

def get_vocab_embeddings(model_name="bert-base-uncased"):
    """
    Get embeddings for the vocabulary of a BERT model.

    Args:
    - model_name (str): Name of the pre-trained BERT model.
    
    Returns:
    - dict: A dictionary with tokens as keys and their embeddings as values.
    """
    # Load pre-trained BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-2.7b", cache_dir="/mnt/home/renjie3/.cache/huggingface"
        )
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", cache_dir="/mnt/home/renjie3/.cache/huggingface")
    
    # Ensure the model is in eval mode
    model.eval()
    
    # Get the embeddings for all tokens in the tokenizer's vocabulary
    embeddings = model.get_input_embeddings()

    import pdb; pdb.set_trace()

    vocab_embeddings = {}
    for token, index in tokenizer.get_vocab().items():
        vocab_embeddings[token] = embeddings[index].numpy()

    import pdb; pdb.set_trace()

    return vocab_embeddings

if __name__ == "__main__":
    embeddings = get_vocab_embeddings()
    # For demonstration purposes, print the embedding of the word "hello"
    print(embeddings['hello'])

