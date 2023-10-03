import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_compressor import AECompressorLLM

def compression_ratio(msg, tokenizer, model):
  batch = tokenizer(msg, return_tensors="pt")
  input_ids = batch.input_ids
  with torch.no_grad():
    logits = model(**batch).logits.squeeze()
  probs = torch.softmax(logits, dim=1)
  uniform_prob = torch.ones(probs.shape[1]) / probs.shape[1]
  next_token_probs = torch.concat([uniform_prob.unsqueeze(0), probs[:-1, :]], dim=0)

  compressor = AECompressorLLM()
  data_ids = input_ids.squeeze().tolist()

  msg = compressor.compress(data_ids, next_token_probs)
  recon = compressor.decompress(msg, len(data_ids), next_token_probs)
  assert all(a==b for a, b in zip(recon, data_ids))
  msg_len = len(msg)
  data_len = len(data_ids) * 16
  print(f"message length: {msg_len} bits")
  print(f"data length: {data_len} bits")
  print(f"compress ratio: {msg_len/data_len:.4f}")
  return msg_len/data_len

def test_english():  
  tokenizer = AutoTokenizer.from_pretrained("gpt2")  
  model = AutoModelForCausalLM.from_pretrained("gpt2")
  
  test_sentence = "This is a test sentence."
  scrambled = "sentence a this is Test."
  letter_scrambled = "a net h t.eTtisc nsseies"
  cr1 = compression_ratio(test_sentence, tokenizer, model)
  cr2 = compression_ratio(scrambled, tokenizer, model)
  cr3 = compression_ratio(letter_scrambled, tokenizer, model)

  assert cr1 < cr2 < cr3
  



  

