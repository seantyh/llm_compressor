from . import arithmetic_coding as AE
from .llm_model import LlmModel

# Compress using arithmetic encoding


class AECompressorLLM:
  def __init__(self) -> None:
    pass

  def compress(self, input_ids, probs):

    model = LlmModel(probs)
    encoder = AE.Encoder()

    import time
    for sym_idx, symbol in enumerate(input_ids):
      # Use the model to predict the probability of the next symbol

      # t0 = time.perf_counter()
      probability = model.cdf(sym_idx)
      # t1 = time.perf_counter()
      # encode the symbol
      encoder.encode_symbol(probability, symbol)
      # t2 = time.perf_counter()

      # print(f"{sym_idx}: Time to compute cdf: {t1-t0}")
      # print(f"{sym_idx}: Time to encode symbol: {t2-t1}")

    encoder.finish()
    return encoder.get_encoded()

  def decompress(self, encoded, length_encoded, probs):
    decoded = []
    model = LlmModel(probs)
    decoder = AE.Decoder(encoded)
    for sym_idx in range(length_encoded):
      # probability of the next symbol      
      probability = model.cdf(sym_idx)

      # decode symbol
      symbol = decoder.decode_symbol(probability)

      decoded += [symbol]
    return decoded
