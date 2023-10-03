import time
import numpy as np
import torch

class Range:
  def __init__(self, low, high):
    self.low, self.high = low, high

  def __repr__(self):
    return f"[{self.low}, {self.high})"
  
SCALE_FACTOR = 1e3
MIN_PROB = 1/SCALE_FACTOR

class LlmModel:
  """
  """

  def __init__(self, probs: torch.Tensor):
    ## probability: nseq x vocab_size
    symbols = list(np.arange(probs.shape[1]))

    self.name = "LlmProb"
    self.symbols = symbols
    probs = probs.clone()
    probs[probs<MIN_PROB] = MIN_PROB
    self.__prob = probs
    self.cdf_buffer = [Range(0,0)] * probs.shape[1]

  def cdf_loop(self, loc: int):
    # compute cdf from given probability    
    cdf = {}
    prev_freq = 0
    probability = self.__prob[loc].numpy().tolist()  
    for sym, prob in enumerate(probability):
      freq = round(SCALE_FACTOR * prob)
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    self.cdf_object = cdf        
    return self.cdf_object
  
  def cdf_numpy(self, loc: int):
    # compute cdf from given probability    
    cdf = self.cdf_buffer

    probability = self.__prob[loc].numpy()
    prev_freq = 0
    freqs = np.round(SCALE_FACTOR * probability).astype(np.int32)
    cumufreq = freqs.cumsum()   
    t0 = time.perf_counter()          
    for sym, freq in enumerate(cumufreq):      
      cdf[sym] = Range(prev_freq, freq)
      prev_freq = freq
    t1 = time.perf_counter()
    # print("time spent on creating cdf: ", t1-t0)
    self.cdf_object = cdf        
    return self.cdf_object

  def cdf(self, loc:int):
    return self.cdf_numpy(loc)
  
  def probability(self):
    raise NotImplementedError()
    # return self.__prob

  def predict(self, loc: int, symbol: int):
    assert symbol < len(self.symbols)
    return self.__prob[loc, symbol]

