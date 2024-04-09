import torch
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fp = 'Shakespeare.txt' if torch.cuda.is_available() else 'Shakespeare.txt'
blk_size = 64
batch_size = 128
lr = 3e-4
iters = 5000
eval_iters = 250
n_emb = 256
n_layers = 8
n_heads = 8
dropout = 0.2
head_size = n_emb // n_heads

with open(fp, 'r',encoding='utf-8') as f:
  text = f.read()
  chars = sorted(set(text))
  vocab_size = len(chars)

str2int = { ch:i for i,ch in enumerate(chars)}
int2str = { i:ch for i,ch in enumerate(chars)}
enc = lambda x: [str2int[i] for i in x]
dec = lambda x: [int2str[i] for i in x]

data = torch.tensor(enc(text), dtype=torch.long)
n = int(0.8 * len(data))
dataTrain = data[:n].to(device)
dataTest = data[n:].to(device)

class Head(torch.nn.Module):
  i = 0
  def __init__(self, head_size) -> None:
    super().__init__()
    self.key = torch.nn.Linear(n_emb, head_size, bias=False)
    self.query = torch.nn.Linear(n_emb, head_size, bias=False)
    self.value = torch.nn.Linear(n_emb, head_size, bias=False)
    self.register_buffer('tril',torch.tril(torch.ones((blk_size, blk_size))))
    self.sm = torch.nn.Softmax(dim=-1)
    self.drop = torch.nn.Dropout(dropout)


  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2,-1) * (k.shape[-1]**(-0.5))
    Head.i += 1
    # print(Head.i)
    # print(f'WEi {wei.shape} | Trill : {self.tril.shape}')
    wei = wei.masked_fill(self.tril[:T][:T]==0, float('-inf'))
    wei = self.sm(wei)
    wei = self.drop(wei)
    v = self.value(x)
    op = wei @ v
    return op

class MultiheadAttention(torch.nn.Module):
  def __init__(self, n_heads, head_size) -> None:
    super().__init__()
    self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.drop = torch.nn.Dropout(dropout)
    self.Linear = torch.nn.Linear(n_heads*head_size, n_emb)

  def forward(self,x):
    cat = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.drop(self.Linear(cat))


class Block(torch.nn.Module):
  def __init__(self, n_emb, n_heads) -> None:
    super().__init__()
    head_size = n_emb // n_heads
    self.mulAtt = MultiheadAttention(n_heads,head_size)
    self.n1 = torch.nn.LayerNorm(n_emb)
    self.n2 = torch.nn.LayerNorm(n_emb)
    self.ff = torch.nn.Sequential(torch.nn.Linear(n_emb, 4*n_emb),torch.nn.ReLU(),torch.nn.Linear(4*n_emb, n_emb),torch.nn.Dropout(dropout))

  def forward(self,x):
    y = self.mulAtt(x)
    x = x + y
    x = self.n1(x)
    y = self.ff(x)
    x = x + y
    x = self.n2(x)
    return x

class GPT_LLM(torch.nn.Module):
  def __init__(self, vocab_size) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.embedT = torch.nn.Embedding(vocab_size, n_emb)
    self.embedPos = torch.nn.Embedding(blk_size, n_emb)
    self.decBlock = torch.nn.Sequential(*[Block(n_emb, n_heads) for _ in range(n_layers)])
    self.norm = torch.nn.LayerNorm(n_emb)
    self.lm_head = torch.nn.Linear(n_emb, vocab_size)
    self.sm = torch.nn.Softmax(-1)
    self.apply(self._init_weights)

  def _init_weights(self,module):
    if isinstance(module, torch.nn.Linear):
      torch.nn.init.normal_(module.weight,mean = 0.0 , std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, ip, op = None):
    B,T = ip.shape
    log0 = self.embedT(ip)
    log1 = self.embedPos(torch.arange(T, device=device))
    # print(log0.shape, log1.shape)
    logits = log0 + log1
    logits = self.decBlock(logits)
    logits = self.lm_head(self.norm(logits))
    # print(f'Pre Logits shape {logits.shape}')
    if op is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      # print(f'Post Logits shape {logits.shape}')
      op = op.view(B*T)
      loss = torch.nn.functional.cross_entropy(logits,op)
    return logits, loss

  def genTokens(self, iter , ip):
    for _ in range(iter):
      ip_crop = ip[:,-blk_size:]
      logits, loss = self.forward(ip_crop)
      logits = logits[:,-1,:]
      probs = torch.nn.functional.softmax(logits,dim=-1)
      nxt = torch.multinomial(probs, num_samples=1)
      ip = torch.cat((ip,nxt),dim=1)
    return ip

#Tensor Shapes breakdown : model.forward() | B: Batch Size  T: TimeStamp / Block_Size  C: Char Embedding
#Input: [B,T] --(text embed)--> [B,T,C] + [T] --(pos embed)--> [T,C] --> [B,T,C]
#  Decoder Block : (
#    MultiHeadAttention : (
#      concatenate n_heads at last dim (
#        Head : 4 k,q,v - {ip[B,T,C] -> [B,T,headSize]}
#         : wei : q[B,T,headSize] @ k.T[B,headSize,T] --> [B,T,T] --masked fill(0=>-inf) / softmax /dropout--> [B,T,T]
#         : op : wei[B,T,T] @ v[B,T,headsize] => [B,T,headSize]
#       ) - [B,T,headSize]*n --> [B,T,headSize*n=C] --Linear / dropout--> [B,T,C]
#    ) - mulAtt[B,T,C] + x[B,T,C] --NormLayer--> [B,T,C] + FeedForward[B,T,C] --NormLayer--> [B,T,C]
#  ) - [B,T,C] --NormLayer / Linear--> [B,T,Vocab_size]
# if op is not None : logits[B,T,Vocab_size] --View--> [B*T,Vocab_size] & op[B,T] --> [B*T]
# loss = crossEntropy( logits[B*T,Vocab_size]  and op[B*T])
# return logits[B*T, Vocab_size], loss[1]

model = GPT_LLM(vocab_size).to(device)
model.load_state_dict(torch.load(f='gpt0.pt', map_location=torch.device(device)))

while True:
  n = int(input("Number of words: \n"))
  prompt = input("Prompt:\n")
  context = torch.tensor(enc(prompt), dtype=torch.long, device=device).unsqueeze(dim=0)
  res = model.genTokens(n,context)
  str = ''
  for i in dec(res[0].tolist()):
    str += i 
  print(f'Completion! \n {str}')
