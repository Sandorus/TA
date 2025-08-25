import torch
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
print(torch.version.cuda)  # or None if CPU only
