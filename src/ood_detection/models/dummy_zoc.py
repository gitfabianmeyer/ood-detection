from typing import Tuple

import clip
import torch.cuda
from ood_detection.config import Config
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ood_detection.models.captioning_utils import generate_beam, generate2

use_beam_search = True
model_path = ""

device = Config.DEVICE

T = torch.Tensor
D = device
CPU = torch.device('cpu')


class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.model(x)


class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # memory issues?
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length)


class CaptionGenerator:
    def __init__(self,
                 model_path,
                 clip_model,
                 tokenizer,
                 prefix_length: int,
                 prefix_size: int = 512,
                 device=D):
        self.model = ClipCaptionModel(prefix_length, prefix_size)
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.device = device
        self.prefix_length = prefix_length
        self._init_model()

    def _init_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=CPU))
        self.model.eval()
        self.model = self.model.to(self.device)

    def generate_caption(self, image):
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            if use_beam_search:
                generated_text_prefix = generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0]
            else:
                generated_text_prefix = generate2(self.model, self.tokenizer, embed=prefix_embed)
            print('\n')
            print(generated_text_prefix)
            return generated_text_prefix
