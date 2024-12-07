import torch
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model
import einops

from diffusers.models.normalization import AdaGroupNorm

from timm.layers import use_fused_attn
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, get_act_layer

class Attention(nn.Module):
    fused_attn = True

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVDNoiseUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=128): # resolution = size // 8
        super(SVDNoiseUnet, self).__init__()

        _in = int(resolution * in_channels // 2)
        _out = int(resolution * out_channels // 2)
        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )

        self.attention = Attention(_out)

        self.bn = nn.BatchNorm2d(_out)

        self.mlp4 =  nn.Sequential(
            nn.Linear(_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out),
        )

    def forward(self, x, residual=False):
        b, c, h, w = x.shape
        x = einops.rearrange(x, "b (a c)h w ->b (a h)(c w)", a=2,c=2) # x -> [1, 256, 256]
        U, s, V = torch.linalg.svd(x) # U->[b 256 256], s-> [b 256], V->[b 256 256]
        U_T = U.permute(0, 2, 1)
        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1) # s -> [b, 1, 256]  => [b, 256, 256]
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        pred = U @ torch.diag_embed(out) @ V
        return einops.rearrange(pred, "b (a h)(c w) -> b (a c) h w", a=2,c=2)
    

class SVDNoiseUnet_Concise(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=128):
        super(SVDNoiseUnet_Concise, self).__init__()


class NoiseTransformer(nn.Module):
    def __init__(self, resolution=128):
        super().__init__()
        self.upsample = lambda x: F.interpolate(x, [224,224])
        self.downsample = lambda x: F.interpolate(x, [resolution,resolution])
        self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.downconv = nn.Conv2d(4,3,(1,1),(1,1),(0,0))
        # self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.swin = create_model("swin_tiny_patch4_window7_224",pretrained=True)


    def forward(self, x, residual=False):
        if residual:
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x))))) + x
        else:
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x)))))

        return x

class NPNet(nn.Module):
      def __init__(self, model_id, pretrained_path=True, device='cuda') -> None:
            super(NPNet, self).__init__()

            assert model_id in ['SDXL', 'DreamShaper', 'DiT']

            self.model_id = model_id
            self.device = device
            self.pretrained_path = pretrained_path

            (
                  self.unet_svd, 
                  self.unet_embedding, 
                  self.text_embedding, 
                  self._alpha, 
                  self._beta
             ) = self.get_model()

      def get_model(self):

            unet_embedding = NoiseTransformer(resolution=128).to(self.device).to(torch.float32)
            unet_svd = SVDNoiseUnet(resolution=128).to(self.device).to(torch.float32)

            if self.model_id == 'DiT':
                  text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32)
            else:
                  text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32) 

            
            if '.pth' in self.pretrained_path:
                  gloden_unet = torch.load(self.pretrained_path)
                  unet_svd.load_state_dict(gloden_unet["unet_svd"])
                  unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
                  text_embedding.load_state_dict(gloden_unet["embeeding"])
                  _alpha = gloden_unet["alpha"]
                  _beta = gloden_unet["beta"]

                  print("Load Successfully!")

                  return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            
            else:
                  assert ("No Pretrained Weights Found!")
            

      def forward(self, initial_noise, prompt_embeds):

            prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
            text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

            golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                        2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

            return golden_noise

class NPNetGoldenNoise:
    npnet = None
    noise = None
    cond = None
    seed = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "prompt": ("CONDITIONING",),
                "model_path": ("STRING",{"default": "/path/to/sdxl.pth"}),
                "model_type": (["SDXL", "DreamShaper", "DiT"],),
            }
        }

    RETURN_TYPES = ("NOISE",)
    CATEGORY = "_for_testing/golden_noise"

    FUNCTION = "doit"

    def generate_noise(self, input_latent):
        seed = self.noise.seed
        init_noise = self.noise.generate_noise(input_latent).to('cuda')
        cond = self.cond[0].clone().to('cuda')
        self.npnet.to('cuda')
        try:
            print("Applying NPNet to noise")
            r = self.npnet(init_noise, cond)
        except Exception as e:
            print("Running NPNet failed with error (non-square latent can cause shape errors):", e)
            print("Returning unmodified noise")
            return init_noise
        print("NPNet ran ok")
        return r.to("cpu")

    def doit(self, noise, prompt, model_path, model_type):
        if self.npnet is None:
            print("Loading NPNet")
            self.npnet = NPNet(model_type, model_path)
        self.noise = noise
        self.cond = prompt[0]

        return (self,)




NODE_CLASS_MAPPINGS = {"NPNetGoldenNoise": NPNetGoldenNoise}