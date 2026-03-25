import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = r"E:\deep-learning\video\KPPCwmU5OHQ_424_430.avi"
MODEL_PATH = r"E:\deep-learning\video\new\model_epoch20 (1).pth"

FIXED_FRAMES = 60
NUM_QUERIES = 32
QFORMER_LAYERS = 4

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()



def load_video_frames(video_path, num_frames=FIXED_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        raise ValueError("视频读取失败")

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()


    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


# ================= 提取CLIP特征 =================
def extract_clip_features(frames):
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)

    return feats  # (T, 512)


# ================= Q-Former =================
class QFormerBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, visual_feats, feat_mask):
        q = self.norm1(q + self.attn(q, q, q)[0])

        attn_out, _ = self.cross_attn(
            q, visual_feats, visual_feats,
            key_padding_mask=(feat_mask == 0)
        )
        q = self.norm2(q + attn_out)

        return self.norm3(q + self.ffn(q))


class QFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, NUM_QUERIES, 512))
        self.blocks = nn.ModuleList([
            QFormerBlock(512) for _ in range(QFORMER_LAYERS)
        ])

    def forward(self, visual_feats, feat_mask):
        B = visual_feats.size(0)
        q = self.queries.expand(B, -1, -1)

        for block in self.blocks:
            q = block(q, visual_feats, feat_mask)

        return q


# ================= 主模型 =================
class VideoCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.temp_emb = nn.Parameter(torch.randn(1, FIXED_FRAMES, 512))

        self.qformer = QFormer()

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 768),
            nn.LayerNorm(768)
        )

        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, visual_feats):
        visual_feats = visual_feats + self.temp_emb

        feat_mask = (visual_feats.abs().sum(dim=-1) != 0).float()

        q_feats = self.qformer(visual_feats, feat_mask)

        prefix = self.fc(q_feats)

        return prefix


# ================= 生成 =================
def generate_caption(
    model,
    feat,
    tokenizer,
    temperature=0.3,
    top_k=50,
    top_p=0.5,
    max_new_tokens=30
):
    model.eval()

    with torch.no_grad():
        feat = feat.unsqueeze(0).to(DEVICE)

        prefix = model(feat)

        attn_mask = torch.ones((1, NUM_QUERIES), device=DEVICE)

        output_ids = model.gpt2.generate(
            inputs_embeds=prefix,
            attention_mask=attn_mask,


            do_sample=True,


            temperature=temperature,
            top_k=top_k,
            top_p=top_p,


            max_new_tokens=max_new_tokens,


            repetition_penalty=1.2,
            no_repeat_ngram_size=2,

            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return text



if __name__ == "__main__":

    print("读取视频...")
    frames = load_video_frames(VIDEO_PATH)

    print("提取CLIP特征...")
    feats = extract_clip_features(frames)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = VideoCaptionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("生成描述...")
    caption = generate_caption(model, feats, tokenizer)

    print("\n Caption:")
    print(caption)