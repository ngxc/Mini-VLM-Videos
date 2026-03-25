import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import math
from transformers import CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch.nn.functional as F

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./msvd"
VIDEO_DIR = os.path.join(DATA_DIR, "YouTubeClips")
FEAT_DIR = "./msvd_features"
MAX_LEN = 50
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

GEN_TEST_VIDEO = "WeOU0Iba1Xg_1_30.avi"
FIXED_FRAMES = 60
NUM_QUERIES = 32
QFORMER_LAYERS = 4


# ================= 数据集类=================
class MSVDDataset(Dataset):
    def __init__(self, json_paths):
        self.samples = []
        for jp in json_paths:
            with open(jp, "r") as f:
                data = json.load(f)
                for item in data:
                    for cap in item["caption"]:
                        self.samples.append((item["video"], cap))

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 显式设置 pad_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_name, cap = self.samples[idx]
        feat_path = os.path.join(FEAT_DIR, v_name.replace(".avi", ".pt"))
        feat = torch.load(feat_path).to(torch.float32)


        T = feat.shape[0]
        if T > FIXED_FRAMES:
            feat = feat[torch.linspace(0, T - 1, FIXED_FRAMES).long()]
        elif T < FIXED_FRAMES:
            feat = torch.cat([feat, torch.zeros(FIXED_FRAMES - T, 512)], dim=0)

        tokens = self.tokenizer(cap + self.tokenizer.eos_token, padding="max_length",
                                truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return feat, tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)


# ================= 模型部分=================
class QFormerBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, visual_feats, feat_mask):
        q = self.norm1(q + self.attn(q, q, q)[0])
        attn_out, _ = self.cross_attn(q, visual_feats, visual_feats, key_padding_mask=(feat_mask == 0))
        q = self.norm2(q + attn_out)
        return self.norm3(q + self.ffn(q))


class QFormer(nn.Module):
    def __init__(self, dim=512, layers=QFORMER_LAYERS):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, NUM_QUERIES, dim))
        self.blocks = nn.ModuleList([QFormerBlock(dim) for _ in range(layers)])

    def forward(self, visual_feats, feat_mask):
        B = visual_feats.size(0)
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks: q = block(q, visual_feats, feat_mask)
        return q


class VideoCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp_emb = nn.Parameter(torch.randn(1, FIXED_FRAMES, 512))
        self.qformer = QFormer(dim=512)
        self.fc = nn.Sequential(nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 768),
                                nn.LayerNorm(768))
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        for param in self.gpt2.parameters(): param.requires_grad = False
        for param in self.gpt2.transformer.h[-2:].parameters(): param.requires_grad = True

    def forward(self, visual_feats, input_ids=None, attention_mask=None, labels=None):
        visual_feats = visual_feats + self.temp_emb
        feat_mask = (visual_feats.abs().sum(dim=-1) != 0).float()
        q_feats = self.qformer(visual_feats, feat_mask)
        prefix = self.fc(q_feats)
        if input_ids is not None:
            text_embeds = self.gpt2.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
            prefix_mask = torch.ones(prefix.shape[0], prefix.shape[1], device=visual_feats.device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        return prefix


def generate_caption(model, video_feat, tokenizer):
    model.eval()
    with torch.no_grad():
        T = video_feat.shape[0]
        if T > FIXED_FRAMES:
            video_feat = video_feat[torch.linspace(0, T - 1, FIXED_FRAMES).long()]
        elif T < FIXED_FRAMES:
            video_feat = torch.cat([video_feat, torch.zeros(FIXED_FRAMES - T, 512)], dim=0)
        video_feat = video_feat.unsqueeze(0).to(DEVICE)
        prefix = model(video_feat)
        attn_mask = torch.ones((1, NUM_QUERIES), device=DEVICE)
        output_ids = model.gpt2.generate(inputs_embeds=prefix, attention_mask=attn_mask, max_new_tokens=40,
                                         num_beams=4, early_stopping=True, no_repeat_ngram_size=2,
                                         pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return text


# ================= 主程序 =================
if __name__ == "__main__":
    json_paths = [os.path.join(DATA_DIR, f"msvd_{s}.json") for s in ["train", "val", "test"]]
    dataset = MSVDDataset(json_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VideoCaptionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=len(dataloader) * EPOCHS)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    test_feat_path = os.path.join(FEAT_DIR, GEN_TEST_VIDEO.replace(".avi", ".pt"))
    test_feat = torch.load(test_feat_path) if os.path.exists(test_feat_path) else torch.zeros(FIXED_FRAMES, 512)

    step = 0
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for feats, ids, mask in pbar:
            feats, ids, mask = feats.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE)
            labels = torch.cat([torch.full((feats.shape[0], NUM_QUERIES), -100, device=DEVICE), ids], dim=1)
            labels[labels == tokenizer.pad_token_id] = -100
            loss = model(feats, ids, mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1
            if step % 500 == 0:
                print(f"\n[Step {step}] Test: {generate_caption(model, test_feat, tokenizer)}")
        torch.save(model.state_dict(), f"model_epoch{epoch + 1}.pth")