import os
import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg19
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    device = torch.device("mps")
print("Running on device:", device)

class LossModule(nn.Module):                                                                                                                                                            
    def __init__(self):
        super(LossModule, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def compute_content_loss(self, generated, target):
        return self.mse_loss(generated, target)
    
    def compute_style_loss(self, feat, target_feat):
        _, channels, h, w = feat.size()
        feat = feat.view(channels, h * w)
        _, channels, h, w = target_feat.size()
        target_feat = target_feat.view(channels, h * w)
        gram_feat = torch.mm(feat, feat.t())
        gram_target = torch.mm(target_feat, target_feat.t())
        return self.mse_loss(gram_feat, gram_target)
    
    def forward(self, gen_feats, cont_feats, style_feats, lam_content=0.5, lam_style=0.5):
        c_loss = self.compute_content_loss(gen_feats[3], cont_feats[3])
        s_loss = 0
        for i in range(len(gen_feats)):
            s_loss += self.compute_style_loss(gen_feats[i], style_feats[i])
        return lam_content * c_loss + lam_style * s_loss

class StyleTransferModel:
    def __init__(self, content_img, style_img, optimizer_type='lbfgs',
                 lr=0.1, lam_content=0.5, lam_style=0.5, iterations=1000, device='cpu'):
        self.content_img = content_img.to(device)
        self.style_img = style_img.to(device)
        self.generated_img = self.content_img.clone().detach().to(device).requires_grad_(True)
        self.feature_net = vgg19(weights='DEFAULT').features.to(device)
        self.loss_mod = LossModule().to(device)
        self.lam_content = lam_content
        self.lam_style = lam_style
        self.iterations = iterations
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam([self.generated_img], lr=lr)
        else:
            self.optimizer = torch.optim.LBFGS([self.generated_img], lr=lr)
        self.opt_type = optimizer_type.lower()

    def extract_features(self, img):
        feats = []
        for name, layer in self.feature_net._modules.items():
            img = layer(img)
            if name in {'0', '2', '5', '7', '10'}:
                feats.append(img)
            if name == '11':
                break
        return feats

    def closure_fn(self):
        self.optimizer.zero_grad()
        cont_feats = self.extract_features(self.content_img)
        style_feats = self.extract_features(self.style_img)
        gen_feats = self.extract_features(self.generated_img)
        loss = self.loss_mod(gen_feats, cont_feats, style_feats, lam_content=self.lam_content, lam_style=self.lam_style)
        loss.backward()
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = param.grad.contiguous()
        return loss

    def run_transfer(self):
        for it in range(self.iterations):
            if self.opt_type == 'adam':
                loss_val = self.closure_fn()
                self.optimizer.step()
            else:   
                loss_val = self.optimizer.step(self.closure_fn)
            self.generated_img.data.clamp_(0, 1)
            if it % 5 == 0:
                print(f"Iteration {it}: Loss = {loss_val.item()}")
        return self.generated_img
    
def fetch_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)


content_directory = 'content'
style_directory = 'styles'
content_files = sorted(os.listdir(content_directory))
style_files = sorted(os.listdir(style_directory))
viz_images = []    
viz_titles = []   

for cont_file in content_files:
    cont_path = os.path.join(content_directory, cont_file)
    cont_tensor = fetch_image(cont_path)
    for sty_file in style_files:
        sty_path  = os.path.join(style_directory, sty_file)
        sty_tensor  = fetch_image(sty_path)
        print(f"\nProcessing pair:\n  Content: {cont_file}\n  Style:   {sty_file}")
        model = StyleTransferModel(cont_tensor, sty_tensor, optimizer_type='lbfgs', lr=0.1, lam_content=1e-6, lam_style=1e-8, iterations=11, device=device)
        output_tensor = model.run_transfer()
        cont_np = cont_tensor.squeeze().cpu().numpy()
        cont_np = np.transpose(cont_np, (1, 2, 0))
        sty_np = sty_tensor.squeeze().cpu().numpy()
        sty_np = np.transpose(sty_np, (1, 2, 0))
        out_np = output_tensor.squeeze().detach().cpu().numpy()
        out_np = np.clip(out_np, 0, 1)
        out_np = (out_np * 255).astype(np.uint8)
        out_np = np.transpose(out_np, (1, 2, 0))
        viz_images.extend([cont_np, sty_np, out_np])
        viz_titles.extend([f"Content: {cont_file}", f"Style: {sty_file}", "Generated"])




import os
import matplotlib.pyplot as plt
save_dir = "long"
os.makedirs(save_dir, exist_ok=True)

num_cols = 3   
num_pairs = len(content_files) * len(style_files)

for pair_idx in range(num_pairs):
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))   

    image_filenames = []  

    for col in range(num_cols):
        idx = pair_idx * num_cols + col
        if idx < len(viz_images):
            axes[col].imshow(viz_images[idx])
            axes[col].set_title(viz_titles[idx])
            axes[col].axis('off')

    plt.tight_layout()
    content_name = os.path.splitext(os.path.basename(content_files[pair_idx // len(style_files)]))[0]
    style_name = os.path.splitext(os.path.basename(style_files[pair_idx % len(style_files)]))[0]
    save_filename = f"{style_name}{content_name}.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)   

