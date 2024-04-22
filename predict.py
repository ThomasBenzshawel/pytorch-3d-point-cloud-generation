import torch
import numpy as np
import transform
import scipy.io

def predict_point_cloud(generator, latent_dim, device):
    # Sample random noise
    z = torch.randn(1, latent_dim).to(device)
    # Generate point cloud
    with torch.no_grad():
        point_cloud = generator(z)
    return point_cloud

class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.result_path = f"results/{cfg.model}_{cfg.experiment}"

    def predict(self, model, image_in):
        fuseTrans = self.cfg.fuseTrans
        input_image = torch.from_numpy(image_in)\
                            .permute((0,3,1,2))\
                            .float().to(self.cfg.device)
        points24 = np.zeros([self.cfg.inputViewN, 1], dtype=object)

        XYZ, maskLogit = model(input_image)
        mask = (maskLogit > 0).float()
        # ------ build transformer ------
        XYZid, ML = transform.fuse3D(
            self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
        
        XYZid, ML = XYZid.permute([0, 2, 1]), ML.squeeze()
        for a in range(self.cfg.inputViewN):
            xyz = XYZid[a] #[VHW, 3]
            ml = ML[a] #[VHW]
            points24[a, 0] = (xyz[ml > 0]).detach().cpu().numpy()
            scipy.io.savemat(
            f"{self.result_path}/{"guessed"}.mat", 
            {"pointcloud": points24})