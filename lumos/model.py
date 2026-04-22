import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Learns a transformation matrix (input transform / feature transform)
"""
class TNet(nn.Module):
    def __init__(self, K=3):
        """
        K: Number of input channels for each point / feature, typically 3 for XYZ coordinates
        """
        self.K = K
        super(TNet, self).__init__()

        # Point-wise shared MLP implemented as a conv1d
        self.conv1 = nn.Conv1d(in_channels=K, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(num_features=1024)

        # Fully connected layers 
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=K*K)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if m is not self.fc3:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.constant_(self.fc3.weight, 0.0)
        eye = torch.eye(self.K).flatten()
        with torch.no_grad():
            self.fc3.bias.copy_(eye)

    # Forward pass
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        input: point_cloud, tensor of shape (B, N, K)
        return: Output transformation matrix of shape (B, K, K)
        """
        x = point_cloud.transpose(1, 2)     # (B, K, N)

        # Point-wise shared MLP (K, 64, 128, 1024), map each single point to a higher dimensional space.
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 1024, N)

        # Max pooling aggregation
        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = x.view(-1, 1024)                # (B, 1024)

        # Global MLP (1024, 512, 256, K*K), generate the transformation matrix
        x = F.relu(self.bn4(self.fc1(x)))   # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))   # (B, 256)
        x = self.fc3(x)                     # (B, K*K)
        return x.view(-1, self.K, self.K)   # (B, K, K)


"""
Shape encoder, from point cloud to latent space
"""
class ShapeEncoder(nn.Module):
    def __init__(self, tnet1: bool = True, tnet2: bool = True, latent_dim: int = 256):
        super(ShapeEncoder, self).__init__()
        
        # Input transform net (probably optional)
        self.tnet1 = TNet(K=3) if tnet1 else None

        # Point-wise shared MLP (3, 64, 64) implemented as a conv1d
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        
        # Feature transform net
        self.tnet2 = TNet(K=64) if tnet2 else None  

        # Point-wise shared MLP (64, 64, 128, 1024) implemented as conv1d
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm1d(num_features=1024)

        # Point-wise shared MLP on concatenated features
        self.conv6 = nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm1d(num_features=512)
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=latent_dim, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # Forward pass
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        input: point_cloud, tensor of shape (B, N, K)
        return: latent representation, tensor of shape (B, 128)
        """

        B, N, K = point_cloud.shape

        # Input transform, align points in a canonical coordinate
        if self.tnet1 is not None:
            T1 = self.tnet1(point_cloud)                        # (B, 3, 3)
            x = torch.bmm(point_cloud, T1).transpose(1,2)       # (B, 3, N)
        else:
            x = point_cloud.transpose(1, 2)                     # (B, 3, N)

        # Point-wise shared MLP on aligned point (3, 64, 64), map each point to a high dimensional space
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 64, N)

        # Feature transform, align features in a canonical space
        if self.tnet2 is not None:
            x_transpose = x.transpose(1, 2)                 # (B, N, 64)
            T2 = self.tnet2(x_transpose)                    # (B, 64, 64)
            x = torch.bmm(x_transpose, T2).transpose(1, 2)  # (B, 64, N)

        point_feature = x

        # Global feature extraction
        x = F.relu(self.bn3(self.conv3(x))) # (B, 64, N)
        x = F.relu(self.bn4(self.conv4(x))) # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x))) # (B, 1024, N)
        global_feature = F.max_pool1d(x, kernel_size=x.size(2))  # (B, 1024, 1)
        global_feature = global_feature.repeat(1, 1, N)           # (B, 1024, N)

        concatenated_feature = torch.cat([point_feature, global_feature], dim=1)  # (B, 1088, N)

        x = F.relu(self.bn6(self.conv6(concatenated_feature)))  # (B, 512, N)
        x = F.relu(self.bn7(self.conv7(x)))                    # (B, 256, N)
        x = F.relu(self.conv8(x))                                 # (B, latent_dim, N)
        latent = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)  # (B, latent_dim)

        return latent


"""
Shape decoder, from latent space to point cloud
Number of points: 2048
"""
class ShapeDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, num_points: int = 1024):
        super(ShapeDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = self._floor_points(num_points)
        self.residual_points = num_points - self.num_points

        # Fully connected layer
        if self.residual_points > 0:
            self.fc1 = nn.Linear(in_features=latent_dim, out_features=256)
            self.bn_fc1 = nn.BatchNorm1d(num_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=512)
            self.bn_fc2 = nn.BatchNorm1d(num_features=512)
            self.fc3 = nn.Linear(in_features=512, out_features=1024)
            self.bn_fc3 = nn.BatchNorm1d(num_features=1024)
            self.fc4 = nn.Linear(in_features=1024, out_features=2048)
            self.bn_fc4 = nn.BatchNorm1d(num_features=2048)
            self.fc5 = nn.Linear(in_features=2048, out_features=self.residual_points * 3)


        # Upconv layers (Common)
        self.up1 = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.up4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=3, padding=0)
        self.bn_up = nn.BatchNorm2d(num_features=128)

        # Upconv layers (exact number of points)
        # 1024 = 32 * 32
        self.head_1024_up = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0)

        # 2048 = 32 * 64
        self.head_2048_up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1,4), stride=(1,2), padding=(0,1))
        self.head_2048_bn1 = nn.BatchNorm2d(num_features=64)
        self.head_2048_up2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

        # 4096 = 64 * 64
        self.head_4096_up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.head_4096_bn1 = nn.BatchNorm2d(num_features=64)
        self.head_4096_up2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

        # 8192 = 64 * 128
        self.head_8192_up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.head_8192_bn1 = nn.BatchNorm2d(num_features=64)
        self.head_8192_up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1,4), stride=(1,2), padding=(0,1))
        self.head_8192_bn2 = nn.BatchNorm2d(num_features=32)
        self.head_8192_up3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights
        self._init_weights()
    
    @staticmethod
    def _floor_points(n):
        supported = (1024, 2048, 4096, 8192)
        opts = [m for m in supported if m <= n]
        return max(opts) if len(opts) > 0 else 0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Forward pass
    def _common_upconv(self, latent: torch.Tensor):
        u = latent.view(latent.size(0), latent.size(1), 1, 1) # (B, latent_dim, 1, 1)
        u = F.relu(self.up1(u))       # (B, 512, 2, 2)
        u = F.relu(self.up2(u))       # (B, 256, 4, 4)
        u = F.relu(self.up3(u))       # (B, 256, 10, 10)
        u = F.relu(self.bn_up(self.up4(u)))       # (B, 128, 32, 32)
        return u
    
    def _upconv_head(self, u: torch.Tensor):
        B = u.size(0)
        if self.num_points == 0:
            return u.new_zeros(B, 0, 3)

        if self.num_points == 1024:
            y = self.head_1024_up(u)                       # (B, 3, 32, 32)
        elif self.num_points == 2048:
            y = F.relu(self.head_2048_bn1(self.head_2048_up1(u)))  # (B, 64, 32, 64)
            y = self.head_2048_up2(y)                      # (B, 3, 32, 64)
        elif self.num_points == 4096:
            y = F.relu(self.head_4096_bn1(self.head_4096_up1(u)))  # (B, 64, 64, 64)
            y = self.head_4096_up2(y)                      # (B, 3, 64, 64)
        elif self.num_points == 8192:
            y = F.relu(self.head_8192_bn1(self.head_8192_up1(u)))  # (B, 64, 64, 64)
            y = F.relu(self.head_8192_bn2(self.head_8192_up2(y))) # (B, 32, 64, 128)
            y = self.head_8192_up3(y)                      # (B, 3, 64, 128)
        else:
            raise ValueError(f"Unsupported number of points: {self.num_points}")
        
        pts = y.view(B, 3, -1).transpose(1, 2).contiguous()   # (B, num_points, 3)
        return pts
    
    def _fc_points(self, latent: torch.Tensor):
        B = latent.size(0)
        if self.residual_points == 0:
            return latent.new_zeros(B, 0, 3)

        x = F.relu(self.bn_fc1(self.fc1(latent)))   # (B, 256)
        x = F.relu(self.bn_fc2(self.fc2(x)))        # (B, 512)
        x = F.relu(self.bn_fc3(self.fc3(x)))        # (B, 1024)
        x = F.relu(self.bn_fc4(self.fc4(x)))        # (B, 2048)
        x = self.fc5(x)                             # (B, residual_points*3)
        pts = x.view(B, self.residual_points, 3)    # (B, residual_points, 3)
        return pts
    

    def forward(self, latent: torch.Tensor, return_branches: bool = False) -> torch.Tensor:
        """
        input: latent, tensor of shape (B, latent_dim)
        return: Reconstructed point cloud, tensor of shape (B, 2048, 3)
        """
        B = latent.size(0)

        u = self._common_upconv(latent)              # (B, 128, H, W)
        points_upconv = self._upconv_head(u)         # (B, num_points, 3)
        points_fc = self._fc_points(latent)          # (B, residual_points, 3)
        points = torch.cat([points_upconv, points_fc], dim=1) if self.residual_points > 0 else points_upconv  # (B, total_points, 3)
        assert points.size(1) == self.num_points + self.residual_points, \
            f"Output points size mismatch: expected {self.num_points + self.residual_points}, got {points.size(1)}"
        
        if return_branches:
            return points, {'upconv': points_upconv, 'fc': points_fc}
        return points
    
"""
Point cloud autoencoder
"""
class PCAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_points: int = 1024, tnet1: bool = True, tnet2: bool = True):
        super(PCAutoencoder, self).__init__()

        # Encoder
        self.encoder = ShapeEncoder(tnet1=tnet1, tnet2=tnet2, latent_dim=latent_dim)

        # Decoder
        self.decoder = ShapeDecoder(latent_dim=latent_dim, num_points=num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

"""
PD readings to shape neural network.
Input:
    PD readings (B, 150)
Output:
    Shape feature (B, latent_dim)
"""
class PD2Latent(nn.Module):
    def __init__(self, in_features: int = 150, out_features: int = 128):
        super(PD2Latent, self).__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=out_features)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class ShapeNet(nn.Module):
    def __init__(self, pd_in_features: int = 150, latent_dim: int = 128, num_points: int = 1024, tnet1: bool = True, tnet2: bool = True):
        super(ShapeNet, self).__init__()
        self.pd2latent = PD2Latent(in_features=pd_in_features, out_features=latent_dim)
        self.decoder = ShapeDecoder(latent_dim=latent_dim, num_points=num_points)
    
    def forward(self, pd_readings: torch.Tensor) -> torch.Tensor:
        latent = self.pd2latent(pd_readings)
        reconstructed = self.decoder(latent)
        return reconstructed