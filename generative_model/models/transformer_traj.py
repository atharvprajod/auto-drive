import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=200):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                  nhead=nhead,
                                                  dim_feedforward=1024,
                                                  dropout=dropout,
                                                  batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)
        
    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, input_dim]
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)
        
        # Apply transformer encoder
        memory = self.transformer_encoder(embedded, src_mask)
        
        # Decode trajectories
        output = self.decoder(memory)
        return output

class TrajectoryPredictor:
    def __init__(self, model_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrajectoryTransformer().to(self.device)
        
    def predict(self, current_state, scene_context):
        """
        Predict future trajectory given current state and scene context
        
        Args:
            current_state: Tensor of shape [batch_size, state_dim]
            scene_context: Tensor of shape [batch_size, context_length, feature_dim]
            
        Returns:
            predicted_trajectory: Tensor of shape [batch_size, prediction_horizon, state_dim]
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare input sequence
            input_seq = self._prepare_input(current_state, scene_context)
            
            # Generate trajectory
            predicted_trajectory = self.model(input_seq)
            
            return predicted_trajectory
    
    def _prepare_input(self, current_state, scene_context):
        # Combine current state with scene context
        return torch.cat([current_state.unsqueeze(1), scene_context], dim=1) 