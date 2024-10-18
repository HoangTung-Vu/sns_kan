import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset

dtype = torch.get_default_dtype()


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(self, layers : torch.nn.ModuleList, device = 'cpu'):
        super(KAN, self).__init__()
        self.layers = layers
        self.device = device

    def to(self, device):
        super(KAN, self).to(device=device)
        self.device  = device
        for layer in self.layers : 
            layer.to(device)
        return self

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    # def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    #     return sum(
    #         layer.regularization_loss(regularize_activation, regularize_entropy)
    #         for layer in self.layers
    #     )
    
    def fit(self, 
            train_data: TensorDataset, 
            test_data: TensorDataset, 
            opt: str = None, 
            steps: int = 10, 
            loss_fn = None, 
            batch: int = 64, 
            log: int = 1, 
            scheduler= None) -> dict:
        """Training function

        Args:
            train_data (TensorDataset): training dataset
            test_data (TensorDataset): testing dataset, model will evalute after training an epoch
            opt (str, optional): Optimization function, recommend use torch.optim. Defaults to None.
            steps (int, optional): Number of epochs. Defaults to 10.
            update_grid (bool, optional): update_grid for Spl-KANLayer. Defaults to True.
            grid_update_num (int, optional): _description_. Defaults to 1.
            loss_fn (_type_, optional): loss function recommend torch.nn. Defaults to None.
            batch (int, optional): batch_size. Defaults to 64.
            log (int, optional): . Defaults to 1.
            scheduler (_type_, optional): learning rate scheduler. Defaults to None.

        Returns:
            dict: train_losses, test_losses, train_acc, test_acc
        """
        device = self.device
        test_size = len(test_data)
        batch_test = batch
        while test_size % batch_test != 0:
            batch_test -= 1
            
        test_loader = DataLoader(test_data, batch_size=batch_test, shuffle=False)
        
        if loss_fn is None:
            loss_fn = lambda x, y: torch.mean((x - y) ** 2)

        train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)


        if opt == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1.0, history_size=10, 
                                        line_search_fn="strong_wolfe", tolerance_grad=1e-32, 
                                        tolerance_change=1e-32)
        else: 
            optimizer = opt

        def closure():
            global train_loss
            optimizer.zero_grad()
            pred = self.forward(batch_data)
            train_loss = loss_fn(pred, batch_labels)
            objective = train_loss
            objective.backward()
            return objective
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for step in range(steps):
            pbar = tqdm(train_loader, desc=f'Step {step+1}/{steps}', ncols=100)
            total_correct = 0
            total_samples = 0
            current_lr = optimizer.param_groups[0]['lr']
            for batch_idx, (batch_data, batch_labels) in enumerate(pbar):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                if opt == "LBFGS":
                    optimizer.step(lambda: closure(batch_data, batch_labels))
                else:
                    pred = self.forward(batch_data)
                    total_correct += (torch.argmax(pred.type(dtype), dim=1) == batch_labels.type(dtype)).sum().item()
                    total_samples += batch_labels.size(0)
                    train_loss = loss_fn(pred, batch_labels)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                train_losses.append(train_loss.cpu().detach().numpy())
                pbar.set_description("| train_loss: %.2e |" % (train_loss.cpu().detach().numpy()))
                        #Evaluate for every epochs
            
            if scheduler is not None:
                scheduler.step()
            train_acc_ep = total_correct / total_samples if total_samples > 0 else 0.0
            print(f"Train accuracy in this epoch : {train_acc_ep}")
            train_acc.append(train_acc_ep)
            
            with torch.no_grad():
                test_loss = 0.0
                total_correct = 0
                total_samples = 0
                test_acc_ep = 0.0

                for test_batch_data, test_batch_labels in test_loader:
                    test_batch_data = test_batch_data.to(device)
                    test_batch_labels = test_batch_labels.to(device)

                    loss = loss_fn(self.forward(test_batch_data), test_batch_labels)
                    test_loss += loss.item()

                    predictions = self.forward(test_batch_data)
                    predicted_labels = torch.argmax(predictions, dim=1)

                    total_correct += (predicted_labels == test_batch_labels).sum().item()
                    total_samples += test_batch_labels.size(0)

                test_loss /= len(test_loader)
                test_acc_ep = total_correct / total_samples

                print(f'Test loss in this epoch: {test_loss}')
                print(f'Test accuracy in this epoch : {test_acc_ep}')

                test_losses.append(test_loss)
                test_acc.append(test_acc_ep)

            if step % log == 0:
                print(f'Step {step+1}/{steps} completed. Train loss: {train_loss:.2e}, Test loss: {test_loss:.2e}, Learning rate: {current_lr:.2e}')
            

        torch.cuda.empty_cache()
        return {
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_acc': train_acc,
            'test_acc' : test_acc
        }

    def save(self, path = 'model'):
        try:
            # Save the model's state_dict along with layers, device, and seed
            torch.save({
                'model_state_dict': self.state_dict(),
                'layers': self.layers,  # Save the architecture (layers)
                'device': self.device,
                'seed': torch.initial_seed()
            }, path)
            print(f"Model saved successfully at {path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    @staticmethod
    def load(path):
        try:
            checkpoint = torch.load(path)
            model = KAN(layers=checkpoint['layers'], device=checkpoint['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            torch.manual_seed(checkpoint['seed'])
            model.to(checkpoint['device'])
            print(f"Model loaded successfully from {path}")
            return model
        
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
