import random
import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from tqdm import tqdm
from kan.KANLayer import KANLayer
from torch.utils.data import DataLoader, TensorDataset
class SimpleKAN(torch.nn.Module):
    def __init__(self, layers : torch.nn.ModuleList, base_fun = torch.nn.SiLU(), seed = 1, device = 'cpu'):
        """Initialize
        Args:
            layers (torch.nn.ModuleList): Layers of KAN
            base_fun (_type_, optional):. Defaults to torch.nn.SiLU().
            seed (int, optional): random seed. Defaults to 1.
            device (str, optional): device Defaults to 'cpu'.
        """
        super(SimpleKAN, self).__init__()
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.layers = layers
        self.acts = None
    def to(self, device):
        super(SimpleKAN, self).to(device=device)
        self.device  = device
        for layer in self.layers : 
            layer.to(device)
        return self
    
    def update_grid(self,x):
        for l in range(len(self.layers)) :
            if isinstance(self.layers[l], KANLayer) : 
                self.get_act(x)
                #print(self.acts[l].shape)
                self.layers[l].update_grid_from_samples(self.acts[l])
        
       
    def get_act(self, x=None):
        '''
        collect intermidate activations
        '''
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            raise Exception("missing input data x")
        self.forward(x)
    
    def forward(self, x):
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts.append(x)  # acts shape: (batch, width[l])
        for layer in self.layers:
            if isinstance(layer, KANLayer) :
                x, preacts, postacts_numerical, postspline = layer(x)
            else :
                x = layer(x)
            self.acts.append(x)
        return x 
    
    
    def train_pykan(self, dataset, opt = None, steps = 50, update_grid = True, grid_update_num = 10, loss_fn = None, start_grid_update_step = -1, stop_grid_update_step = 50, batch = -1, log = 1):
        """This function i copied it from pykan, eliminated symbolic and multiplication nodes 
        Args:
            dataset (dictionary): dataset which has 4 keys : 'train_input', 'train_label', 'test_input', 'test_label'. Each is a torch tensor : input size : (no. of samples, dim) label_size : (number of samples)
            opt (torch.optim, optional): Optimization used, you can use "LBFGS" for using LBFGS.
            steps (int, optional): Number of step iterate through a random batch. Defaults to 50.
            update_grid (bool, optional):. Defaults to True.
            grid_update_num (int, optional): . Defaults to 10.
            loss_fn (torch.nn, optional): Loss function Defaults to None.
            start_grid_update_step (int, optional): _description_. Defaults to -1.
            stop_grid_update_step (int, optional): _description_. Defaults to 50.
            batch (int, optional): batch_size. Defaults to -1.
            log (int, optional): _description_. Defaults to 1.
        Returns:
            dictionary : train_loss and test_loss over steps
        """
        pbar = tqdm(range(steps), desc='description', ncols=100)
        if loss_fn == None:
            loss_fn = lambda x, y: torch.mean((x - y) ** 2)
        grid_update_freq = int(stop_grid_update_step / grid_update_num)
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch
        global train_loss
        if opt == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr = 1. ,history_size = 10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
        else : 
            optimizer = opt
        def closure():
            global train_loss
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id])
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            objective = train_loss
            objective.backward()
            return objective
    
        for _ in pbar : 
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)
            
            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])
            if opt == "LBFGS":
                optimizer.step(closure)
            else :
                pred = self.forward(dataset['train_input'][train_id])
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                loss = train_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            test_loss = loss_fn(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            if _ % log == 0 :
                pbar.set_description("| train_loss: %.2e | test_loss: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy()))
        
        
        return results
    
    
    def train(self, train_data: TensorDataset, test_data: TensorDataset, opt: str = None, 
            steps: int = 10, update_grid: bool = True, grid_update_num: int = 1, 
            loss_fn = None, batch: int = 64, log: int = 1, 
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
            dict: train_losses and test_losses
        """
        device = self.device
        test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
        
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
        for step in range(steps):
            pbar = tqdm(train_loader, desc=f'Step {step+1}/{steps}', ncols=100)

            for batch_idx, (batch_data, batch_labels) in enumerate(pbar):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                if update_grid and step == 0 and ((grid_update_num - batch_idx) > 0):
                    self.update_grid(batch_data)

                if opt == "LBFGS":
                    optimizer.step(lambda: closure(batch_data, batch_labels))
                else:
                    pred = self.forward(batch_data)
                    train_loss = loss_fn(pred, batch_labels)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                train_losses.append(train_loss.cpu().detach().numpy())
                pbar.set_description("| train_loss: %.2e |" % (train_loss.cpu().detach().numpy()))
            
            #Evaluate for every epochs
            
            with torch.no_grad():
                test_loss = 0.0
                num = 0.0
                for test_batch_data, test_batch_labels in test_loader :
                    test_batch_data = test_batch_data.to(device)  
                    test_batch_labels = test_batch_labels.to(device) 
                    test_loss += loss_fn(self.forward(test_batch_data), test_batch_labels).item()
                    num = num+1
                
                test_loss /= num
                print(f'Test loss in this epoch: {test_loss}')
                test_losses.append(test_loss)

            if step % log == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Step {step+1}/{steps} completed. Train loss: {train_loss:.2e}, Test loss: {test_loss:.2e}, Learning rate: {lr:.2e}')
            
            if scheduler is not None:
                scheduler.step()

        torch.cuda.empty_cache()
        return {
            'train_loss': train_losses,
            'test_loss': test_losses
        }