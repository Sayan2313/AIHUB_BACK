import torch
import torch.nn as nn
import numpy as np
from functools import lru_cache
from sympy import symbols, sympify, lambdify
## Model
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size: list[int],activation):
        super().__init__()
        l = [nn.Linear(input_size, hidden_size[0]), activation, ]
        for i in range(1, len(hidden_size)):
            l.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))
            l.append(activation)
        l.append(nn.Linear(hidden_size[-1], output_size))
        self.net = nn.ModuleList(l)

    def forward(self, x):
        for i, l in enumerate(self.net):
            x = l(x)
        return x
###---------------------------###
# ===== Safe eval (basic) =====
def parse_function(expr_str):
    x = symbols('x')

    expr = sympify(expr_str)   # safe parsing
    func = lambdify(x, expr, "numpy")  # fast evaluation

    return func
# ===== Activation selector =====
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "linear":
        return nn.Identity()
    else:
        return nn.ReLU()
def train_and_predict(data:dict):
    torch.manual_seed(data['seed'])
    x = np.linspace(data['xMin'], data['xMax'], data['points'])
    y = parse_function(data['expression'])(x)

    # Convert to torch
    x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    activation = get_activation(data['activation'])
    model = Net(1, 1, data['neuronsPerLayer'], activation)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=data['learningRate'])
    total_loss = 0
    for epoch in range(data['epochs']):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
    return x_train.numpy().flatten(),y_pred.numpy().flatten(),((total_loss**0.5) / data['epochs'])



