import torch
import torch.nn as nn
import math
from sympy import symbols, sympify, Poly
import torch.nn.functional as F
from torch.nn import Parameter, init

class TN_layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        super(TN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = symbolic_expression
        self.terms = self._parse_expression(symbolic_expression)
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) for _ in self.terms])
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _parse_expression(self, expression):
        x = symbols('x')
        # Replace @ with * for sympy compatibility
        expression = expression.replace('@', '*')
        expr = sympify(expression)
        poly_expr = Poly(expr, x)
        
        # Get the coefficients and corresponding exponents
        terms = []
        for monom, coeff in poly_expr.terms():
            exponent = monom[0]  # The exponent of x
            coefficient = float(coeff)  # Convert sympy number to float
            terms.append({'coefficient': coefficient, 'exponent': exponent})
        
        return terms

    def forward(self, x):
        result = 0
        # For 3D input (batch_size, seq_len, features)
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.reshape(-1, self.in_features)
            
            for i, term in enumerate(self.terms):
                exponent = term['exponent']
                coefficient = term['coefficient']
                
                if exponent == 0:
                    term_result = coefficient * torch.ones(x_reshaped.size(0), self.out_features, device=x.device)
                else:
                    x_powered = torch.pow(x_reshaped.clamp(-100, 100), exponent)
                    term_result = coefficient * nn.functional.linear(x_powered, self.weights[i])
                
                result = result + term_result
            
            result = result.reshape(batch_size, seq_len, self.out_features)
            
        else:
            for i, term in enumerate(self.terms):
                exponent = term['exponent']
                coefficient = term['coefficient']
                
                if exponent == 0:
                    term_result = coefficient * torch.ones(x.size(0), self.out_features, device=x.device)
                else:
                    x_powered = torch.pow(x.clamp(-100, 100), exponent)
                    term_result = coefficient * nn.functional.linear(x_powered, self.weights[i])               
                result = result + term_result

        if self.bias is not None:
            result = result + self.bias
            
        return result




class TNConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = True):
        super(TNConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_x2 = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))  
        self.weight_x = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))   
        self.weight_x3 = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_x3, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_x2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_x, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_x)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv_x3 = F.conv2d(torch.pow(x, 3), self.weight_x2, None, self.stride, self.padding)
        conv_x2 = F.conv2d(torch.pow(x, 2), self.weight_x2, None, self.stride, self.padding)
        conv_x = F.conv2d(x, self.weight_x, None, self.stride, self.padding)
        
        output =  conv_x + conv_x3
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)  
        return output
    

# Example usage
if __name__ == '__main__':
    expr = '0.5@x**2 - 3@x + 2'
    layer = TN_layer(5, 10, expr)
    # Test with different input dimensions

    x_2d = torch.randn(8, 5)  
    output_2d = layer(x_2d)
    print(f"2D input shape: {x_2d.shape}, output shape: {output_2d.shape}")
    
    x_3d = torch.randn(8, 4, 5) 
    output_3d = layer(x_3d)
    print(f"3D input shape: {x_3d.shape}, output shape: {output_3d.shape}")