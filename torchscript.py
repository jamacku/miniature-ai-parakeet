#!/usr/bin/env python3
# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

import torch
class MyModule(torch.nn.Module):
  def __init__(self, N, M, state):
    super(MyModule, self).__init__()
    self.weight = torch.nn.Parameter(torch.rand(N, M))
    self.state = state

  def forward(self, input):
    self.state.append(input)
    if input.sum() > 0:
      output = self.weight.mv(input)
    else:
      output = self.weight + input
    return output

# Compile the model code to a static representation
my_module = MyModule(3,4, [torch.rand(3, 4)])
my_script_module = torch.jit.script(my_module)

# Save the compiled code and model data
# so it can be loaded elsewhere
my_script_module.save("my_script_module.pt")

# Load the compiled code and model data
# so it can be used in another script
loaded_module = torch.jit.load("my_script_module.pt")
print(loaded_module.state)
