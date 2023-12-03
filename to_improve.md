## So that other people can USE ConSpec
- Make a ConSpec class that follows the template in template.py

## So that other people can make MODIFICATIONS to ConSpec
- Consider changing the storageConSpec to a [PyTorch replay buffer](https://pytorch.org/rl/reference/generated/torchrl.data.ReplayBuffer.html)
    - Not sure it is feasible since I'm not sure exactly what it does, if you can't change it to a PyTorch replay buffer then it needs a lot of name changing and documentation
- Follow the naming conventions [here](https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html)
- Add a docstring for each function (probably Google style -- good practice for your future employment :)), see [here](https://www.geeksforgeeks.org/python-docstrings/)
- Add the full setup procedure FROM SCRATCH into README.md (including python version)
- Change code so it is runnable with a command like the following: python train.py --env Multikeytodoor --num_prototypes 16 ... etc.