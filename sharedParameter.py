import numpy as np

def generate_shared_parameters(width, depth, seed=55):
    """
    Generate shared parameters for a neural network model.
    width: int, number of neurons in each layer
    depth: int, number of layers in the model
    seed: int, random seed for reproducibility

    Returns:
    weight_list: list of numpy arrays, containing the weights for each layer
    bias_list: list of numpy arrays, containing the biases for each layer
    """

    np.random.seed(seed)
    weight_list = [np.random.randn(width, width) * np.sqrt(2.0 / width) for _ in range(depth)]
    bias_list = [np.random.randn(width) for _ in range(depth)]

    # print("Generated shared Weights:")
    #for i, w in enumerate(weight_list):
    #    print(f"Layer {i} weight matrix:\n{w}")
    
    #print("Generated shared Biases:")
    #for i, b in enumerate(bias_list):
    #    print(f"Layer {i} bias vector:\n{b}")

    return weight_list, bias_list

# Test the function
# generat_shared_parameters(1, 4, seed=42)