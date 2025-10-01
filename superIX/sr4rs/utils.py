import tensorflow as tf
import torch
import numpy as np

def load_cesbio_sr(weight_path) -> tf.function:
    """Prepare the CESBIO model
    Returns:
        tf.function: A tf.function to get the SR image
    """

    # read the model
    model = tf.saved_model.load(weight_path)

    # get the signature
    signature = list(model.signatures.keys())[0]

    # get the function
    func = model.signatures[signature]

    return func


def run_sr4rs(
        model: tf.function,
        lr: tf.Tensor
) -> np.ndarray:
    """Run the SR4RS model
    Args:
        model (tf.function): The model to use
        lr (tf.Tensor): The low resolution image
        hr (tf.Tensor): The high resolution image
        cropsize (int, optional): The cropsize. Defaults to 32.
        overlap (int, optional): The overlap. Defaults to 0.
    Returns:
        dict: The results
    """

    # Run inference
    # selection of four input bands from s2: red, green, blue, nir
    # permute to: red, blue, nir, green
    img = lr[[3, 2, 1, 7]]
    lr_padded = np.zeros((4, 144, 144), dtype=img.dtype)
    lr_padded[:, 8:136, 8:136] = img
    Xnp = torch.from_numpy(lr_padded[None]).permute(0, 2, 3, 1)

    #Xnp = torch.from_numpy(lr[[3, 2, 1, 7]][None]).permute(0, 2, 3, 1)
    Xtf = tf.convert_to_tensor(Xnp, dtype=tf.float32)
    pred = model(Xtf)

    # Save the results
    pred_np = pred['output_32:0'].numpy()

    # re-permute to: red, green, blue, nir
    pred_torch = torch.from_numpy(pred_np).permute(0, 3, 1, 2)
    pred_arr = pred_torch.squeeze().numpy() / 10_000

    return pred_arr
