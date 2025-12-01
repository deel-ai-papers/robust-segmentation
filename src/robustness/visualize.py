import torch
from scipy.ndimage import label


def find_large_connected_components(classes: torch.Tensor, N_min: int) -> list:
    """
    Finds connected components of pixels with the same class prediction that are
    larger than a given threshold.

    Args:
        classes (torch.Tensor): A 2D tensor of shape (H, W) containing integer
                                class predictions.
        N_min (int): The minimum number of pixels for a connected component to be
                     included in the result.

    Returns:
        list: A list of lists, where each inner list contains the set of
              pixel coordinates (tuples of (row, col)) for a connected
              component larger than N_min.
    """
    if not isinstance(classes, torch.Tensor) or classes.dim() != 2:
        raise TypeError("Input 'classes' must be a 2D torch tensor.")
    if not isinstance(N_min, int) or N_min <= 0:
        raise ValueError("Input 'N_min' must be a positive integer.")

    output_components = []
    unique_classes = torch.unique(classes)

    for c in unique_classes:
        # Create a binary mask for the current class
        binary_mask = (classes == c).numpy()

        # Perform connected component labeling
        labeled_array, num_features = label(binary_mask)

        if num_features > 0:
            # Iterate through each found component
            for i in range(1, num_features + 1):
                component = labeled_array == i
                component_size = component.sum()

                if component_size > N_min:
                    # Get the coordinates of the pixels in the component
                    rows, cols = component.nonzero()

                    # Store the coordinates as a list of tuples
                    component_coords = list(zip(rows.tolist(), cols.tolist()))
                    output_components.append(component_coords)

    return output_components
