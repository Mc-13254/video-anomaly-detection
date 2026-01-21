# utils/lime_explainer.py
import numpy as np
import torch
from lime import lime_image
import skimage.segmentation

def explain_with_lime(model, image, device):
    model.eval()
    
    def batch_predict(images):
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        images = images.to(device)
        with torch.no_grad():
            recon = model(images)
            errors = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
        errors = errors.cpu().numpy()
        # Return [P(normal), P(anomaly)]
        return np.stack([1.0 - np.clip(errors, 0, 1), errors], axis=1)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype(np.double),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=50,
        segmentation_fn=lambda x: skimage.segmentation.slic(x, n_segments=100, compactness=10)
    )
    
    # Get the TOP predicted label (most confident class)
    top_label = explanation.top_labels[0]
    
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    return skimage.segmentation.mark_boundaries(temp, mask)