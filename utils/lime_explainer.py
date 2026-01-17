# utils/lime_explainer.py
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

def explain_with_lime(model, image, device):
    """
    image: numpy array (224, 224, 3) in [0,1]
    """
    model.eval()
    
    def batch_predict(images):
        # images: (N, 224, 224, 3) in [0,1]
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        images = images.to(device)
        with torch.no_grad():
            recon = model(images)
            errors = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
        return errors.cpu().numpy()  # higher = more anomalous

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype(np.double),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=100,
        segmentation_fn=lambda x: skimage.segmentation.slic(x, n_segments=100, compactness=10)
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    return mark_boundaries(temp, mask)