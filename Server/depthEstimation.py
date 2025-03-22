from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


class DepthEstimation:
    def __init__(self, model_type='indoor', device='cuda'):
        model_name = "zoedepth"
        if model_type == 'indoor':
            model_type_full = "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
        else:
            model_type_full = "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"
        # Load default pretrained resource defined in config if not set
        overwrite = {"pretrained_resource": model_type_full}
        config = get_config(model_name, "eval", dataset=None, **overwrite)
        self.model = build_model(config)
        self.model = self.model.to(device)

    def generate(self, image):
        depth_numpy = self.model.infer_pil(image)  # as numpy
        return depth_numpy
