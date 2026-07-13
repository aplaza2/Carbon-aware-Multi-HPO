import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel


PRETRAINED_MODELS = [
    {
        "model_name": "dinov2_base",
        "family": "dinov2",
        "hf_id": "facebook/dinov2-base",
        "size": 224,
    },
    {
        "model_name": "clip_vit_b32",
        "family": "clip",
        "hf_id": "openai/clip-vit-base-patch32",
        "size": 224,
    },
]


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def get_transformer_blocks(backbone, family):
    """
    Devuelve los bloques transformer del backbone para poder descongelar
    solo los últimos n bloques.

    DINOv2 en transformers suele tener:
        backbone.encoder.layer

    CLIPVisionModel suele tener:
        backbone.vision_model.encoder.layers
    """

    if family == "dinov2":
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            return list(backbone.encoder.layer)

        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
            return list(backbone.encoder.layers)

    elif family == "clip":
        if hasattr(backbone, "vision_model"):
            vision_model = backbone.vision_model

            if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
                return list(vision_model.encoder.layers)

    raise ValueError(f"No pude encontrar los bloques transformer para family={family}")


class PretrainedVisionClassifier(nn.Module):
    def __init__(
        self,
        family,
        hf_id,
        num_classes,
        dropout,
        finetune_strategy,
        unfreeze_last_n,
    ):
        super().__init__()

        self.family = family
        self.hf_id = hf_id

        if family == "dinov2":
            self.backbone = AutoModel.from_pretrained(hf_id)

        elif family == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(hf_id)

        else:
            raise ValueError(f"Modelo no soportado: {family}")

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.apply_finetuning_strategy(
            finetune_strategy=finetune_strategy,
            unfreeze_last_n=unfreeze_last_n,
        )

    def apply_finetuning_strategy(self, finetune_strategy, unfreeze_last_n):
        """
        Estrategias:
        - linear_probe: congela todo el backbone
        - partial_finetune: congela todo y descongela últimos n bloques
        - full_finetune: entrena todo
        """

        if finetune_strategy == "linear_probe":
            freeze_module(self.backbone)

        elif finetune_strategy == "partial_finetune":
            freeze_module(self.backbone)

            blocks = get_transformer_blocks(self.backbone, self.family)
            unfreeze_last_n = min(unfreeze_last_n, len(blocks))

            for block in blocks[-unfreeze_last_n:]:
                unfreeze_module(block)

            # Opcionalmente descongelar normalizaciones finales si existen.
            if self.family == "clip" and hasattr(self.backbone, "vision_model"):
                vision_model = self.backbone.vision_model
                if hasattr(vision_model, "post_layernorm"):
                    unfreeze_module(vision_model.post_layernorm)

            if self.family == "dinov2":
                if hasattr(self.backbone, "layernorm"):
                    unfreeze_module(self.backbone.layernorm)
                if hasattr(self.backbone, "norm"):
                    unfreeze_module(self.backbone.norm)

        elif finetune_strategy == "full_finetune":
            unfreeze_module(self.backbone)

        else:
            raise ValueError(f"Estrategia no soportada: {finetune_strategy}")

        # El head siempre se entrena.
        unfreeze_module(self.classifier)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)

        pooled = getattr(outputs, "pooler_output", None)

        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled)

        return logits


def get_pretrained_model(
    family,
    hf_id,
    num_classes,
    dropout,
    finetune_strategy,
    unfreeze_last_n,
    device,
):
    model = PretrainedVisionClassifier(
        family=family,
        hf_id=hf_id,
        num_classes=num_classes,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_last_n=unfreeze_last_n,
    )

    return model.to(device)