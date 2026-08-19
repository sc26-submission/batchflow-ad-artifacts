from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torchmultimodal.models.albef.image_encoder import ALBEFVisionEncoder
    from torchmultimodal.models.albef.model import ALBEFModel, ALBEFModelWithSimilarity
    from torchmultimodal.models.albef.multimodal_encoder import ALBEFMultimodalEncoder
    from torchmultimodal.modules.encoders.bert_text_encoder import bert_text_encoder
    from torchmultimodal.modules.losses.albef import ImageTextContrastiveLoss
except ImportError as exc:  # pragma: no cover - depends on optional W3 dependency
    raise ImportError(
        "COCO/ALBEF workloads require TorchMultimodal. Install the artifact "
        "environment before running W3."
    ) from exc


class ALBEFModelForRetrieval(nn.Module):
    """ALBEF retrieval fine-tuning head used by the COCO workload."""

    def __init__(
        self,
        model_with_similarity: ALBEFModelWithSimilarity,
        itc_loss: ImageTextContrastiveLoss,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.model_with_similarity = model_with_similarity
        self.itc_loss = itc_loss
        self.itm_head = nn.Linear(hidden_size, 2)

    def _train_forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
        idx: Tensor,
        alpha: float,
    ) -> Tensor:
        output = self.model_with_similarity(image, text, text_atts, idx)

        similarity = output.similarity
        itc_loss = self.itc_loss(
            similarity.sim_i2t,
            similarity.sim_t2i,
            similarity.sim_i2t_m,
            similarity.sim_t2i_m,
            output.sim_targets,
            alpha,
        )

        positive = output.multimodal_embeddings[:, 0, :]
        negative = output.multimodal_embeddings_neg[:, 0, :]
        embeddings = torch.cat([positive, negative], dim=0)
        logits = self.itm_head(embeddings)
        labels = torch.cat([
            torch.ones(positive.size(0), dtype=torch.long),
            torch.zeros(negative.size(0), dtype=torch.long),
        ]).to(embeddings.device)

        return itc_loss + F.cross_entropy(logits, labels)

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_atts: Optional[Tensor] = None,
        idx: Optional[Tensor] = None,
        alpha: float = 0.0,
        *,
        is_train: bool = True,
    ) -> Tensor:
        if not is_train:
            raise NotImplementedError("the artifact only uses ALBEF retrieval training")
        if image is None or text is None or text_atts is None or idx is None:
            raise ValueError("ALBEF retrieval training requires image, text, text_atts, and idx")

        return self._train_forward(image, text, text_atts, idx, alpha)


def build_albef_retrieval_model(*, vision_layers: int) -> ALBEFModelForRetrieval:
    """Build the ALBEF retrieval architecture used by W3.

    The four W3 jobs differ only in the number of vision-transformer layers
    (2, 4, 8, or 16). Other dimensions match the retrieval configuration used
    by the previous artifact implementation.
    """
    if vision_layers not in {2, 4, 8, 16}:
        raise ValueError(f"vision_layers must be one of 2, 4, 8, 16, got {vision_layers}")

    hidden_size = 768
    embed_size = 256

    vision_encoder = ALBEFVisionEncoder(
        hidden_size=hidden_size,
        image_size=384,
        patch_size=16,
        num_hidden_layers=vision_layers,
        num_attention_heads=12,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
    )
    text_encoder = bert_text_encoder(
        vocab_size=30522,
        hidden_size=hidden_size,
        type_vocab_size=2,
        max_position_embeddings=512,
        pad_token_id=0,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        dropout=0.0,
    )
    multimodal_encoder = ALBEFMultimodalEncoder(
        hidden_size=hidden_size,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
    )
    vision_proj = nn.Linear(hidden_size, embed_size)
    text_proj = nn.Linear(hidden_size, embed_size)

    albef_model = ALBEFModel(vision_encoder, text_encoder, multimodal_encoder)
    model_with_similarity = ALBEFModelWithSimilarity(
        albef_model,
        vision_proj,
        text_proj,
        embed_size=embed_size,
        queue_size=65536,
        temp=0.07,
    )

    return ALBEFModelForRetrieval(
        model_with_similarity=model_with_similarity,
        itc_loss=ImageTextContrastiveLoss(),
        hidden_size=hidden_size,
    )
