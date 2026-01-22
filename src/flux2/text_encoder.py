from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    TorchAoConfig,
    pipeline,
)

from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

from .sampling import cap_pixels, concatenate_images
from .system_messages import (
    PROMPT_IMAGE_INTEGRITY,
    PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
    PROMPT_TEXT_INTEGRITY,
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
    SYSTEM_PROMPT_CONTENT_FILTER,
)

OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512
NSFW_THRESHOLD = 0.85
UPSAMPLING_MAX_IMAGE_SIZE = 768**2


# Combined FP8 model with tokenizer (single repo for simplified deployment)
MISTRAL_FP8_MODEL_SPEC = "TensorTemplar/flux2-text-encoder-fp8"
MISTRAL_BF16_MODEL_SPEC = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


class Mistral3SmallEmbedder(nn.Module):
    def __init__(
        self,
        model_spec: str = MISTRAL_BF16_MODEL_SPEC,
        torch_dtype: str = "bfloat16",
        use_fp8: bool = False,
        quantization: Literal["torchao_fp8", "compressed_fp8", "none"] = "none",
        enable_moderation: bool = False,
    ):
        super().__init__()

        # Handle legacy use_fp8 parameter - map to quantization
        if use_fp8 and quantization == "none":
            quantization = "compressed_fp8"

        # Use FP8 model spec for compressed_fp8 quantization
        if quantization == "compressed_fp8":
            model_spec = MISTRAL_FP8_MODEL_SPEC

        if quantization == "torchao_fp8":
            # True FP8 compute using torchao - native tensor core ops, no dequantization
            quant_config = Float8DynamicActivationFloat8WeightConfig()
            torchao_config = TorchAoConfig(quant_type=quant_config)
            self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
                model_spec,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=torchao_config,
                local_files_only=True,
            )
        elif quantization == "compressed_fp8":
            # FP8 storage with bf16 compute (dequantizes on forward pass)
            self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
                model_spec,
                local_files_only=True,
            )
        else:
            # No quantization - standard bf16
            self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
                model_spec,
                torch_dtype=getattr(torch, torch_dtype),
                local_files_only=True,
            )

        # For torchao_fp8, processor comes from bf16 model spec (weights are quantized on-the-fly)
        processor_spec = model_spec if quantization != "torchao_fp8" else MISTRAL_BF16_MODEL_SPEC
        self.processor = AutoProcessor.from_pretrained(processor_spec, use_fast=False, local_files_only=True)
        self.yes_token, self.no_token = self.processor.tokenizer.encode(
            ["yes", "no"], add_special_tokens=False
        )

        self.max_length = MAX_LENGTH
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

        self.enable_moderation = enable_moderation
        self.nsfw_classifier = None
        if enable_moderation:
            self.nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    def _validate_and_process_images(
        self, img: list[list[Image.Image]] | list[Image.Image]
    ) -> list[list[Image.Image]]:
        if not img:
            return []

        if isinstance(img[0], Image.Image):
            img = [[im] for im in img]

        img = [[concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in img]
        img = [[cap_pixels(img_i, self.upsampling_max_image_size) for img_i in img_i] for img_i in img]
        return img

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[list[dict]]:
        """
        Format a batch of text prompts into the conversation format expected by apply_chat_template.
        Optionally, add images to the input.

        Args:
            txt: List of text prompts
            system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
            img: List of images to add to the input.

        Returns:
            List of conversations, where each conversation is a list of message dicts
        """
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        if img is None or len(img) == 0:
            return [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                for prompt in cleaned_txt
            ]
        else:
            assert len(img) == len(txt), "Number of images must match number of prompts"
            img = self._validate_and_process_images(img)

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                ]
                for _ in cleaned_txt
            ]

            for i, (el, images) in enumerate(zip(messages, img)):
                if images is not None:
                    el.append(
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": image_obj} for image_obj in images],
                        }
                    )
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cleaned_txt[i]}],
                    }
                )

            return messages

    @torch.no_grad()
    def upsample_prompt(
        self,
        txt: list[str],
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
        temperature: float = 0.15,
    ) -> list[str]:
        """
        Upsample prompts using the model's generate method.

        Args:
            txt: List of input prompts to upsample
            img: Optional list of images or list of lists of images. If None or all None, uses t2i mode, otherwise i2i mode.

        Returns:
            List of upsampled prompts
        """
        # Set system message based on whether images are provided
        if img is None or len(img) == 0 or img[0] is None:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I

        # Format input messages
        messages_batch = self.format_input(txt=txt, system_message=system_message, img=img)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        try:
            inputs = self.processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048,
            )
        except ValueError as e:
            print(
                f"Error processing input: {e}, your max length is probably too short, when you have images in the input."
            )
            raise e

        # Move to device
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device, self.model.dtype)

        # Generate text using the model's generate method
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
            )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]

            raw_txt = self.processor.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return raw_txt
        except Exception as e:
            print(f"Error generating upsampled prompt: {e}, returning original prompt")
            return txt

    @torch.no_grad()
    def forward(self, txt: list[str]):
        # Format input messages
        messages_batch = self.format_input(txt=txt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Sets all tokens but yes/no to the minimum.
        """
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    def test_image(self, image: Image.Image | str | Path | torch.Tensor) -> bool:
        if not self.enable_moderation:
            return False  # Not NSFW when moderation disabled

        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, (str, Path)):
            image = Image.open(image)

        classification = next(c for c in self.nsfw_classifier(image) if c["label"] == "nsfw")
        if classification["score"] > NSFW_THRESHOLD:
            return True

        # 512^2 pixels are enough for checking
        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )

        return generate_ids[0, -1].item() == self.yes_token

    def test_txt(self, txt: str) -> bool:
        if not self.enable_moderation:
            return False  # Not flagged when moderation disabled

        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype="auto",
            device_map=str(device),
            local_files_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_spec, local_files_only=True)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def test_txt(self, txt: str) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support text testing")

    def test_image(self, image) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support image testing")

    def upsample_prompt(self, txt: list[str], img=None, **kwargs) -> list[str]:
        raise NotImplementedError("Qwen3Embedder does not support upsampling")


def load_mistral_small_embedder(
    device: str | torch.device = "cuda",
    use_fp8: bool = False,
    quantization: Literal["torchao_fp8", "compressed_fp8", "none"] = "none",
    enable_moderation: bool = False,
) -> Mistral3SmallEmbedder:
    embedder = Mistral3SmallEmbedder(
        use_fp8=use_fp8,
        quantization=quantization,
        enable_moderation=enable_moderation,
    )
    # For torchao_fp8, model is already on device via device_map="auto"
    if quantization != "torchao_fp8":
        embedder = embedder.to(device)
    return embedder


def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    return Qwen3Embedder(model_spec=f"Qwen/Qwen3-{variant}-FP8", device=device)
