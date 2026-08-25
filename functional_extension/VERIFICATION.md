# Functional extension verification

Not paper fidelity. These tests lock the CPU HuggingFace substitute provider.

## How to run

From the repository root:

```bash
python -m unittest discover -s functional_extension/tests -t functional_extension
```

## Environment

- No GPU, no network, no checkpoints: Transformers / Torch / PIL are mocked
- `transformers>=4.49.0` is declared for real runs (Idefics3 / SmolVLM processors)

## Result

```
Ran 8 tests
OK
```

## What the tests check

| Test | Behavior verified |
|---|---|
| `test_default_model_id_is_smolvlm` | Default id is the documented substitute VLM |
| `test_model_name_returns_specified_id` | Constructor `model_id` is reported as `model_name` |
| `test_construction_does_not_import_transformers` | Instantiation does not import HuggingFace |
| `test_from_pretrained_does_not_use_device_map` | Load uses `.to("cpu")`, not `device_map` (no `accelerate`) |
| `test_generate_decodes_image_bytes` | `Image.data` is decoded; empty-URL `load_image` is not used |
| `test_generate_opens_path_not_empty_url` | `Image.path` opens the file |
| `test_generate_loads_url` | `Image.url` uses `load_image` |
| `test_generate_applies_chat_template_with_image_placeholder` | Chat template includes an image token and the raw prompt |

## Reproduction status

**NOT REPRODUCED.** A public 256M VLM is a substitute for InternVL2-8B, not the paper experiment.
