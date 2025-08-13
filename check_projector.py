from transformers import AutoModelForImageTextToText

# Check LlavaNext
model = AutoModelForImageTextToText.from_pretrained("trl-internal-testing/tiny-LlavaNextForConditionalGeneration")
print("LlavaNext has multi_modal_projector:", hasattr(model, 'multi_modal_projector'))
if hasattr(model, 'multi_modal_projector'):
    print("Projector:", model.multi_modal_projector)

# Check Qwen2.5 VL
model = AutoModelForImageTextToText.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
print("\nQwen2.5 VL has visual:", hasattr(model, 'visual'))
if hasattr(model, 'visual'):
    print("Visual model type:", type(model.visual))
    # Check if it has a projector
    if hasattr(model.visual, 'merger'):
        print("Has merger/projector:", model.visual.merger)
