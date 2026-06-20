# Devstral Small 2 (24B) on vLLM — vision-capable inference image.
#
# A self-contained image you can run on any Docker-based GPU host (Koyeb, Berget,
# bare metal, k8s). Based on the official vLLM image; adds one source guard so the
# Pixtral (vision) multimodal init doesn't crash on this transformers build.
#
# The stock vllm/vllm-openai:v0.22.1 image crashes at engine-init building a dummy
# image:
#   AttributeError: 'MistralCommonImageProcessor' object has no attribute 'fetch_images'
# (transformers/processing_utils.py: prepare_inputs_layout unconditionally calls
# fetch_images, which the mistral-common image processor doesn't implement).
# fetch_images only downloads URL images; already-loaded PIL images pass through
# unchanged — so we guard the call: use it if present, else pass images through.
#
# The grep makes the BUILD fail loudly if the upstream line ever moves.

FROM vllm/vllm-openai:v0.22.1

# Pin the transformers / mistral_common pair (matches our reference deployment).
RUN pip install -q --no-cache-dir transformers==5.12.1 mistral_common==1.11.3

# Guard the fetch_images call.
RUN F=/usr/local/lib/python3.12/dist-packages/transformers/processing_utils.py && \
    sed -i 's/self\.image_processor\.fetch_images(images)/getattr(self.image_processor, "fetch_images", lambda x: x)(images)/' "$F" && \
    grep -q 'getattr(self.image_processor, "fetch_images", lambda x: x)(images)' "$F"
