# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)

# Create an LLM.
llm = LLM(
    model="/home/data/weights/deepseek-ai/deepseekv3-lite-base-latest",
    # model="/home/data/weights/facebook/opt-125m",
    tensor_parallel_size=4,
    distributed_executor_backend="mp",
    trust_remote_code=True,
    enforce_eager=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
# llm.stop_profile()
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")