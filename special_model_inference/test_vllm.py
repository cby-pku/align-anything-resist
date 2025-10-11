from vllm import LLM, SamplingParams

model_name_or_path = '/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-2.8b'
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=model_name_or_path, trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")