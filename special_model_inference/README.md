OLMo2, Pythia, MiniCPM 系列模型

基础测试：
[] 测试是否有正常的 inference 的代码
[] vllm 是否支持
[] align-anything 训练框架是否支持

进阶测试：
[] 是否有 Resistance 的实验潜质
[] 是否有 Rebound 的实验潜质
[] 是否有 Resistance 的实验效果
[] 是否有 Rebound 的实验效果

-----

# Pythia

基础测试：
[x] 测试是否有正常的 inference 的代码
[x] vllm 是否支持 (用的 GPT-NeoX 的框架)
    - 2.8b 效果还可以，410M 能够正常续写，而且确实看出来用的是一套预训练的文本，回答差不多，而且没有经过专门的安全处理等 
[x] align-anything 训练框架是否支持

进阶测试：
[] 是否有 Resistance 的实验潜质
[] 是否有 Rebound 的实验潜质
[] 是否有 Resistance 的实验效果
[] 是否有 Rebound 的实验效果


# OLMo2

基础测试：
[x] 测试是否有正常的 inference 的代码
[x] vllm 是否支持 (用的OLMo2ForCausalLM的框架)
     - 1B 表现还可以 
[x] align-anything 训练框架是否支持

进阶测试：
[] 是否有 Resistance 的实验潜质
[] 是否有 Rebound 的实验潜质
[] 是否有 Resistance 的实验效果
[] 是否有 Rebound 的实验效果


# MiniCPM

基础测试：
[0] 测试是否有正常的 inference 的代码, 正常 inference 代码会有 DynamicCache 的报错, 但是 vllm 对这个问题处理好了 
[x] vllm 是否支持 (用的OLMo2ForCausalLM的框架)
     - 1B 表现还可以 
     - " 注意：我们发现使用Huggingface生成质量略差于vLLM，因此推荐使用vLLM进行测试。我们正在排查原因。"
     - 主要是中文模型 
     - trust_remote_mode = True , minicpm 系列的模型都有自己的pretrained 写法 
     - bf16 精度更准 
[x] align-anything 训练框架是否支持

进阶测试：
[] 是否有 Resistance 的实验潜质
[] 是否有 Rebound 的实验潜质
[] 是否有 Resistance 的实验效果
[] 是否有 Rebound 的实验效果