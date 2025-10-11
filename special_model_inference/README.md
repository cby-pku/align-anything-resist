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