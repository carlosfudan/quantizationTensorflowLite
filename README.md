# 基于TensorflowLite的模型量化

## 环境
- tensorflow 2.3

## 量化技术（训练后量化）

- float16 量化
- int8 量化
- 动态量化

## 使用说明

- 修改配置文件
```
vim config.py 
```
- 训练模型
```
python main.py 
```
- 模型量化
```
python quantize.py 
```
- 使用量化后的模型进行预测
```
python infernce.py 

````