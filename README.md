## 🐍，移動
BUAASE 结对任务，cpp编写，T3基于DQN

## 编译
根目录下
```
make # 如果你想要编译一个wasm用于所有代码（注意要修改代码的调用）
make lib_t1 # 编译并运行t1
make lib_t2 # 编译并运行t2
make lib_t3 # 编译并运行t3
make test # 编译运行所有测试
```
T1、T2、T3包含已经编译好的wasm，可以运行

## 训练
``train/``下包含所有训练代码和一组权重

## 推理
cpp实现了简单的推理框架，模型权重包含在``param.hpp``
