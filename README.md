```text
torchmetrics
rich
```

## 设计原则

灵活性和可扩展性，扩展而非修改
在不更改源代码文件的情况下、仅通过修改/新建配置文件实现应用不同的模型、数据集、优化算法、损失函数、度量指标等

良好组织与记录
可复现性
最大的兼容性

## 明确概念：

project
    run - workflow 
            stage

project
    exp_1
        run_1 - job_type
        run_2 - job_type
        run_3 - job_type


一个 experiment 是一组 runs，每个 run 都具有一种 job_type


## 目录结构：


## 编写配置文件
配置文件主要用于定义如何实例化相应的对象，
编写配置文件也是一个选择的过程
一般不应该含有复杂的逻辑

在配置组件的源代码文件中，需要定义一些 getter 函数，这是为了将这些配置组件的实例化过程发生在调用 `` 方法时，

命名为 `get_<>_instance`


为了更好的可读性，强制使用关键字参数
配置文件中使用相对路径导入 config components

To Do List:
- [ ] 追踪超参数
- [ ] 补充 docstring
- [ ] tune
- [ ] write run chains 
  - 根据 --fit-resumes-from  --test-predict-ckpt 查找依赖关系、绘制依赖链

## 扩展
### 模型

### 数据集

### task wrapper



```shell
python main.py init -pn MyFirstProject -pd "This is my first project." -en baseline -ed "Baseline exps."
```

```shell
python main.py exec -p 1 -e 1 -c configs/exp_on_oracle_mnist.py -r fit -n SimpleCNN -d "Using a simple CNN."
```

