```text
torchmetrics
rich
```



[TOC]

# 1. 简介

此仓库是一个基于 [PyTorch](https://github.com/pytorch/pytorch) 和 [PyTorch-Lightning](https://lightning.ai/pytorch-lightning/) 的、开箱即用的、非常轻量的深度学习项目模板，用于更省心、省力地组织深度学习实验，使之井井有条。

## 1.1 动机

得益于深度学习领域的飞速发展，开源社区中已经涌现了相当多不同层次的、深度学习相关的开源库。例如，PyTorch 提供了强大的深度学习建模能力，PyTorch-Lightning 则对 PyTorch 进行了抽象和包装，省去了编写大量样板代码的麻烦，但即使有了这些，仍然需要一个项目模板来快速启动，这样的模板值得精心设计，尤其是在配置系统和实验的组织方式等方面。否则很容易因为缺少规则约束，导致大量的实验输出产物与结果分散、堆积在各处，从而陷入混乱，使得研究者/开发者不得不将大量的精力和时间花费在整理和比较实验结果上，而非方法本身。

另外，在进行研究的过程中不可避免地需要使用其他人的开源算法，由于每个人的代码习惯不同，开源算法具有不尽相同的组织结构，部分仓库服务于特定的方法或数据集等，没有经过良好的顶层设计，在使用这些代码进行自定义实验时是相当痛苦的。当想将一些来源不同的算法聚合在一起时，需要一个通用性较强的代码结构。

于是，此深度学习项目模板诞生于此，希望在灵活与抽象之间找到一个良好的平衡点。

## 1.2 主要特点

- **崭新的配置系统**：深度学习项目往往涉及大量的可配置参数，如何省力地配置这些参数是个重要的问题。并且，这些参数往往对最终结果具有关键的影响，因此详细记录这些 超参数是十分有必要的。此模板根据深度学习项目的特点重新设计了整个配置系统，使之易用且可追溯。

- **三层组织结构**：为了更有条理地组织大量实验， 所有实验被强制性地分为 Project, Experiment 和 Run 三个层次。

- **目录结构**：所有的输出产物将会有条理地


## 1.3 设计理念
本项目模板在开发过程中尽可能地遵循了以下原则：

- **通用性**：与具体的研究领域无关，在不同领域之间切换时无需从头开始

- **灵活性和可扩展性**：“扩展而非修改”，当需要实现新的模型、数据集、优化算法、损失函数、度量指标等组件时，尽量不需要更改现有代码，而是通过添加新的代码来实现扩展
- **良好组织与记录**：每次运行结果都应该被良好地组织、记录
- **可追溯/可复现性**：导致某个结果的所有变量应该是可以追溯的，保证实验可复现
- **最大的兼容性**：便于以最低的成本将现有的其他代码迁移到当前代码库中
- **最低的学习成本**：阅读完 README 即可掌握如何使用，而无需从几十个页面的文档学习大量 API

## 1.4 前置知识

使用者应对 Python、PyTorch 和 PyTorch-Lightning 有基本了解。





# 2. 项目说明

## 2.1 关键概念

### 2.1.1 代码结构的三个部分

一般来说，深度学习中的核心组件包括 [<sup>1</sup>](https://d2l.ai/chapter_introduction/index.html#key-components)：
- 可以学习的**数据**
- 转换数据的**模型**
- 量化模型有效性的**目标函数**
- 调节模型参数以优化目标函数的**优化算法**

基于以上分类和组件化的思想，本模板将深度学习项目的相关组件重新组织为三大部分：

- **Task Wrapper**: 对某种深度学习任务的包装，例如 [`BasicTaskWrapper`](./wrappers/basic_task_wrapper.py) 中定义了使用单个优化算法的常规 task wrapper。本质上是 PyTorch-Lightning 的 [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)，不过与使用 LightningModule 的常规方式不同，这里的 task wrapper 不仅仅是模型本身，还应该包括目标函数、优化算法、学习率调节器、验证及测试时使用的度量指标等。当需要将现有的基于 PyTorch 定义的模型加入到现有代码库中时，如果现有代码库已有支持 task wrapper，则可以直接使用该模型，而不需要将其改写为 LightningModule。
- **Data Wrapper**: 对 PyTorch 的 Dataset 类和 DataLoader 类的包装，例如 [`BasicDataWrapper`](./wrappers/basic_data_wrapper.py) 中定义了适用于常规数据集的 data wrapper。本质上是 PyTorch-Lightning 的 [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)，但类似于 task wrapper，data wrapper 不特定于某个数据集，具体的数据集类作为 data wrapper 的初始化参数传入。当需要将现有的基于 PyTorch 定义的数据集类加入到现有代码库中时，大多情况下都可以直接使用 `BasicDataModule`，而无需将其改写为 LightningDataModule。
- **Trainer**: 即 PyTorch-Lightning 的 [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)，包含了进行模型训练、验证、测试、推理等工程层面的代码。

```text
                                                   ┌────────────────┐
                                                   │     Model      │
                                                   ├────────────────┤
                                                   │ Loss Function  │
                                                   ├────────────────┤
                            ┌────────────────┐     │   Optimizer    │
                            │  Task Wrapper  │─────▶────────────────┤
                            └────────────────┘     │  LR Scheduler  │
                                                   ├────────────────┤
                                                   │Val/Test Metrics│
                                                   ├────────────────┤
                                                   │     ......     │
                                                   └────────────────┘
                                                   ┌────────────────┐
                                                   │    Datasets    │
                            ┌────────────────┐     ├────────────────┤
                            │  Data Wrapper  │─────▶  Batch Sizes   │
                            └────────────────┘     ├────────────────┤
                                                   │     ......     │
                                                   └────────────────┘
                                                                     
                            ┌────────────────┐                       
                            │    Trainer     │                       
                            └────────────────┘                       
```



### 2.1.2 组织实验的三层结构

为了更有条理地组织所有实验，本模板强制性地要求用户以下面的三层结构：

- **Project** (后文简称 **proj**)：包含多个 exps
- **Experiment** (后文简称 **exp**)：一组具有共同主题的 runs，例如 “baseline”、“ablation”......
- **Run**：运行的最小原子单位，每个 run 必须与某个 proj 中的某个 exp 相关联，每个 run 都具有一种 job type，指示此 run 在做什么事情



## 2.2 配置系统

### 2.2.1 与其他配置方式的比较

argparse

YAML / 

​	LightningCLI

一般的 Python 源代码文件

​	mmdet

​	一些直接



### 2.2.2 配置文件

**Root Config**: 根配置，



配置文件主要用于定义如何实例化相应的对象，
编写配置文件也是一个选择的过程
一般不应该含有复杂的逻辑

在配置组件的源代码文件中，需要定义一些 getter 函数，这是为了将这些配置组件的实例化过程发生在调用 `` 方法时，

命名为 `get_<>_instance`

- root_config
- task_wrapper
- data_wrapper
- model

为了更好的可读性，强制使用关键字参数
配置文件中使用相对路径导入 config components



## 2.3 目录结构

```text
configs
data
datasets
```



当需要自定义某些组件时，推荐在根目录下分类创建相应的 Python Package，然后将代码文件放置于其中。例如将自定义的优化算法放到根目录下 “optimizer” 包中，自定义的 PyToch-Lightning Callback 放到根目录下 “pl_callbacks”包中。



## 2.4 main.py 的命令与参数

`main.py`是所有入口点，包含`init`、`add-exp`和`exec`三个子命令，其作用及参数如下：

### `init`

初始化一个新的 proj 和 exp

| 参数名                        | 类型 | 是否必需 | 含义        |
| ----------------------------- | :--: | :------: | ----------- |
| -pn, --proj-name, --proj_name | str  |    ✔️     | proj 的名称 |
| -pd, --proj-desc, --proj_desc | str  |    ✔️     | proj 的描述 |
| -en, --exp-name, --exp_name   | str  |    ✔️     | exp 的名称  |
| -ed, --exp-desc, --exp_desc   | str  |    ✔️     | exp 的描述  |

示例：

```shell
python main.py init -pn "MyFirstProject" -pd "This is my first project." -en "Baseline" -ed "Baseline exps."
```



### `add-exp`

向某个 proj 添加一个新的 exp

### `exec`

执行 run(s)





### 其他配置项

- `OUTPUT_DIR`

- `GLOBAL_SEED`





# 3. 工作流示例

## 3.1 准备模板及其依赖

- 直接将此模板克隆到本地：`git clone https://github.com/Alive1024/DL-Project-Template.git`

- Github



## 3.2 进行扩展

### Task Wrapper

### Data Wrapper

### 模型

### 数据集

推荐使用软链接 `ln -s <DatasetSrcDir> <data/Dataset>`将数据集



### 配置文件



## 3.3 进行实验

初始化：

```shell
python main.py init -pn MyFirstProject -pd "This is my first project." -en baseline -ed "Baseline exps."
```



执行

```shell
python main.py exec -p 1 -e 1 -c "configs/exp_on_oracle_mnist.py" -r "fit,test" -n "SimpleCNN" -d "Using a simple CNN."
```



添加新的 exp:

```shell

```



继续中断的训练：

```shell

```
