# Cube-DL-Project-Template

**A lightweight, instant, out-of-the-box Deep Learning project template based on PyTorch and PyTorch-Lightning.**

***Make your Deep Learning life easier and happier.***

***Relive you from chaos of tons of hyper-parameters and experiments.***

![wheels](https://github.com/Alive1024/Cube/actions/workflows/packaging_wheel_on_push.yml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


**目录**：

- [Cube-DL-Project-Template](#cube-dl-project-template)
- [1. 简介](#1-简介)
  - [1.1 动机](#11-动机)
  - [1.2 主要特点](#12-主要特点)
  - [1.3 设计原则](#13-设计原则)
  - [1.4 前置知识](#14-前置知识)
- [2. 项目说明](#2-项目说明)
  - [2.1 关键概念](#21-关键概念)
    - [2.1.1 代码结构的三个部分](#211-代码结构的三个部分)
    - [2.1.2 组织实验的三层结构](#212-组织实验的三层结构)
  - [2.2 配置系统](#22-配置系统)
    - [2.2.1 配置文件](#221-配置文件)
    - [2.2.2 配置参数的自动记录](#222-配置参数的自动记录)
    - [2.2.3 配置文件的自动归档](#223-配置文件的自动归档)
    - [2.2.4 与其他配置方式的比较](#224-与其他配置方式的比较)
  - [2.3 目录结构](#23-目录结构)
  - [2.4 main.py 的命令与参数](#24-mainpy-的命令与参数)
    - [`init`](#init)
    - [`add-exp`](#add-exp)
    - [`ls`](#ls)
    - [`fit`, `validate`, `test`, `predict` 共有的参数](#fit-validate-test-predict-共有的参数)
    - [`validate`, `test`, `predict` 共有的参数](#validate-test-predict-共有的参数)
    - [`fit`](#fit)
    - [`resume-fit`](#resume-fit)
    - [`validate`](#validate)
    - [`test`](#test)
    - [`predict`](#predict)
    - [其他配置项](#其他配置项)
- [3. 工作流](#3-工作流)
  - [3.1 准备模板](#31-准备模板)
  - [3.2 同步模板的更新](#32-同步模板的更新)
  - [3.3 准备依赖](#33-准备依赖)
  - [3.4 进行扩展](#34-进行扩展)
    - [模型](#模型)
    - [数据集](#数据集)
    - [Task Wrapper](#task-wrapper)
    - [Data Wrapper](#data-wrapper)
    - [配置文件](#配置文件)
  - [3.5 进行实验](#35-进行实验)
- [附录](#附录)
  - [配置系统中的参数收集](#配置系统中的参数收集)
  - [c3lyr](#c3lyr)


# 1. 简介

此仓库是一个基于 [PyTorch](https://github.com/pytorch/pytorch) 和 [PyTorch-Lightning](https://lightning.ai/pytorch-lightning/) 的、开箱即用的、非常轻量的深度学习项目模板，用于更省心、省力地组织深度学习实验，使之井井有条。

按需求阅读此文档：

- 如果想先从了解本代码仓库的一些基本概念开始，推荐从头开始阅读；
- 如果想立即动手用起来，可以直接跳转到 [3. 工作流](#3-工作流) 一节即刻上手；
- 如果想了解更多实现细节，请参阅 [附录](#附录)

## 1.1 动机

得益于深度学习领域的飞速发展，开源社区中已经涌现了相当多不同层次的、深度学习相关的开源库。例如，PyTorch 提供了强大的深度学习建模能力，PyTorch-Lightning 则对 PyTorch 进行了抽象和包装，省去了编写大量样板代码的麻烦，但即使有了这些，仍然需要一个项目模板来快速启动，这样的模板值得精心设计，尤其是在配置系统和实验的组织方式等方面。否则很容易因为缺少规则约束，导致大量的实验输出产物与结果分散、堆积在各处，从而陷入混乱，使得研究者/开发者不得不将大量的精力和时间花费在整理和比较实验结果上，而非方法本身。

另外，在进行研究的过程中不可避免地需要使用其他人的开源算法，由于每个人的代码习惯不同，开源算法具有不尽相同的组织结构，部分仓库服务于特定的方法或数据集等，没有经过良好的顶层设计，在使用这些代码进行自定义实验时是相当痛苦的。当想将一些来源不同的算法聚合在一起时，需要一个通用性较强的代码结构。

于是，此深度学习项目模板诞生于此，希望在灵活与抽象之间找到一个良好的平衡点。

## 1.2 主要特点

- **崭新的配置系统**：深度学习项目往往涉及大量的可配置参数，如何省力地配置这些参数是个重要的问题。并且，这些参数往往对最终结果具有关键的影响，因此详细记录这些超参数是十分有必要的。此模板根据深度学习项目的特点重新设计了整个配置系统，使之易用且可追溯；

- **三层组织结构**：为了更有条理地组织大量实验， 所有实验被强制性地分为 Project, Experiment 和 Run 三个层次，每次执行任务都将会自动保存相应记录以供查阅；

- **整洁的目录结构**：所有的输出产物将会有条理地放置到合适的位置。


## 1.3 设计原则
本项目模板在开发过程中尽可能地遵循了以下原则：

- **通用性**：与具体的研究领域无关，在不同领域之间切换时无需从头开始；

- **灵活性和可扩展性**：“扩展而非修改”，当需要实现新的模型、数据集、优化算法、损失函数、度量指标等组件时，尽量不需要更改现有代码，而是通过添加新的代码来实现扩展；
- **良好组织与记录**：每次运行结果都应该被良好地组织、记录；
- **可追溯/可复现性**：导致某个结果的所有变量应该是可以追溯的，保证实验可复现；
- **最大的兼容性**：便于以最低的成本将现有的其他代码迁移到当前代码库中；
- **最低的学习成本**：阅读完 README 即可掌握如何使用，而无需再从几十个页面的文档学习大量 API

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

几部分之间的关系如下所示：

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

为了更有条理地组织所有实验，本模板强制性地要求用户使用“三层结构”：

- **Project** (后文简称 **proj**)：包含多个 exps
- **Experiment** (后文简称 **exp**)：一组具有共同主题的 runs，每个 exp 必须与某个 proj 相关联，例如 “baseline”、“ablation”......
- **Run**：运行的最小原子单位，每个 run 必须与某个 proj 中的某个 exp 相关联，每个 run 都具有一种 job type，指示此 run 在做什么事情

在输出目录中，将会自动组织为类似下面这样的结构：

```text
                         ┌───────────────────────┐
                       ┌▶│proj_75kbcnng_FirstProj│
                       │ └───────────────────────┘               ┌────────────────────┐
                       │             │                         ┌▶│ run_6lwbqdco_first │
                       │             │ ┌─────────────────────┐ │ ├────────────────────┤
                       │             ├▶│exp_7hrm9een_baseline│─┼▶│run_b3n1pjse_second │
                       │             │ └─────────────────────┘ │ ├───────┬────────────┘
        ┌────────────┐ │             │                         └▶│  ...  │
        │   Output   │ │             │ ┌─────────────────────┐   └───────┘
        │ Directory  │─┤             ├▶│exp_3wpy1wa8_ablation│
        └────────────┘ │             │ └─────────────────────┘
                       │             │
                       │             │ ┌───────┐
                       │             └▶│  ...  │
                       │               └───────┘
                       │ ┌───────┐
                       └▶│  ...  │
                         └───────┘
```



在 proj 根目录下，会有一份与 proj 同名的 json 文件，其中的内容是对当前 proj 的所有 exp 和 run 的记录，例如：

```json
{
  "Proj ID": "75kbcnng",
  "Proj Name": "cstu",
  "Proj Desc": "blabla",
  "Created Time": "2023-04-17 (Mon) 04:06:18",
  "Storage Path": "/DL-Project-Template/outputs/proj_75kbcnng_cstu",
  "Logger": "CSV",
  "Exps": {
    "7hrm9een": {
      "Exp Name": "baseline",
      "Exp Desc": "blabla",
      "Created Time": "2023-04-17 (Mon) 04:06:18",
      "Storage Path": "/DL-Project-Template/outputs/proj_75kbcnng_cstu/exp_7hrm9een_baseline",
      "Runs": {
        "b3n1pjse": {
          "Run Name": "dev",
          "Run Desc": "dev",
          "Created Time": "2023-04-17 (Mon) 04:06:52",
          "Storage Path": "/DL-Project-Template/outputs/proj_75kbcnng_cstu/exp_7hrm9een_baseline/run_b3n1pjse_dev",
          "Job Type": "fit"
        }
      }
    }
  }
}
```



## 2.2 配置系统

如前所述，深度学习项目往往涉及大量的可配置参数，如何传入和记录这些参数是十分重要的。考虑到配置的本质是为实例化类提供初始化参数，本模板设计了一套全新的配置系统，编写配置文件就如同编写正常的实例化类的代码一样自然。

### 2.2.1 配置文件

在本模板中，配置文件实际上就是 `.py` 源代码文件，主要用于定义如何实例化相应的对象，编写配置文件即是一个选择(将需要使用的`import`进来)并定义如何实例化的过程。例如，下面是一个配置 root config 的代码片段：

```python
from cube.config_sys import RootConfig, cube_root_config
from .components.task_wrappers.basic_task_wrapper import get_task_wrapper_instance
from .components.data_wrappers.oracle_mnist import get_data_wrapper_instance
from .components.trainers.basic_trainer import get_trainer_instance


@cube_root_config
def get_root_config_instance():
  return RootConfig(
    task_module_getter=get_task_wrapper_instance,
    data_module_getter=get_data_wrapper_instance,
    default_runner_getter=get_trainer_instance,
  )
```

可以看到，在配置文件中，需要将实例化过程放入到一个 "getter" 函数中，最终将实例化的对象 `return`，之所以不是直接在配置文件中实例化某个对象，是为了能够控制实例化配置组件的时机。



配置文件可以和普通的 Python 源代码文件一样包含任意逻辑，但一般不会很复杂，例如下面的 task wrapper 配置文件：

```python
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from wrappers import BasicTaskWrapper
from cube.config_sys import cube_task_module
from ..models.example_cnn_oracle_mnist import get_model_instance


@cube_task_module(model_getter_func=get_model_instance)
def get_task_wrapper_instance():
    model = get_model_instance()
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return BasicTaskWrapper(
        model=model,
        loss_function=loss_func,
        optimizer=optimizer,
        validate_metrics=loss_func,
        test_metrics=[loss_func, MulticlassAccuracy(num_classes=10)],
    )
```



配置文件有以下五种类型：

- **Root Config**: 根配置，所有配置的根结点，包含 task wrapper、data wrapper 和 trainer 三大部分，与前文所述的深度学习三大组件一一对应；
- **Task Wrapper**: 对任务的配置，包含了 model、目标函数、优化算法、验证及测试时使用的度量指标等；
- **Model**: 包含于 task wrapper 中，对某种模型的配置；
- **Data Wrapper**: 对 data wrapper 的配置；
- **Trainer**: 对 trainer (即 PyTorch-Lighnting 中的 `Trainer`) 的配置。

其中，model、task wrapper、data wrapper 和 trainer 都是配置系统中的模块化的、可复用的配置组件，使用时可以自由组合到 root config 中。

五种配置之间的关系如下图所示：

```text
                               ┌────────────Components──────────┐
                               │    ┌────────────┐   ┌─────────┐│
                               │┌──▶│Task Wrapper│──▶│  Model  ││
                               ││   └────────────┘   └─────────┘│
              ┌─────────────┐  ││   ┌────────────┐              │
              │ Root Config │──┼┼──▶│Data Wrapper│              │
              └─────────────┘  ││   └────────────┘              │
                               ││   ┌────────────┐              │
                               │└──▶│  Trainer   │              │
                               │    └────────────┘              │
                               └────────────────────────────────┘
```

对于配置文件的一些强制性规则：

- 为了更好的可读性，在配置文件中初始化 `RootConfig` 时必须使用关键字参数 (模板中给出的 `BasicTaskWrapper`和`BasicDataWrapper`亦是如此，推荐在对 task/data wrapper 进行扩展时也遵循此规则，强制使用关键字参数)；

- Root config 的 getter 函数名必须为 `get_root_config_instance`，其他类型的配置项没有此限制；
- Task wrapper 的 getter 函数必须使用 `config_decorator.py` 中的 `task_wrapper_getter` 装饰器装饰，并向该装饰器传入 `model_getter_func` 参数，就如同上文的  task wrapper 示例中的写法。对于其他配置项，虽然目前不使用相应的装饰器也是没错的，但考虑到将来可能的扩展，强烈建议在编写 getter 函数时都使用`config_decorator.py` 中的相应的装饰器进行装饰。

另外，建议在配置文件中使用相对 import 导入 config components。


### 2.2.3 配置文件的自动归档

为了方便地复现某个 run，每个 run 运行时使用的配置文件都将会被自动归档。默认配置下，相应的 run 的根目录下会保存名为 `archived_config_<RUN_ID>.py` 的配置文件归档，此文件将运行时指定的若干个配置文件融合到一起，形成一个单独的文件，在需要复现此实验时可以直接使用。还可以配置成归档为 zip 压缩包或目录。



### 2.2.4 与其他配置方式的比较

与目前几种主流的配置方式的比较如下：

1. **通过 argparse 定义命令行参数**：一些项目直接使用 argparse 添加可配置的参数，例如 [ViT-pytorch - main.py](https://github.com/jeonsworld/ViT-pytorch/blob/460a162767de1722a014ed2261463dbbc01196b6/train.py#L243)，显而易见，这种配置方式纷繁复杂，当参数数量不断膨胀时，极易出错，在运行时也十分麻烦；
2. **使用 XML/JSON/YAML 等配置文件**：例如 [detectron2](https://github.com/facebookresearch/detectron2) 的部分配置和 PyTorch-Lightning 提供的 [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) 使用了 YAML 文件，这类方法有一个明显的缺陷：IDE 的提示功能十分有限，在编辑时和纯文本文件几乎相同。当配置项繁多时，手写或来回复制粘贴几百行的文本是十分痛苦的，在进行配置时还需要花时间查阅可选择的值，且只能实现简单的逻辑；
3. **使用 OmegaConf 等配置系统库**： [OmegaConf](https://github.com/omry/omegaconf) 是一个基于 YAML 的分层配置系统，支持来自合并多个来源的配置，灵活性很强。但在编写涉及众多参数的深度学习项目时，编辑诸如 YAML 这类文件时，同样要面临编写大量文本文件的麻烦；
4. **实现特定的 Config 类**：例如 [Mask_RCNN - config.py](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py) 中实现了 `Config` 基类，使用时需要派生出子类并按需要覆盖部分属性的值，这种方式灵活性不足，和当前代码紧紧耦合，不适合通用的场景；
5. **一般的 Python 源代码文件**：[OpenMMLab](https://github.com/open-mmlab) 的大部分开源库都采用了这种配置方式，例如 [mmdetection](https://github.com/open-mmlab/mmdetection)，其中的配置文件形如 [atss_r50_fpn_8xb8-amp-lsj-200e_coco.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/atss/atss_r50_fpn_8xb8-amp-lsj-200e_coco.py)。虽然使用了 Python 源代码文件进行配置，但自成体系，有着特殊的规则 (例如需要使用 `_base_` 来继承)，需要付出学习成本，而且本质上是在定义若干个 `dict`，在其中定义要使用的类及其参数，这些参数以 key-value 的形式传入，同样无法充分利用 IDE 的代码提示，与文本型配置方式有着类似的弊端。并且，各种配置项作为变量在配置文件中被直接赋值，是相当松散、容易出错的。

这些配置方式本质上是以各种形式传递参数，然后配置系统将使用这些参数去实例化一些类或传递到某处。而本模板中的配置方式相当于翻转了此过程，在使用时直接定义如何实例化类，配置系统将会自动记录以及归档。这样，编写配置文件的过程如同正常实例化类一样自然，几乎不需要学习如何配置，而且可以充分利用 IDE 的提示来提高编写效率，还可以加入任意的逻辑。



## 2.3 目录结构

代码仓库的目录结构及其含义如下所示：

```text
DL-Project-Template	【项目根目录】
├── c3lyr 【"Core Triple Layers" 的简称，实现三层实体：proj, exp 和 run】
├── config_sys 【"Config System", 实现配置系统】
├── configs 【配置文件存放目录】
│   ├── __init__.py
│   ├── components 【四类配置组件】
│   │   ├── __init__.py
│   │   ├── data_wrappers
│   │   ├── models
│   │   ├── task_wrappers
│   │   └── trainers
│   └── exp_on_oracle_mnist.py 【根配置 (Root config) 文件】
├── data     【数据存放目录】
├── datasets 【数据集类】
├── main.py  【项目入口点】
├── models   【模型定义】
├── outputs  【输出目录，存放所有输出产物】
│   └── proj_75kbcnng_cstu 【创建的 proj】
│       ├── exp_7hrm9een_baseline 【当前 proj 中的 exp】
│       │   ├── run_6lwbqdco_dev  【当前 exp 中的 run】
│       │   │   ├── archived_config_run_6lwbqdco.py 【自动合并、归档的配置文件】
│       │   │   ├── checkpoints   【模型训练期间保存的 checkpoints】
│       │   │   ├── hparams.json  【自动记录的参数】
│       │   │   └── metrics.csv   【度量指标记录】
│       │   └── run_b3n1pjse_dev  【当前 exp 中的 run】
│       │       ├── archived_config_run_b3n1pjse.py
│       │       ├── checkpoints
│       │       ├── hparams.json
│       │       └── metrics.csv
│       └── proj_75kbcnng_cstu.json【此 proj 对应的记录文件】
├── requirements.txt
└── wrappers 【Task/data wrappers 的定义】
    ├── __init__.py
    ├── basic_data_wrapper.py 【常规的 task wrapper】
    ├── basic_task_wrapper.py 【常规的 data wrapper】
    └── wrapper_base.py 【Wrapper 基类】
```

当需要自定义某些组件时，推荐在根目录下分类创建相应的 Python Package，然后将代码文件放置于其中。例如将自定义的优化算法放到根目录下 “optimizer” 包中，自定义的 PyToch-Lightning Callback 放到根目录下 “pl_callbacks”包中。



## 2.4 基本命令与参数


### `init`

初始化一个新的 proj 和 exp。

| 参数名                            | 类型 | 是否必需 | 含义             |
| --------------------------------- | :--: | :------: | ---------------- |
| **-pn**, --proj-name, --proj_name | str  |    ✔️     | 新建 proj 的名称 |
| **-pd**, --proj-desc, --proj_desc | str  |    ✔️     | 新建 proj 的描述 |
| **-en**, --exp-name, --exp_name   | str  |    ✔️     | 新建 exp 的名称  |
| **-ed**, --exp-desc, --exp_desc   | str  |    ✔️     | 新建 exp 的描述  |

示例：

```shell
cube init -pn "MyFirstProject" -pd "This is my first project." -en "Baseline" -ed "Baseline exps."
```



### `add-exp`

向某个 proj 添加一个新的 exp。

| 参数名                       | 类型 | 是否必需 | 含义                       |
| ---------------------------- | :--: | :------: | -------------------------- |
| **-p**, --proj-id, --proj_id | str  |    ✔️     | 新建 exp 所属的 proj 的 ID |
| **-n**, --name               | str  |    ✔️     | 新建 exp 的名称            |
| **-d**, --desc               | str  |    ✔️     | 新建 exp 的描述            |

示例：

```shell
cube add-exp -p 3xp4svcs -n "Ablation" -d "Ablation exps."
```



### `ls`

在终端中以表格的形式显示关于 proj, exp 和 run 的信息。

以下这些参数互斥，使用此子命令时必须指定其中一个。

| 参数名                                     |      类型      |                           含义                           |
|-----------------------------------------|:------------:|:------------------------------------------------------:|
| **-pe**, --projs-exps, --projs_exps     | "store_true" |                    显示所有的 proj 和 exp                    |
| **-p**, --projs                         | "store_true" |                       显示所有的 proj                       |
| **-er**, --exps-runs-of, --exps_runs_of |     str      |              显示指定的 proj ID 下的所有 exp 和 run              |
| **-e**, --exps-of, --exps_of            |     str      |                 显示指定的 proj ID 下所有的 exp                 |
| **-r**, --runs-of, --runs_of            |  str (2 个)   | 显示指定的 proj ID 和 exp ID 下所有的 run (proj ID 在前，exp ID 在后) |

示例：

```shell
cube ls -r p2em5umz 43vfatjk
```



### `fit`, `validate`, `test`, `predict` 共有的参数

`fit`, `validate`, `test`, `predict` 四个子命令都具有下列参数：

| 参数名                               | 类型 | 是否必需 | 含义                       |
| ------------------------------------ | :--: | :------: | -------------------------- |
| **-c**, --config-file, --config_file | str  |    ✔️     | 配置文件的路径             |
| **-p**, --proj-id, --proj_id         | str  |    ✔️     | 新建 run 所属的 proj 的 ID |
| **-e**, --exp-id, --exp_id           | str  |    ✔️     | 新建 run 所属的 exp 的 ID  |
| **-n**, --name                       | str  |    ✔️     | 新建 run 的名称            |
| **-d**, --desc                       | str  |    ✔️     | 新建 run 的描述            |

### `validate`, `test`, `predict` 共有的参数

除以上参数外，`validate`, `test`, `predict` 三个子命令还具有下列参数：

| 参数名                                | 类型 | 是否必需 | 含义                                                         |
| ------------------------------------- | :--: | :------: | ------------------------------------------------------------ |
| **-lc**, --loaded-ckpt, --loaded_ckpt | str  |    ❌     | 要加载的模型 checkpoint 的路径，默认为 None（这意味着不加载任何权重） |



### `fit`

在训练集上进行训练。

示例：

```shell
cube fit -c configs/exp_on_oracle_mnist.py -p 3xp4svcs -e voxc2xhj -n "SimpleCNN" -d "Use a 3-layer simple CNN as baseline."
```



### `resume-fit`

从某个中断的训练中恢复。

| 参数名                               | 类型 | 是否必需 | 含义                                                         |
| ------------------------------------ | :--: | :------: | ------------------------------------------------------------ |
| **-c**, --config-file, --config_file | str  |    ✔️     | 配置文件的路径                                               |
| **-r**, --resume-from, --resume_from | str  |    ✔️     | 要恢复的中断的 fit 的模型 checkpoint 的路径，路径中需要包含 proj、exp 和 run 所在的目录名 (推断 ID 需要这些信息) |

示例：

```shell
cube resume-fit -c configs/exp_on_oracle_mnist.py -r "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/checkpoints/epoch\=3-step\=1532.ckpt"
```



### `validate`

在验证集上进行评估。

示例：

```shell
cube validate -c configs/exp_on_oracle_mnist.py -p 3xp4svcs -e voxc2xhj -n "Val" -d "Validate the simple CNN." -lc "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/checkpoints/epoch=4-step=1915.ckpt"
```



### `test`

在测试集上进行评估。

示例：

```shell
cube test -c configs/exp_on_oracle_mnist.py -p 3xp4svcs -e voxc2xhj -n "Test" -d "Test the simple CNN." -lc "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/checkpoints/epoch=4-step=1915.ckpt"
```



### `predict`

进行预测。

示例：

```shell

```



### 其他配置项

- `OUTPUT_DIR`：输出目录的路径，默认为整个项目根目录下的 `outputs`；
- `ARCHIVED_CONFIGS_FORMAT`：配置文件归档的格式，可选值为：
  - "SIMPLE_PY": 将所有配置文件合并为单个 .py 文件；
  - "ZIP": 保持配置文件原来的目录结构，然后将其打包为 .zip 文件；
  - "DIR": 保持配置文件原来的目录结构，直接拷贝到目的目录。



# 3. 工作流

## 3.3 准备依赖

可以通过以下命令直接安装：

```shell
pip install -r requirements.txt
```



## 3.4 进行扩展

正如前文的 [1.3 设计原则](#13-设计原则) 中的 “扩展而非修改” 原则，此模板提供了尽可能强的扩展能力。



### 模型

可以直接将现有的基于 PyTorch 的模型代码移植到 `models` 中，但需要遵循 PyTorch-Lightning 描述的规则，例如模型代码中不应该包含与硬件相关的代码 (例如调用 `.cuda()`， `.to()` 等)，详见 [Hardware Agnostic Training (Preparation)](https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html?highlight=cuda)。



### 数据集

对于数据，推荐使用软链接 `ln -s <DatasetSrcDir> <data/Dataset>` （这里的 "DatasetSrcDir" 需要是绝对路径）将数据集链接到 `data` 目录中。

对于数据集类，可以直接将现有的数据集类的定义移植到 `datasets` 中。


### Task Module


### Data Module



### 配置文件

按照 [2.2.1 配置文件](#221-配置文件) 所述编写配置文件。



## 3.5 进行实验

下面罗列了常规工作流中命令示例。



**初始化**：

```shell
cube init -pn "MyFirstProject" -pd "This is my first project." -en "Baseline" -ed "Baseline exps."
```



**（可选项）添加新的 exp**:

```shell
cube add-exp -p 3xp4svcs -n "Ablation" -d "Ablation exps."
```



**显示所有的 proj 和 exp**：

```shell
cube ls
```



**进行训练 （proj ID 和 exp ID 可以直接从 `ls` 命令的输出结果复制）**：

```shell
cube fit -c configs/exp_on_oracle_mnist.py -p 3xp4svcs -e voxc2xhj -n "SimpleCNN" -d "Use a 3-layer simple CNN as baseline."
```



**继续中断的训练**：

```shell
cube resume-fit -c configs/exp_on_oracle_mnist.py -r "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/checkpoints/epoch\=3-step\=1532.ckpt"
```



**进行测试**：

```shell
cube test -c configs/exp_on_oracle_mnist.py -p 3xp4svcs -e voxc2xhj -n "Test" -d "Test the simple CNN." -lc "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/checkpoints/epoch=4-step=1915.ckpt"
```



**进行预测**：

```shell

```



**复现某次 run** (通过向子命令的 `-c` 参数传入一个 archived config 的路径)：

```shell
cube fit -c "outputs/proj_3xp4svcs_MyFirstProject/exp_voxc2xhj_Baseline/run_rw4q66gx_SimpleCNN/archived_config_run_rw4q66gx.py" -p 3xp4svcs -e voxc2xhj -n "ReproduceSimpleCNN" -d "Reproduce the 3-layer simple CNN baseline."
```
