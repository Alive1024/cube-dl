# cube-dl

Languages: [English](../README.md) | 简体中文

**训练深度学习模型的"最后一站"。**

**对现有代码的进行少量更改，来管理大量的配置项和实验。**

[![Packaging Wheel](https://github.com/Alive1024/cube-dl/actions/workflows/packaging_wheel_on_push.yml/badge.svg)](https://github.com/Alive1024/cube-dl/actions/workflows/packaging_wheel_on_push.yml)
[![Publishing to PyPI](https://github.com/Alive1024/cube-dl/actions/workflows/publishing_on_tag.yml/badge.svg)](https://github.com/Alive1024/cube-dl/actions/workflows/publishing_on_tag.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**从 PyPI 安装（稳定状态，推荐）**：

```shell
pip install -U cube-dl
```

**使用 wheel 文件安装（最新状态）**：
进入本项目的 [Actions](https://github.com/Alive1024/cube-dl/actions) 页面， 在 "Packaging Wheel" 对应的 actions 中选择最新的一个 workflow run，在 Artifacts 中下载 wheel 文件的压缩包，解压后使用 pip 安装：

```shell
pip install xxx.whl
```

**从源码安装（最新状态）**：

```shell
git clone git@github.com:Alive1024/cube-dl.git
cd cube-dl
pip install .
```

**目录**：

- [cube-dl](#cube-dl)
- [1. 简介](#1-简介)
  - [1.1 动机](#11-动机)
  - [1.2 主要特点](#12-主要特点)
  - [1.3 设计原则](#13-设计原则)
  - [1.4 前置知识](#14-前置知识)
- [2. 项目说明](#2-项目说明)
  - [2.1 关键概念](#21-关键概念)
    - [2.1.1 四种核心组件](#211-四种核心组件)
    - [2.1.2 组织实验的三层结构](#212-组织实验的三层结构)
  - [2.2 配置系统](#22-配置系统)
    - [2.2.1 配置文件](#221-配置文件)
    - [2.2.3 配置文件的自动归档](#223-配置文件的自动归档)
    - [2.2.4 在不同配置文件之间共享预设值](#224-在不同配置文件之间共享预设值)
    - [2.2.5 与其他配置方式的比较](#225-与其他配置方式的比较)
  - [2.3 Starter](#23-starter)
  - [2.4 Starter 的目录结构](#24-starter-的目录结构)
  - [2.4 基本命令与参数](#24-基本命令与参数)
    - [`start`](#start)
    - [`new`](#new)
    - [`add-exp`](#add-exp)
    - [`ls`](#ls)
    - [`fit`, `validate`, `test`, `predict` 共有的参数](#fit-validate-test-predict-共有的参数)
    - [`validate`, `test`, `predict` 共有的参数](#validate-test-predict-共有的参数)
    - [`fit`](#fit)
    - [`resume-fit`](#resume-fit)
    - [`validate`](#validate)
    - [`test`](#test)
    - [`predict`](#predict)
  - [2.5 其他](#25-其他)
    - [2.5.1 回调函数](#251-回调函数)
    - [2.5.2 运行时上下文](#252-运行时上下文)

# 1. 简介

**cube-dl** 是一个用于管理和训练深度学习模型的高层次 Python 库，用于更省心、省力地管理大量的深度学习配置项和实验，使之井井有条。

## 1.1 动机

如我们所见，开源社区中已有相当多不同层次的、深度学习相关库。例如，[PyTorch](https://github.com/pytorch/pytorch) 提供了强大的深度学习建模能力，[PyTorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning) 则对 PyTorch 进行了抽象和包装，省去了编写大量样板代码的麻烦，但即使有了这些，在训练深度学习模型时仍然可能因为大量的可配置项、实验等陷入混乱，使得研究者/开发者不得不将大量的精力和时间花费在整理和比较实验结果上，而非方法本身。另外，在进行研究的过程中不可避免地需要使用其他人的开源算法，由于每个人的代码习惯不同，开源算法具有不尽相同的组织结构，部分仓库服务于特定的方法或数据集等，没有经过良好的顶层设计，在使用这些代码进行自定义实验时是相当痛苦的。再者，当想将一些来源不同的算法聚合在一起时，需要一个通用性较强的代码结构。

**cube-dl** 因此诞生，通过在配置和实验的管理方式等方面施加一些规则约束来使得深度学习项目更易于管理，并在抽象与灵活之间找到一个良好的平衡点。

## 1.2 主要特点

- **组件化**：深度学习模型的训练过程中涉及的元素被明确地划分为四个部分，以实现低耦合度和高复用性；

- **崭新的配置系统**：深度学习项目往往涉及大量的可配置参数，如何省力地配置这些参数是个重要的问题。并且，这些参数往往对最终结果具有关键的影响，因此详细记录这些参数是十分有必要的。cube-dl 根据深度学习项目的特点重新设计了整个配置系统，使之易用且可追溯；

- **三层组织结构**：为了更有条理地组织大量实验， 所有实验被强制性地分为 Project, Experiment 和 Run 三个层次，每次执行任务都将会自动保存相应记录以供查阅；

- **简洁快速的 CLI**：cube-dl 提供了一套简洁的 CLI，可以通过少量的几个命令进行管理、训练和测试等。


## 1.3 设计原则
cube-dl 尽可能地遵循了以下原则：

- **通用性**：与具体的研究领域无关，在不同领域之间切换时无需从头开始；
- **灵活性和可扩展性**：“扩展而非修改”，当需要实现新的模型、数据集、优化算法、损失函数、度量指标等组件时，尽量不需要更改现有代码，而是通过添加新的代码来实现扩展；
- **良好组织与记录**：每次运行结果都应该被良好地组织、记录；
- **最大的兼容性**：便于以最低的成本将现有的其他代码迁移到当前代码库中；
- **最低的学习成本**：阅读完 README 即可掌握如何使用，而无需再从几十个页面的文档学习大量 API

## 1.4 前置知识

使用者应对 Python 和 PyTorch 有基本了解。


# 2. 项目说明

## 2.1 关键概念

### 2.1.1 四种核心组件

一般来说，深度学习中的核心组件包括 [<sup>1</sup>](https://d2l.ai/chapter_introduction/index.html#key-components)：

- 可以学习的**数据**
- 转换数据的**模型**
- 量化模型有效性的**目标函数**
- 调节模型参数以优化目标函数的**优化算法**

基于以上分类和组件化的思想，cube-dl 将深度学习项目的相关组件重新组织为四部分：

- **Model**: 即要训练的模型；
- **Task Module**: 对某种深度学习任务的过程的定义，对应的是某种训练范式，例如最常见的全监督式学习。Task Module 可进一步细分为若干个组件，例如损失函数、优化算法、学习率调节器、验证及测试时使用的度量指标等。同时，要训练的模型作为 Task Module 的初始化参数指定；
- **Data Module**: 与数据相关的，对应于 PyTorch 的 Dataset 和 DataLoader 的组合，类似于 PyTorch-Lightning 的 [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)，但用法略有不同，这里的 Data Module 不特定于某个数据集，具体的数据集类作为 Data Module 的初始化参数传入；
- **Runner**: 执行模型训练、验证、测试、推理等过程的工程层面的代码。

```text
                        ┌────────────────┐
                        │     Model      │
                        └────────────────┘     ┌────────────────┐
                                               │ Loss Function  │
                                               ├────────────────┤
                        ┌────────────────┐     │   Optimizer    │
                        │  Task Module   │─────▶────────────────┤
                        └────────────────┘     │  LR Scheduler  │
                                               ├────────────────┤
                                               │Val/Test Metrics│
                                               ├────────────────┤
                                               │     ......     │
                                               └────────────────┘
                                               ┌────────────────┐
                                               │    Datasets    │
                        ┌────────────────┐     ├────────────────┤
                        │  Data Module   │─────▶  Batch Sizes   │
                        └────────────────┘     ├────────────────┤
                                               │     ......     │
                                               └────────────────┘

                        ┌────────────────┐
                        │     Runner     │
                        └────────────────┘
```



### 2.1.2 组织实验的三层结构

为了更有条理地组织所有实验，cube-dl 强制性地要求用户使用“三层结构”：

- **Project** (后文简称 **proj**)：包含多个 exps；
- **Experiment** (后文简称 **exp**)：一组具有共同主题的 runs，每个 exp 必须与某个 proj 相关联，例如 “baseline”、“ablation”、“contrast”等；
- **Run**：运行的最小原子单位，每个 run 必须隶属于某个 proj 中的某个 exp，每个 run 都具有一种 job type，指示此 run 在做什么事情。

以上三种实体都具有相应的由小写字母和数字构成的随机 ID，proj 和 exp 的 ID 为 2 位；run 的 ID 为 4 位。

输出目录的结构将形如：

```text
                 ┌───────────────────────┐
               ┌▶│   proj_6r_DummyProj   │
               │ └───────────────────────┘          ┌─────────────────────┐
               │             │                   ┌─▶│run_z2hi_fit_DummyRun│
               │             │ ┌───────────────┐ │  └─────────────────────┘
               │             ├▶│exp_1b_DummyExp│─┤
               │             │ └───────────────┘ │  ┌───────┐
┌────────────┐ │             │                   └─▶│  ...  │
│   Output   │ │             │ ┌───────┐            └───────┘
│ Directory  │─┤             └▶│  ...  │
└────────────┘ │               └───────┘
               │
               │ ┌───────┐
               └▶│  ...  │
                 └───────┘
```

在 proj 根目录下，会有一份与 proj 同名的 json 文件，其中的内容是对当前 proj 的所有 exp 和 run 的记录，例如：

```json
{
  "ID": "6r",
  "Name": "DummyProj",
  "Desc": "This is a dummy proj for demonstration.",
  "CreatedTime": "2024-03-18 22:11:15",
  "Path": "./outputs/proj_6r_DummyProj",
  "Exps": {
    "1b": {
      "Name": "DummyExp",
      "Desc": "This is a dummy exp for demonstration.",
      "CreatedTime": "2024-03-18 22:11:15",
      "Path": "./outputs/proj_6r_DummyProj/exp_1b_DummyExp",
      "Runs": {
        "z2hi": {
          "Name": "DummyRun",
          "Desc": "A dummy run for demonstration.",
          "CreatedTime": "2024-03-18 22:12:49",
          "Path": "./outputs/proj_6r_DummyProj/exp_1b_DummyExp/run_z2hi_fit_DummyRun",
          "Type": "fit"
        }
      }
    }
  }
}
```

在默认情况下，这些 proj 记录文件将会被 git 追踪，以便于多人以分布式的方式通过 git 进行协作。这意味着用户 A 创建的 proj、exp 和 run 可以被用户 B 看到（但 run 的输出产物不会被 git 追踪）。



## 2.2 配置系统

如前所述，深度学习项目往往涉及大量的可配置参数，如何传入和记录这些参数是十分重要的。考虑到配置的本质是为实例化类提供初始化参数，cube-dl 设计了一套全新的配置系统，编写配置文件就如同编写正常的实例化类的代码一样自然。

### 2.2.1 配置文件

在 cube-dl 中，配置文件实际上就是 `.py` 源代码文件，主要用于定义如何实例化相应的对象，编写配置文件即是一个选择(将需要使用的`import`进来)并定义如何实例化的过程。例如，下面是一个配置 runner 的代码片段：

```python
@cube_runner
def get_fit_runner():
    run = CUBE_CONTEXT["run"]
    return pl.Trainer(
        accelerator="auto",
        max_epochs=shared_config.get("max_epochs"),
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                dirpath=osp.join(run.run_dir, "checkpoints"),
                filename="{epoch}-{step}-{val_mean_acc:.4f}",
                save_top_k=1,
                monitor="val_mean_acc",
                mode="max",
            ),
        ],
        logger=get_csv_logger(run),
    )
```

可以看到，在配置文件中，需要将实例化过程放入到一个 "getter" 函数中，最终将实例化的对象 `return`，之所以不是直接在配置文件中实例化某个对象，是为了允许 cube-dl 控制实例化配置项的时机。

由于配置文件本质上就是 Python 源代码文件，因此可以和普通的 Python 源代码文件一样包含任意逻辑，但一般不会很复杂。

与前文描述的四种核心组件相对应，核心配置项也主要有四种，分别是 `cube_model`、`cube_task_module`、`cube_data_module` 和 `cube_runner`，这些配置项可以作为配置系统中的模块化的、可复用的配置组件。除此之外， 实际进行实验时还需要将四种组件自由组合起来、形成 **Root Config**，这是所有配置的根结点。

五种配置项之间的关系如下：

```text
                                  ┌────────────────────────┐
                                  │       Components       │
                                  │     ┌────────────┐     │
                                  │ ┌──▶│  Model(s)  │──┐  │
                                  │ │   └────────────┘  │  │
                                  │ │                   │  │
                  ┌─────────────┐ │ │   ┌────────────┐  │  │
                  │ Root Config │─┼─┼──▶│Task Module │◀─┘  │
                  └─────────────┘ │ │   └────────────┘     │
                                  │ │   ┌────────────┐     │
                                  │ ├──▶│Data Module │     │
                                  │ │   └────────────┘     │
                                  │ │   ┌────────────┐     │
                                  │ └──▶│ Runner(s)  │     │
                                  │     └────────────┘     │
                                  └────────────────────────┘
```

对于配置文件的一些规则：

- 为了更好的可读性，在配置文件中初始化 `RootConfig` 时必须使用关键字参数 (推荐在编写 task/data modules 时也遵循此规则，强制使用关键字参数)；

- Root config 的 getter 函数名必须为 `get_root_config`，每个配置文件中仅能有一个，其他类型的配置项没有此限制；

- Task module 的 getter 函数必须有一个名为 `model` 的参数，对应于传给 root config 的 `model_getters`， 此参数用于传入模型对象，这在 task module 中配置的优化器等配置项中都需要用到。当传给 `model_getters` 的是一个列表 (表示多个模型) 时，`model` 参数也将是一个列表。

- `cube_dl.config_sys` 中可以导入名为 `cube_root_config`、 `cube_model`、`cube_task_module`、`cube_data_module` 和 `cube_runner` 的装饰器，强烈建议在编写 getter 函数时都使用相应的装饰器进行装饰，一方面是为了允许装饰器进行检查，另一方面是为了将来的扩展。

另外，当需要从其他配置文件中导入所需的 config components 时，建议使用相对 import 语句导入。

### 2.2.3 配置文件的自动归档

为了方便地复现某个 run，每个 run 运行时使用的配置文件都将会被自动归档。默认配置下，相应的 run 的根目录下会保存名为 `archived_config_<RUN_ID>.py` 的配置文件归档，此文件将运行时指定的若干个配置文件融合到一起，形成一个单独的文件，在需要复现此实验时可以直接使用。

### 2.2.4 在不同配置文件之间共享预设值

在一些场景中，一些配置值需要在不同的配置项之间扩展，例如 epochs 可能既被 task module 中的 lr scheduler 需要，又被 runner 需要。为了便于一次性全部修改、防止因遗漏而产生错误，当所有配置组件都在同一个配置文件中时，可以将需要共享的预设值定义为全局变量，但这种方式当配置组件分散在多个文件时不可行，在这种情况下，可以使用 cube-dl 提供的 `shared_config` （可从 `cube_dl.config_sys` 中导入）。在 root config getter 中进行 `set`，然后在其他需要使用时进行 `get`。

### 2.2.5 与其他配置方式的比较

与目前几种主流的配置方式的比较如下：

1. **通过 argparse 定义命令行参数**：一些项目直接使用 argparse 添加可配置的参数，例如 [ViT-pytorch - main.py](https://github.com/jeonsworld/ViT-pytorch/blob/460a162767de1722a014ed2261463dbbc01196b6/train.py#L243)，显而易见，这种配置方式纷繁复杂，当参数数量不断膨胀时，极易出错，在运行时也十分麻烦；
2. **使用 XML/JSON/YAML 等配置文件**：例如 [detectron2](https://github.com/facebookresearch/detectron2) 的部分配置和 PyTorch-Lightning 提供的 [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) 使用了 YAML 文件，这类方法有一个明显的缺陷：IDE 的提示功能十分有限，在编辑时和纯文本文件几乎相同。当配置项繁多时，手写或来回复制粘贴几百行的文本是十分痛苦的，在进行配置时还需要花时间查阅可选择的值，且只能实现简单的逻辑；
3. **使用 OmegaConf 等配置系统库**： [OmegaConf](https://github.com/omry/omegaconf) 是一个基于 YAML 的分层配置系统，支持来自合并多个来源的配置，灵活性很强。但在编写涉及众多参数的深度学习项目时，编辑诸如 YAML 这类文件时，同样要面临编写大量文本文件的麻烦；
4. **实现特定的 Config 类**：例如 [Mask_RCNN - config.py](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py) 中实现了 `Config` 基类，使用时需要派生出子类并按需要覆盖部分属性的值，这种方式灵活性不足，和当前代码紧紧耦合，不适合通用的场景；
5. **一般的 Python 源代码文件**：[OpenMMLab](https://github.com/open-mmlab) 的大部分开源库都采用了这种配置方式，例如 [mmdetection](https://github.com/open-mmlab/mmdetection)，其中的配置文件形如 [atss_r50_fpn_8xb8-amp-lsj-200e_coco.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/atss/atss_r50_fpn_8xb8-amp-lsj-200e_coco.py)。虽然使用了 Python 源代码文件进行配置，但自成体系，有着特殊的规则 (例如需要使用 `_base_` 来继承)，需要付出学习成本，而且本质上是在定义若干个 `dict`，在其中定义要使用的类及其参数，这些参数以 key-value 的形式传入，同样无法充分利用 IDE 的代码提示，与文本型配置方式有着类似的弊端。并且，各种配置项作为变量在配置文件中被直接赋值，是相当松散、容易出错的。

这些配置方式本质上是以各种形式传递参数，然后配置系统将使用这些参数去实例化一些类或传递到某处。而 cube-dl 中的配置方式相当于翻转了此过程，在使用时直接定义如何实例化类，配置系统将会自动记录以及归档。这样，编写配置文件的过程如同正常实例化类一样自然，几乎不需要学习如何配置，而且可以充分利用 IDE 的提示来提高编写效率，还可以加入任意的逻辑。

## 2.3 Starter

所谓的 "starter" 是一组与 cube-dl 兼容的初始文件，用于初始化一个深度学习项目。通过这种方式， cube-dl 可以与具体的框架 (例如 PyTorch-Lightning) 解耦，在创建项目时可以根据实际需要选择灵活性更强的原生 PyTorch，或是抽象程度更高的 PyTorch-Lightning。

标准的 starter 中应该含有名为 "[pyproejct.toml](https://packaging.python.org/en/latest/specifications/pyproject-toml/#)" 的配置文件，并在其中包含名为 `tool.cube_dl` 的配置项。

## 2.4 Starter 的目录结构

Starter 目录结构及其含义如下所示 （以 "pytorch-lightning" 为例）：

```text
pytorch-lightning
├── callbacks 【当前 starter 特定的 CubeCallbacks】
├── configs   【配置文件存放目录】
│   ├── __init__.py
│   ├── components 【配置组件】
│   │   └── mnist_data_module.py
│   └── mnist_cnn_sl.py 【根配置 (Root config) 文件】
├── data    【数据存放目录（符号链接）】
│   └── MNIST -> /Users/yihaozuo/Zyh-Coding-Projects/Datasets/MNIST
├── datasets 【data modules 和数据集类】
│   ├── __init__.py
│   └── basic_data_module.py
├── models   【模型定义】
│   ├── __init__.py
│   ├── __pycache__
│   └── cnn_example.py
├── outputs  【输出目录，存放所有输出产物】
├── pyproject.toml  【项目配置文件】
├── requirements.txt
└── tasks 【Task Modules 的定义】
│   ├── __init__.py
│   ├── base.py  【Task 基类】
│   └── supervised_learning.py【全监督任务定义】
└── utils  【杂项工具】
```

## 2.4 基本命令与参数

### `start`

下载指定的 starter。

可以先通过 `cube start -l` 查看可用的 starters，然后通过以下参数下载指定的 starter:

| 参数名          | 类型 | 是否必需 | 含义 |
| --------------- | :--: | :------: | ---- |
| **-o**, --owner | str  |    ❌     |      |
| **-r**, --repo  | str  |    ❌     |      |
| **-p**, --path  | str  |    ✅     |      |
| **-d**, --dest  | str  |    ❌     |      |

示例：

```shell
cube start -o Alive1024 -r cube-dl -p pytorch-lightning
```

### `new`

创建一对新的 proj 和 exp。

| 参数名                            | 类型 | 是否必需 | 含义             |
| --------------------------------- | :--: | :------: | ---------------- |
| **-pn**, --proj-name, --proj_name | str  |    ✅     | 新建 proj 的名称 |
| **-pd**, --proj-desc, --proj_desc | str  |    ❌     | 新建 proj 的描述 |
| **-en**, --exp-name, --exp_name   | str  |    ✅     | 新建 exp 的名称  |
| **-ed**, --exp-desc, --exp_desc   | str  |    ❌     | 新建 exp 的描述  |

示例：

```shell
cube new -pn "MyFirstProject" -pd "This is my first project." -en "Baseline" -ed "Baseline exps."
```

### `add-exp`

向某个 proj 添加一个新的 exp。

| 参数名                       | 类型 | 是否必需 | 含义                       |
| ---------------------------- | :--: | :------: | -------------------------- |
| **-p**, --proj-id, --proj_id | str  |    ✅     | 新建 exp 所属的 proj 的 ID |
| **-n**, --name               | str  |    ✅     | 新建 exp 的名称            |
| **-d**, --desc               | str  |    ❌     | 新建 exp 的描述            |

示例：

```shell
cube add-exp -p 8q -n "Ablation" -d "Ablation exps."
```

### `ls`

在终端中以表格的形式显示关于 proj, exp 和 run 的信息。

直接使用 `cube ls` 与 `cube ls -pe` 等价，将会显示所有的 proj 和 exp。其他以下参数互斥：

| 参数名                                  |     类型     |                             含义                             |
| --------------------------------------- | :----------: | :----------------------------------------------------------: |
| **-p**, --projs                         | "store_true" |                       显示所有的 proj                        |
| **-er**, --exps-runs-of, --exps_runs_of |     str      |            显示指定的 proj ID 下的所有 exp 和 run            |
| **-e**, --exps-of, --exps_of            |     str      |               显示指定的 proj ID 下所有的 exp                |
| **-r**, --runs-of, --runs_of            |  str (2 个)  | 显示指定的 proj ID 和 exp ID 下所有的 run (proj ID 在前，exp ID 在后) |

示例：

```shell
cube ls -r 8q zy
```

### `fit`, `validate`, `test`, `predict` 共有的参数

`fit`, `validate`, `test`, `predict` 四个子命令都具有下列参数：

| 参数名                               | 类型 | 是否必需 | 含义                       |
| ------------------------------------ | :--: | :------: | -------------------------- |
| **-c**, --config-file, --config_file | str  |    ✅     | 配置文件的路径             |
| **-p**, --proj-id, --proj_id         | str  |    ✅     | 新建 run 所属的 proj 的 ID |
| **-e**, --exp-id, --exp_id           | str  |    ✅     | 新建 run 所属的 exp 的 ID  |
| **-n**, --name                       | str  |    ✅     | 新建 run 的名称            |
| **-d**, --desc                       | str  |    ❌     | 新建 run 的描述            |

### `validate`, `test`, `predict` 共有的参数

除以上参数外，`validate`, `test`, `predict` 三个子命令还具有下列参数：

| 参数名                                | 类型 | 是否必需 | 含义                                               |
| ------------------------------------- | :--: | :------: | -------------------------------------------------- |
| **-lc**, --loaded-ckpt, --loaded_ckpt | str  |    ✅     | 如果不需要加载任何权重，则需要显式指定为空字符串"" |


### `fit`

在训练集上进行训练。

示例：

```shell
cube fit -c configs/mnist_cnn_sl.py -p 8q -e zy -n "ep25-lr1e-3" -d "Use a 3-layer simple CNN as baseline, max_epochs: 25, base lr: 1e-3"
```

### `resume-fit`

从某个中断的训练中恢复。

| 参数名                               | 类型 | 是否必需 | 含义                                                         |
| ------------------------------------ | :--: | :------: | ------------------------------------------------------------ |
| **-c**, --config-file, --config_file | str  |    ✅     | 配置文件的路径                                               |
| **-r**, --resume-from, --resume_from | str  |    ✅     | 要恢复的中断的 fit 的模型 checkpoint 的路径，路径中需要包含 proj、exp 和 run 所在的目录名 (推断 ID 需要这些信息) |

示例：

```shell
cube resume-fit -c configs/mnist_cnn_sl.py -r "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `validate`

在验证集上进行评估。

示例：

```shell
cube validate -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Val" -d "Validate the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `test`

在测试集上进行评估。

示例：

```shell
cube test -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Test" -d "Test the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `predict`

进行预测。

示例：

```shell
cube predict -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Test" -d "Predict using the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

## 2.5 其他

### 2.5.1 回调函数

`RootConfig` 支持通过 `callbacks` 参数添加回调函数，所有回调函数应为 `cube_dl.callback.CubeCallback` 类型。当需要自定义回调函数时，应该继承 `CubeCallback` 类，然后实现所需的钩子。目前 `CubeCallback` 支持 `on_run_start` 和 `on_run_end`。

### 2.5.2 运行时上下文

在运行时，cube-dl 将会将一些上下文存放到特定位置以供访问。

可以从 `cube_dl.core` 中导入 `CUBE_CONTEXT` (实际上是一个 dict)，然后可以通过`run = CUBE_CONTEXT["run"]` 获取到当前的 `Run` 对象。这在需要获取 `Run` 的相关信息时十分有用，例如希望在进行 validation 时将预测结果保存到相应的 run 的目录中时，可以通过 `CUBE_CONTEXT["run"].run_dir` 获取到。

另外，还可以通过访问名为 "CUBE_RUN_ID" 的环境变量获取到当前 `Run` 对象的 ID。
