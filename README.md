# cube-dl

Languages: English | [简体中文](./docs/README_zh-CN.md)

**"The last stop" for training your deep learning models.**

**Manage tons of configurations and experiments with minimal changes to existing code.**


[![Packaging Wheel](https://github.com/Alive1024/cube-dl/actions/workflows/packaging_wheel_on_push.yml/badge.svg)](https://github.com/Alive1024/cube-dl/actions/workflows/packaging_wheel_on_push.yml)
[![Publishing to PyPI](https://github.com/Alive1024/cube-dl/actions/workflows/publishing_on_tag.yml/badge.svg)](https://github.com/Alive1024/cube-dl/actions/workflows/publishing_on_tag.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**Install from PyPI (stable, recommended)**：

```shell
pip install -U cube-dl
```

**Install from wheel file (latest)**:

Enter the [Actions](https://github.com/Alive1024/cube-dl/actions) page of this project, select the latest workflow run from the actions corresponding to "Packaging Wheel", download the compressed package of the wheel file from Artifacts, extract it, and install it using pip:

```shell
pip install xxx.whl
```

**Install from source code (latest)**：

```shell
git clone git@github.com:Alive1024/cube-dl.git
cd cube-dl
pip install .
```

**Table of Contents**：

- [cube-dl](#cube-dl)
- [1. Introduction](#1-introduction)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Main Features](#12-main-features)
  - [1.3 Design Principles](#13-design-principles)
  - [1.4 Prerequisites](#14-prerequisites)
- [2. Project Description](#2-project-description)
  - [2.1 Key Concepts](#21-key-concepts)
    - [2.1.1 Four Core Components](#211-four-core-components)
    - [2.1.2 The Triple-Layer Structure for Organizing Experiments](#212-the-triple-layer-structure-for-organizing-experiments)
  - [2.2 Configuration System](#22-configuration-system)
    - [2.2.1 Configuration Files](#221-configuration-files)
    - [2.2.3 Automatic Archiving of Configuration Files](#223-automatic-archiving-of-configuration-files)
    - [2.2.4 Sharing Preset Values Between Configuration Files](#224-sharing-preset-values-between-configuration-files)
    - [2.2.5 Comparison with Other Configuration Methods](#225-comparison-with-other-configuration-methods)
  - [2.3 Starter](#23-starter)
  - [2.4 Directory Structure of The Starter](#24-directory-structure-of-the-starter)
  - [2.4 Basic Commands and Arguments](#24-basic-commands-and-arguments)
    - [`start`](#start)
    - [`new`](#new)
    - [`add-exp`](#add-exp)
    - [`ls`](#ls)
    - [Common Arguments for `fit`, `validate`, `test`, `predict`](#common-arguments-for-fit-validate-test-predict)
    - [Common Arguments for  `validate`, `test`, `predict`](#common-arguments-for--validate-test-predict)
    - [`fit`](#fit)
    - [`resume-fit`](#resume-fit)
    - [`validate`](#validate)
    - [`test`](#test)
    - [`predict`](#predict)
  - [2.5 Others](#25-others)
    - [2.5.1 Callback Functions](#251-callback-functions)
    - [2.5.2 Runtime Contexts](#252-runtime-contexts)

# 1. Introduction

**cube-dl** is a high-level Python library for managing and training deep learning models, designed to manage a large number of deep learning configuration items and experiments with ease, making it well-organized.

## 1.1 Motivation

As we can see, there are already quite a few libraries related to deep learning at different levels in the open source community. For example, [PyTorch](https://github.com/pytorch/pytorch) provides powerful deep learning modeling capabilities, while [PyTorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning) abstracts and wraps PyTorch, saving the hassle of writing a large amount of boilerplate code. However, even with these, training deep learning models may still be chaotic due to a large number of configurable items, experiments, etc., forcing researchers/developers to spend a lot of energy and time organizing and comparing experimental results, rather than the methods themselves. In addition, in the process of conducting research, it is inevitable to use other people's open-source algorithms. Due to each person's different code habits, open-source algorithms have different organizational structures, and some repositories serve specific methods or datasets without good top-level design. Using these codes for custom experiments can be quite painful. Furthermore, when aggregating algorithms from different sources, a universal code structure is required.

**cube-dl** was born as a result, by imposing some rule constraints on configuration and experimental management to make deep learning projects easier to manage, and striking a good balance between abstraction and flexibility.

## 1.2 Main Features

- **Componentization**: The elements involved in the training process of deep learning models are clearly divided into four parts to achieve low coupling and high reusability;
- **A brand-new configuration system**: Deep learning projects often involve a large number of configurable parameters, and how to effortlessly configure these parameters is an important issue. Moreover, these parameters often have a crucial impact on the final result, so it is necessary to document these parameters in detail. Cube-dl has redesigned the entire configuration system based on the characteristics of deep learning projects, making it easy to use and traceable;
- **Triple organizational structure**: In order to organize a large number of experiments in a more organized manner, all experiments are forcibly divided into three levels: Project, Experiment, and Run. Each task execution will automatically save corresponding records for reference;
- **Simple and Fast CLI**: cube-dl provides a concise CLI that can be managed, trained, and tested with a few commands.


## 1.3 Design Principles

cube-dl follows the following principles as much as possible:

- **Universality**: Independent of specific research fields, there is no need to start from scratch when switching between different fields;
- **Flexibility and Scalability**: "Extend rather than modify". When implementing new models, datasets, optimization algorithms, loss functions, metrics, and other components, try not to change existing code as much as possible. Instead, add new code to achieve extension;
- **Good Organization and Recording**: The results of each operation should be well organized and recorded;
- **Maximum Compatibility**: facilitates the migration of existing other code to the current code repository at the lowest cost;
- **Lowest Learning Cost**: After reading README, you can master how to use it without having to learn a lot of APIs from dozens of pages of documentation.

## 1.4 Prerequisites

Users should have a basic understanding of Python and PyTorch.


# 2. Project Description

## 2.1 Key Concepts

### 2.1.1 Four Core Components

Generally speaking, the core components of deep learning include [<sup>1</sup>](https://d2l.ai/chapter_introduction/index.html#key-components)：

- **Data** that can be learned
- The **model** for converting data
- **Objective function** for quantifying model effectiveness
- **Optimization algorithm** for adjusting model parameters to optimize the objective function

Based on the above classification and componentization ideas, cube-dl reorganizes the relevant components of deep learning projects into four parts:

- **Model**: the model to be trained；
- **Task Module**: The definition of the process for a certain deep learning task, corresponding to a certain training paradigm, such as the most common fully supervised learning. The Task Module can be further subdivided into several components, such as loss functions, optimization algorithms, learning rate regulators, metrics used in validation and testing, etc. Meanwhile, the model to be trained is specified as an initialization parameter for the Task Module;
- **Data Module**: Data related, corresponding to the combination of Dataset and DataLoader for PyTorch, similar to the LightningDataModule for [PyTorch-Lightning](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). However, the usage is slightly different. The Data Module here is not specific to any dataset, and the specific dataset class is passed in as an initialization parameter for the Data Module;
- **Runner**: The engineering level code for executing model training, validation, testing, reasoning, and other processes.

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



### 2.1.2 The Triple-Layer Structure for Organizing Experiments

In order to organize all experiments in a more organized manner, cube-dl mandatorily requires users to use a "triple-layer structure":

- **Project** (hereinafter referred to as **proj**): contains multiple exps;
- **Experiment** (hereinafter referred to as **exp**): a set of runs with a common theme, each exp must be associated with a certain proj, such as "baseline", "abbreviation", "contrast", etc.;
- **Run**: The smallest atomic unit of operation, each run must belong to an exp in a proj, and each run has a job type indicating what the run is doing.

The above three entities all have corresponding random IDs composed of lowercase letters and numbers. ID of proj/exp is of 2 characters, and the ID of run is of 4 characters.


The structure of the output directory will take the form of:

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

In the root directory of proj, there will be a JSON file with the same name, which contains records of all exps and runs of the current proj, such as:

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

By default, these proj record files will be tracked by git to facilitate distributed collaboration among multiple people through git. This means that the proj, exp, and run created by user A can be seen by user B (but the output products of run will not be tracked by git).


## 2.2 Configuration System

As mentioned earlier, deep learning projects often involve a large number of configurable parameters, and it is crucial to pass in and record these parameters. Considering that the essence of configuration is to provide initialization parameters for instantiating classes, cube-dl has designed a brand-new configuration system. Writing configuration files is as natural as writing code for instantiating a class normally.

### 2.2.1 Configuration Files

In cube-dl, the configuration file is actually a `.py` source code file, mainly used to define how to instantiate the corresponding object. Writing a configuration file is a process of selecting (`import` what to be used) and defining how to instantiate it. For example, the following is a code snippet for configuring a runner:

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

As can be seen, in the configuration file, the instantiation process needs to be put into a "getter" function, and the instantiated object will be `return`. The reason why an object is not directly instantiated in the configuration file is to allow the cube-dl to control the timing of instantiation of configuration items.

Since the configuration file is essentially a Python source code file, it can contain any logic like a regular Python source code file, but it is generally not very complex.

Corresponding to the four core components described earlier, there are four main types of core configuration items, namely `cube_model`, `cube_task_module`, `cube_data_module` and `cube_runner`. These configuration items can be used as modular and reusable configuration components in the configuration system. In addition, during actual experiments, it is necessary to freely combine the four components to form a **RootConfig**, which is the root node of all configurations.

The relationship between the five configuration items is as follows:

```text
                                  ┌───────Components───────┐
                                  │     ┌────────────┐     │
                                  │ ┌──▶│  Model(s)  │     │
                                  │ │   └────────────┘     │
                                  │ │   ┌────────────┐     │
                  ┌─────────────┐ │ ├──▶│Task Module │     │
                  │ Root Config │─┼─┤   └────────────┘     │
                  └─────────────┘ │ │   ┌────────────┐     │
                                  │ ├──▶│Data Module │     │
                                  │ │   └────────────┘     │
                                  │ │   ┌────────────┐     │
                                  │ └──▶│ Runner(s)  │     │
                                  │     └────────────┘     │
                                  └────────────────────────┘
```

For some rules regarding configuration files:

- For better readability, keyword parameters must be used when initializing `RootConfig` in the configuration file (it is recommended to force the use of keyword parameters when writing task/data modules, following this rule);
- The getter function name of Root config must be `get_root_config`, and there can only be one in each configuration file. Other types of configuration items do not have this restriction;
- Decorators named  `cube_root_config`, `cube_model`, `cube_task_module`, `cube_data_module` and `cube_runner` can be imported into `cube_dl.config_sys`. It is strongly recommended to use the corresponding decorators when writing getter functions, on the one hand to allow the decorators to check, and on the other hand to expand in the future.

Additionally, it is recommended to use relative import statements when importing the required config components from other configuration files.


### 2.2.3 Automatic Archiving of Configuration Files

For the convenience of replicating a run, the configuration files used during each run will be automatically archived. By default, a configuration file named 'archived_config_<RUN_ID>. py' will be saved in the root directory of the corresponding run. This file combines several configuration files specified at runtime to form a separate file, which can be used directly when replicating this experiment.

### 2.2.4 Sharing Preset Values Between Configuration Files

In some scenarios, some configuration values need to be shared between different configuration files. For example, epochs may be required by both the LR scheduler in the task module and the runner. In order to facilitate one-time modification and prevent errors caused by omissions, when all configuration components are in the same configuration file, the preset values that need to be shared can be defined as global variables. However, this approach is not feasible when configuration components are scattered across multiple files. In this case, the `shared_config` provided by cube-dl can be used (which can be imported from `cube_dl.config_sys`). Perform `set` in the root config getter, and then perform `get` when needed for other purposes.

### 2.2.5 Comparison with Other Configuration Methods

The comparison with several mainstream configuration methods is as follows:

1. **Defining Command Line Arguments by argparse**: some projects directly defines configurable arguments by `argparse`. It is obvious that this configuration method is complex and prone to errors when the number of parameters continues to expand, and it is also very troublesome at runtime;
2. **Using XML/JSON/YAML or Other Configuration Files**：for example, some configurations of [detectron2](https://github.com/facebookresearch/detectron2) and [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) provided by PyTorch-Lightning adopt YAML files. This method has an obvious flaw: the prompt function from the IDEs will be very limited, and it is almost identical to a plain text file during editing. When there are many configuration items, handwriting or copying and pasting hundreds of lines of text back and forth can be very painful. When configuring, you also need to spend time looking up optional values and can only achieve simple logic;
3. **Using OmegaConf or Other Configuration Library**： [OmegaConf](https://github.com/omry/omegaconf) is a YAML-based hierarchical configuration system, supporting configuration from merging multiple sources, with strong flexibility. But when writing deep learning projects involving numerous parameters, editing files like YAML also faces the hassle of writing a large number of text files;
4. **Implementing Specific Config Classes**：for example, [Mask_RCNN - config.py](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py) implements `Config` base class. When using it, subclasses need to be derived and some attribute values need to be covered as needed. This approach is inflexible and tightly coupled with the current project, making it unsuitable for general scenarios;
5. **General Python Source Files**：most of open source libraries from [OpenMMLab](https://github.com/open-mmlab) adopt this method such as [mmdetection](https://github.com/open-mmlab/mmdetection. Their configration files are like [atss_r50_fpn_8xb8-amp-lsj-200e_coco.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/atss/atss_r50_fpn_8xb8-amp-lsj-200e_coco.py). Although Python source code files are used for configuration, it is self-contained and has special rules (such as the need to use `_base_` for inheritance), which incurs learning costs. Essentially, it involves defining several `dict`s, defining the classes to be used and their parameters, which are passed in as key values and cannot fully utilize the code prompts of the IDE. This has similar drawbacks to text-based configuration methods. Moreover, the direct assignment of various configuration items as variables in the configuration file is quite loose and error-prone.

These configuration methods essentially pass parameters in various forms, and then the configuration system will use these parameters to instantiate some classes or pass them to a certain location. The configuration method in cube-dl is equivalent to flipping this process, directly defining how to instantiate the class during use, and the configuration system will automatically record and archive it. In this way, the process of writing configuration files is as natural as instantiating classes normally, with almost no need to learn how to configure them. It can also fully utilize the prompts of the IDE to improve writing efficiency and add any logic.

## 2.3 Starter

The so-called "starter" is a set of initial files compatible with cube-dl, used to initialize a deep learning project. Through this approach, cube-dl can be decoupled from specific frameworks such as PyTorch-Lightning. When creating a project, you can choose more flexible native PyTorch or more abstract PyTorch-Lightning based on actual needs.

The standard starter should contain a file named "[pyproejct.toml](https://packaging.python.org/en/latest/specifications/pyproject-toml/#). And it should contain a configuration item named `tool.cube_dl`.

## 2.4 Directory Structure of The Starter

The structure and meaning of the starter directory are as follows (using "pytorch-lighting" as an example):

```text
pytorch-lightning
├── callbacks 【specific CubeCallbacks for the current starter】
├── configs   【directory for storing configuration files】
│   ├── __init__.py
│   ├── components 【configuration components】
│   │   └── mnist_data_module.py
│   └── mnist_cnn_sl.py 【a RootConfig file】
├── data    【directory for storing（symbolic links）】
│   └── MNIST -> /Users/yihaozuo/Zyh-Coding-Projects/Datasets/MNIST
├── datasets 【data modules and dataset classes】
│   ├── __init__.py
│   └── basic_data_module.py
├── models   【model definition】
│   ├── __init__.py
│   ├── __pycache__
│   └── cnn_example.py
├── outputs  【the output director, string all output products】
├── pyproject.toml  【configuration file for Python project】
├── requirements.txt
└── tasks 【definitions of Task Modules】
│   ├── __init__.py
│   ├── base.py  【task base class】
│   └── supervised_learning.py【task definition of full supervised learning】
└── utils  【miscellaneous utilities】
```

## 2.4 Basic Commands and Arguments

### `start`

Download the specified starter.

You can first view the available starters through `cube start -l`, and then download the specified starter using the following arguments:

| Argument Name   | Type | Required | Meaning |
|-----------------|:----:|:---------:|---------|
| **-o**, --owner | str  |     ❌     |         |
| **-r**, --repo  | str  |     ❌     |         |
| **-p**, --path  | str  |     ✅     |         |
| **-d**, --dest  | str  |     ❌     |         |

For example：

```shell
cube start -o Alive1024 -r cube-dl -p pytorch-lightning
```

### `new`

Create a pair of new proj and exp.

| Argument Name                     | Type | Required | Meaning                     |
|-----------------------------------| :--: | :------: |-----------------------------|
| **-pn**, --proj-name, --proj_name | str  |    ✅     | name of the new proj        |
| **-pd**, --proj-desc, --proj_desc | str  |    ❌     | description of the new proj |
| **-en**, --exp-name, --exp_name   | str  |    ✅     | name of the new exp         |
| **-ed**, --exp-desc, --exp_desc   | str  |    ❌     | description of the new exp  |

For example：

```shell
cube new -pn "MyFirstProject" -pd "This is my first project." -en "Baseline" -ed "Baseline exps."
```

### `add-exp`

Add a new exp to a proj.

| Argument Name                | Type | Required | Meaning                                    |
| ---------------------------- |:----:| :------: |--------------------------------------------|
| **-p**, --proj-id, --proj_id | str  |    ✅     | ID of the proj that the new exp belongs to |
| **-n**, --name               | str  |    ✅     | name of the new exp                        |
| **-d**, --desc               | str  |    ❌     | description of the new exp                 |

For example：

```shell
cube add-exp -p 8q -n "Ablation" -d "Ablation exps."
```

### `ls`

Display information about proj, exp, and run in the form of a table in the terminal.

`cube ls` is equivalent to `cube ls -pe`, which will display all proj and exp.


The following other parameters are mutually exclusive:

| Argument Name                           |     Type     |                                    Meaning                                     |
|-----------------------------------------|:------------:|:------------------------------------------------------------------------------:|
| **-p**, --projs                         | "store_true" |                               display all projs                                |
| **-er**, --exps-runs-of, --exps_runs_of |     str      |             display all exps and runs of the proj specified by ID              |
| **-e**, --exps-of, --exps_of            |     str      |                  display all exps of the proj specified by ID                  |
| **-r**, --runs-of, --runs_of            |  str (2 个)   | display all runs of the exp of the proj specified by two IDs (proj_ID exp_ID). |

For example：

```shell
cube ls -r 8q zy
```

### Common Arguments for `fit`, `validate`, `test`, `predict`

`fit`, `validate`, `test`, `predict` all have the following arguments:

| Argument Name                               | Type | Required | Meaning                                        |
| ------------------------------------ |:----:|:--------:|------------------------------------------------|
| **-c**, --config-file, --config_file | str  |    ✅     | path to the config file                        |
| **-p**, --proj-id, --proj_id         | str  |    ✅     | ID of the proj ID that the new run belongs to. |
| **-e**, --exp-id, --exp_id           | str  |    ✅     | ID of the exp that the new run belongs to.     |
| **-n**, --name                       | str  |    ✅     | name of the new run                            |
| **-d**, --desc                       | str  |    ❌     | description of the new run                                   |

### Common Arguments for  `validate`, `test`, `predict`

In addition to the above parameters, subcommands `validate`, `test`, `predict` also have the following arguments:

| Argument Name                         | Type | Required | Meaning                                               |
| ------------------------------------- | :--: | :------: | -------------------------------------------------- |
| **-lc**, --loaded-ckpt, --loaded_ckpt | str  |    ✅     | File path of the model checkpoint to be loaded. Use an empty string "" to explicitly indicate you are going to conduct validate/test/predict using the initialized model without loading any weights). |


### `fit`

Training on the training set.

For example：

```shell
cube fit -c configs/mnist_cnn_sl.py -p 8q -e zy -n "ep25-lr1e-3" -d "Use a 3-layer simple CNN as baseline, max_epochs: 25, base lr: 1e-3"
```

### `resume-fit`

Resume fit from an interrupted one.

| Argument Name                        | Type | Required | Meaning                                                                                                                                                                |
| ------------------------------------ | :--: | :------: |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **-c**, --config-file, --config_file | str  |    ✅     | path to the config file                                                                                                                                                |
| **-r**, --resume-from, --resume_from | str  |    ✅     | file path to the checkpoint where resumes, the path should include the directory names where proj, exp, and run are located (inferring IDs requires these information) |

For example：

```shell
cube resume-fit -c configs/mnist_cnn_sl.py -r "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `validate`

Evaluate on the validation set.

For example：

```shell
cube validate -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Val" -d "Validate the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `test`

Evaluate on the test set.

For example：

```shell
cube test -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Test" -d "Test the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

### `predict`

Predict.

For example：

```shell
cube predict -c configs/mnist_cnn_sl.py -p 8q -e zy -n "Test" -d "Predict using the simple CNN." -lc "outputs/proj_8q_MNIST/exp_zy_Baseline/run_rw4q_fit_ep25-lr1e-3/checkpoints/epoch\=3-step\=1532.ckpt"
```

## 2.5 Others

### 2.5.1 Callback Functions

`RootConfig` supports adding callback functions through the `callbacks` parameter. All callback functions should be `cube_dl.callback.CubeCallback` type. When custom callback functions are needed, they should inherit the `CubeCallback` class and implement the required hooks. Currently, `CubeCallback` supports `on_run_start` and `on_run_end`.

### 2.5.2 Runtime Contexts

At runtime, the cube-dl will store some context in specific locations for access.

You can import `CUBE_CONTEXT` (actually a dict) from `cube_dl.core`, and then retrieve the current `Run` object through `run = CUBE_CONTEXT["run"]`. This is very useful when obtaining information related to `Run`. For example, when you want to save the predicted results to the corresponding run directory during validation, you can obtain it through `CUBE_CONTEXT["run"].run_dir`.

In addition, the ID of the current `Run` object can also be obtained by accessing the environment variable named `CUBE_RUN_ID`.
