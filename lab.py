import os
import os.path as osp
import inspect


if __name__ == '__main__':
    # from configs.exp_on_oracle_mnist import get_root_config_instance
    # from entities import Run
    #
    # config_instance = get_root_config_instance()
    # config_instance.setup_wrappers()
    # config_instance.setup_trainer("csv", Run(name="dev", desc="123", proj_id=1, exp_id=1,
    #                                          job_type="fit",
    #                                          output_dir="/Users/yihaozuo/Zyh-Coding-Projects/Personal/DL-Template"
    #                                                     "-Project/outputs"))

    print(os.getcwd())
    print(__file__)
    print(osp.split(__file__)[0])
