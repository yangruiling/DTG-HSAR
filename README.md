# DTG-SHAR

## Introduction

本项目基于[MMSkeleton工具包](https://github.com/open-mmlab/mmskeleton)中的[ST-GCN](https://github.com/open-mmlab/mmskeleton/blob/master/doc/START_RECOGNITION.md)模型实现，改进ST-GCN模型的骨架拓扑图构建部分，使用持续学习思想动态构建人体骨架拓扑图. 将具有多关系特性的人体骨架序列数据重新编码为关系三元组, 并基于长短期记忆网络, 通过解耦合的方式学习特征嵌入. 当处理新骨架关系三元组时, 使用部分更新机制
动态构建人体骨架拓扑图, 将拓扑图送入ST-GCN进行动作识别。

## Getting Started

- 运行MMSKeleton工具包参考[GETTING_STARTED.md](./doc/GETTING_STARTED.md)

- 单独使用ST-GCN模型进行人体动作识别参考[START_RECOGNITION.md](./doc/START_RECOGNITION.md)

- 训练基于动态拓扑图的人体骨架动作识别算法
  ``` shell
  cd DTG-SHR
  python ./mmskeleton/fewrel/test_lifelong_model.py
  ```

- 测试基于动态拓扑图的人体骨架动作识别算法
  ``` shell
  cd DTG-SHR
  python ./mmskeleton/fewrel/train_lifelong_model.py
  ```

- 可视化算法运行结果
  基于web server搭建前端  [[参考]](https://blog.csdn.net/gzq0723/article/details/113488110)

  1、前端模块:包含 'static与'templates'文件夹为界面展示相关的代码。 
      
      templates里面包含了两个html的结构文档,用来定义浏览器的显示界面。 
      static里面的css和img用来修饰界面。

  2、服务模块: servel.py里面是web服务的一个业务逻辑。

  运行算法性能可视化web服务
  ``` shell
  cd DTG-SHR
  python ./server.py
  ```

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{
  title={基于动态拓扑图的人体骨架动作识别算法},
  author={解宇 and 杨瑞玲 and 刘公绪 and 李德玉 and 王文剑},
  journal={计算机科学},
  volume={49},
  number={2},
  pages={62--68}
}
```

## Contact
For any question, feel free to contact
```
Ruiling Yang     : yangruiling_xdu@qq.com
```
