# 带人口权重的洛伦兹曲线与组间交叉分解

本项目保存虚构数据共享及产品运营的合成表格与Python读取工具。当前工具只解析文件并显示记录数，不执行统计推断、优化、估计或校准。所有地区、机构、产品与人数均为合成。

## 字段与单位

x是非负人均资源，weight是正人口权，group是预定分组。weight_kind=representation时不能直接给复杂设计置信区间；手算案例单独选取h1/h2。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 运行已有读取器

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.inequality_reader fixtures/distribution.csv
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 固定容器

    docker build -t weighted-inequality:local .
    docker run --rm weighted-inequality:local

容器镜像以完整digest固定，默认仅检查输入契约；实现新增分析功能所需依赖由开发者明确登记。
