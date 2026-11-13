# 带人口权重的洛伦兹曲线与组间交叉分解

本项目保存虚构数据共享及产品运营的合成表格与Python读取/分析工具。读取器只解析文件；
分析器计算加权经验分布的洛伦兹曲线、Gini 系数、人口分位资源份额与组内/跨组成对分解。
所有地区、机构、产品与人数均为合成，输出只描述分布不平等形态，**不估计资源配置的因果效果**。

## 字段与单位

x是非负人均资源，weight是正人口权，group是预定分组。权重表示“代表人口”，大小地区
不得各当一票：W=sum(w)、mu=sum(w*x)/W，
Gini=sum_{i,j} w_i*w_j*|x_i-x_j| / (2*W^2*mu)。weight_kind=representation时不能直接
给复杂设计置信区间；手算案例单独选取h1/h2。

- mu=0（资源全零）时按配置 `zero_mean=undefined` 输出 Gini=null 并单列，不报 0；
- 不接受负值，也不做未经告知的平移；空白字段拒绝隐式填补为 0；
- 分位人口组在边界处按权重拆分记录片段，不允许整条记录跨过人口阈值，证据见
  `.workspace-output/quantile_split_evidence.json`；
- 分组分解按成对项划分为同组贡献与跨组贡献，全部按全体共同分母归一，分量之和恰为总
  Gini；跨组项含分布重叠，不用组均值差替换。另给“仅保留组均值”的反事实 Gini，二者在
  组间交错时不相等，输出中分别标注；
- 仅当配置声明 `weight_kind=frequency` 且 `bootstrap_enabled=true` 时才按频数做多项抽样
  重估；representation 权重返回 not_applicable，不伪造置信区间。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出放
.workspace-output（已gitignore），原始资料不应被覆盖。

## 运行读取器与分析器

Python 3.11以上，标准库即可，不需安装第三方依赖。隔离模式（-I）下请用根目录入口脚本，
它们会先把仓库根显式加入 sys.path：

    python -m src.inequality_reader fixtures/distribution.csv
    python -I run_tests.py                                  # 全部单元测试
    python -I run_analyze.py fixtures/distribution.csv \
        fixtures/inequality.json --out .workspace-output    # 生成分析与证据文件

证据文件：`inequality_result.json`（全量结果）、`quantile_split_evidence.json`（人口分位
切分片段）、`pairwise_contributions.json`（记录数不超过20时物化全部成对项）。

## 算法与复杂度

- 小表（n<=20）用显式 O(n^2) 有序双重求和核对定义式并保存成对贡献；大表用排序后线性
  累积 U=sum_j w_j*(x_j*W_<j - S_<j)，D=2U，整体 O(n log n)，同值项自动贡献 0；
- 分组分解在同一次排序遍历中为每组维护 W_g、S_g 累加器（额外 O(组数) 空间），
  跨组 D 由总 D 减同组 D 得到，不显式平方展开；
- 另用洛伦兹折线梯形面积独立核对 Gini，三种算法在测试中相互校验。

## 固定容器

    docker build -t weighted-inequality:local .
    docker run --rm weighted-inequality:local

容器镜像以完整digest固定，默认仅检查输入契约与全部单元测试；实现新增分析功能所需依赖
由开发者明确登记（当前仍仅用标准库）。
