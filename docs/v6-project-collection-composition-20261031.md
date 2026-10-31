# 普通采集规划的父子 bundle 事实补齐

[本次正式验证](v6-ddp-pipeline-next-validation-20261031.md)中，普通规划器完整读过revisions输入，却把替换旧bundle的候选描述成追加，并选择仅适用于追加的binding_cost。现有执行器正确返回not_applicable，没有产生额外GPU作业；这不改变候选的性能拒绝。

输入之前只包含candidate revision、parent ID及candidate bundles，缺少判定所需的父版本构成。现在[通用数据组装](../src/kernel_optimizer/project/diagnostic_planning.py)根据每个候选的真实直接父版本输出`parent_bundles`及`bundle_changes.added/removed/retained`，由[普通plan_collectors](../src/kernel_optimizer/project/search.py)传入既有`revisions.json`。不拿original reference或discovery代替实际父版本；无父版本时明确为null，不猜空父构成。只给结构事实，collector仍由原角色决定。

没有模型名、实验路径、哈希、GPU UUID或预期答案分支；不修改生产prompt、模型、sampler、collector实现、候选算法、测量或接受规则，也不增加额外版本门槛。旧实验只读runtime和历史输入/输出保持。后续角色是否正确使用这些事实尚未运行验证，不预称规划成功或GPU收益。

两份修改源码及测试文件已同步服务器工作仓库，与通过测试的字节相同；[工作代码记录](evidence/v6-revision-composition-20261031/change-result.json)的新core为`e893f850dff70908e6100b7f6cc0773fbeac03e201759d688eda4d3682b3d0d8`。已完成实验仍使用原只读cb34环境/core ed24，未把新代码回写其运行目录或历史结论。

[相关测试](../tests/test_v6_diagnostic_planning.py)覆盖空父追加、保留父追加、部分替换、同bundle、无父版本，并通过普通入口验证不会混用reference/discovery。该文件共12项测试在现有服务器依赖环境通过（0.76s），没有Agent/API、模型、GPU或真实collective；只传两份修改源码和一份测试文件。首次本地Anaconda缺少optuna，7个导入失败/5通过，未安装依赖或为此扩展环境建设，原失败已记于[测试记录](evidence/v6-revision-composition-20261031/test-result.json)。

这是正式拒绝后已经执行完成的一步通用框架改进，不是分析器项目或再次优化修复链。e610/fine9/coarse5不重试，已接受02b933保持；当前没有另排GPU/selector/候选/评审。项目并未宣告完成，后续优化需从现有真实收益和失败反馈提出新的可检验路线，不能通过重复同一候选获得通过。
