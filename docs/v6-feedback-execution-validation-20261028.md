# 生成策略的真实执行与失败反馈回归

承接[成功反馈角色对照](v6-accepted-feedback-replay-20261028.md)：两臂均正确理解成功并生成预取，未证明提示因果优势；CPU toy probe不能证明完整双卡质量。生产core、prompt、调参和接受规则保持不变。

新目录`/root/autodl-tmp/dev/v6-work/feedback-execution-validation-20261028`，控制器`scripts/experiments/v6_feedback_execution_validation_20261028.py`。两个部分串行执行，结果分开解读。

第一部分复用原始继承revision799a80d2及只读source controls，精确取两臂生成文件SHA。开发者固定共同合法参数K=3，顺序为继承K3、control D0、control D1、treatment D1、treatment D0。D0检查关闭新增预取时的原策略，D1检查实际新增路径。保留原B1/S128、两rank、完整GPT2Medium、FP32/eager/noTF32、两步loss/logits/梯度/更新参数及实际DTensor shard映射、返回值与512B source owner、完整同步及peak<=1.05原baseline。每端点全部5个clean样本和各诊断指标独立3次profile保留；这些时间仅作诊断，不运行配对接受或自动发布，不重测到通过。

三份单独nsys记录检查继承版本及两个D1的原生执行。只有对应质量/覆盖通过才执行该路径采集；unavailable、质量失败和执行错误明确记录。route标签结合源码和native证据判断，不能单凭标签或静态API存在推断优化实际执行。NCCL sum、不同profile窗口及逻辑payload不解释成wire time或暴露关键路径。所有固定参数revision保持分支内staged，不改变接受指针。

第二部分从历史`fsdp-search-20261028`第三轮candidate初始快照generation-c519752a5f2d做一个新真实会话。当时前轮为连续第二步logits质量失败；输入、模型、工具及core原样，仅替换同一中性提示句。不给它旧第三轮答案、当前审计结论或新证据。核对是否仍识别真实失败、解释具体改动与保持原语义。此输出只做生成/schema回归，不在本批执行其GPU性能验证；不自动采用生产提示。

启动前CPU检查已确认两个D0/D1点均在各自合法空间内、生成父版本一致、原约束计算接口在冻结baseline通过、历史失败快照为fresh session且反馈确实rejected，见[预检查](evidence/v6-accepted-feedback-replay-20261028/execution-validation-preflight.json)。正式批次重新核对六份完整source/reference SHA，避免复制大张量。磁盘当前约12GiB，预计新增主要为五端点独立profile，按实际df观察；raw留服务器，没有本地大转存。只启用一个普通完成/8GiB磁盘监听，无Goal、模型轮询或投递测试。
