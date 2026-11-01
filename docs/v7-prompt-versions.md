# v7 提示词版本与一次有界验证

2026-11-01用户授权提示词改进，并要求独立branch/worktree、保留旧文件用于对比和回退。v7从v6提交`42d6d093`建立；本地`D:/Pyhon_projects/opop/v7`，服务器使用独立`/root/autodl-tmp/dev/opop-v7`。v6工作目录和历史提示词不修改。

现有四个角色的`*_prompt.md`保留为v6，默认配置仍使用这些文件。新candidate全文保存在`src/kernel_optimizer/project/prompts/candidate/v7.md`，只增加一段通用实际路径核对要求：依据真实模块属性、参数形状、状态转换判断守卫；CPU检查使用提交文件；区分复制helper、直接调用、真实绑定与GPU验证。没有任务路径、模型名或预期答案，没有新schema、模型设置或接受规则。

通过现有配置覆盖选择，无需换文件：

```sh
kernel-opt --config private.yaml --override project_defaults.prompt_versions.candidate=v7 optimize-project ...
kernel-opt --config private.yaml --override project_defaults.prompt_versions.candidate=v6 optimize-project ...
```

也可在YAML中设置`project_defaults.prompt_versions: {candidate: v7}`。未设置的角色保持v6；不存在的版本报错，不静默降级。普通生成及恢复selection后的新candidate均使用所选版本；已存在的会话保留原会话历史。现有generation snapshot保存实际完整提示词和输入，继续作为对比依据。新增版本使用新文件，已用于实验的版本不得覆盖更新。

主线仍是C1资源证据→C2响应引导改写→C3原任务完整验收。本次只执行一次候选生成对照：重用已保存的候选初始快照、原反馈和选择，换用candidate v7全文，其他角色/模型/工具/输入保持。旧结果作为历史对照，不重启已关闭的两轮任务，不注入审计答案、不手改优化器。一次生成不能隔离随机性或证明跨项目收益；是否执行声明改写及原完整质量/配对性能须据实际结果判断。若有可执行候选，再单独声明一次原评测；不修到通过、不扩成第三轮搜索。

清理已释放约21.81GiB；原始配置、快照及角色历史仍只在服务器。分支、提示词版本管理服务于此次验证，不成为独立的辅助工程终点。
