# skeleton-ik — 角色 IK 骨骼求解修复

## 题面

角色爬梯子时脚掌穿模、手臂还反向折：CCD 只迭代一次，末端离目标还差 0.3；迭代不看 tolerance，固定跑满次数；约束在求解后应用，关节被拉回后末端又偏离；扭转限制用点积，超过 180 度判反；stretch_policy=none 仍按距离缩放骨骼；权重混合不归一，weight=0.5 时整体缩短；链组装不检测环，父子互指时无限递归；正向运动学用世界坐标累加，骨骼缩放下结果偏大；批量求解共用可变状态，角色之间互相污染；批量遇无解 rig 只跳过，输出里留下未收敛姿态。anim/ik.py 是核心，tests/test_ik.py 15个红。要修的是：按 anim/ik.py 的 docstring 契约实现 8 个桩函数，让 CCD 在 1e-6 容差内收敛、约束在迭代内生效、批量求解全有或全无，并且不改动 tests/test_ik.py。

## 运行方式

```bash
python -m pytest tests -q
```

初始骨架状态下输出恰好为：

```text
15 failed
```

8 个公开函数目前全部 `raise NotImplementedError`，不产生 collection / import / syntax error。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/test_ik.py` 不可修改（15 条测试是验收标准）。
- 只允许修改 `anim/ik.py`（必要时可在 `anim/` 内增加私有辅助代码）。
- 仅使用标准库 `math`、`dataclasses` 与 `pytest`，离线、无随机。
