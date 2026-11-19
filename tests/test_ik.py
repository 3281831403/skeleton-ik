"""skeleton-ik 的验收测试：15 条，全部针对 anim/ik.py 的公开契约。

初始骨架状态下 8 个函数均为 ``NotImplementedError`` 桩，因此 15 条测试全红。
本文件是题目的一部分，评测时不可修改。
"""

import copy
import math

import pytest

from anim.ik import (
    IkError,
    apply_constraints,
    chain_from_plan,
    forward_kinematics,
    limit_twist,
    solve_batch,
    solve_ccd,
    stretch_policy,
    weight_blend,
)

TOL = 1e-6


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _joints(chain, directions):
    """由骨骼链与每节单位朝向重建关节世界坐标（含根关节）。"""
    point = list(chain[0]["head"])
    joints = [point]
    for bone, direction in zip(chain, directions):
        length = _dist(bone["head"], bone["tail"])
        point = [point[i] + length * direction[i] for i in range(3)]
        joints.append(point)
    return joints


def _end(chain, directions):
    return _joints(chain, directions)[-1]


# 缺陷 1：CCD 只迭代一次，末端离目标还差 0.3。
def test_ccd_converges_beyond_a_single_iteration():
    chain = [
        {"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}},
        {"head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
        {"head": [0.0, 0.0, 2.0], "tail": [0.0, 0.0, 3.0], "constraints": {}},
    ]
    target = [0.6, 0.0, 2.5]
    # 初始末端在 [0, 0, 3]，单轮之后仍会留下可见误差。
    assert _dist([0.0, 0.0, 3.0], target) > 0.3

    directions = solve_ccd(chain, target, 4000, TOL)

    assert len(directions) == 3
    for direction in directions:
        assert abs(_norm(direction) - 1.0) <= 1e-9
    assert _dist(_end(chain, directions), target) <= TOL + 1e-9


# 缺陷 2：迭代不看 tolerance，固定跑满次数。
def test_ccd_checks_tolerance_instead_of_running_fixed_iterations():
    chain = [
        {"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}},
        {"head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
    ]
    # 目标远超可达范围，tolerance 永远无法满足：必须在耗尽迭代后报错，
    # 而不是不加判断地把固定轮数的结果返回。
    with pytest.raises(IkError):
        solve_ccd(chain, [0.0, 0.0, 5.0], 5, TOL)


# 缺陷 3：约束在求解后应用，关节被拉回后末端又偏离。
def test_constraints_are_enforced_during_solve_not_after():
    chain = [
        {
            "head": [0.0, 0.0, 0.0],
            "tail": [0.0, 0.0, 1.0],
            "constraints": {"x": [-9.0, -0.1]},
        },
        {"head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
        {"head": [0.0, 0.0, 2.0], "tail": [0.0, 0.0, 3.0], "constraints": {}},
    ]
    target = [0.6, 0.0, 2.5]

    directions = solve_ccd(chain, target, 6000, TOL)
    joints = _joints(chain, directions)

    # 约束生效的同时末端仍收敛：说明钳制发生在迭代内，而不是解完再拉回。
    assert _dist(joints[-1], target) <= TOL + 1e-9
    assert -9.0 - 1e-9 <= joints[1][0] <= -0.1 + 1e-9

    # 求解结果已经是可行解：再次钳制不应改变根骨骼的尾端。
    reclamped = apply_constraints([joints[1]], chain[0]["constraints"])
    for axis in range(3):
        assert abs(reclamped[0][axis] - joints[1][axis]) <= 1e-9


# 缺陷 4：扭转限制用点积，超过 180 度判反。
def test_limit_twist_projects_into_cone_without_flipping():
    # 与 Y 轴夹角 120 度（钝角），限制到 45 度锥内。
    direction = [math.sin(math.radians(120.0)), math.cos(math.radians(120.0)), 0.0]
    result = limit_twist(direction, [0.0, 1.0, 0.0], 45.0)

    assert abs(_norm(result) - 1.0) <= 1e-9
    angle = math.degrees(math.acos(max(-1.0, min(1.0, result[1]))))
    assert abs(angle - 45.0) <= 1e-9
    # 方位角保持：仍指向 +X 一侧，没有被翻到 -X 或 -Y。
    assert result[0] > 0.0
    assert result[1] > 0.0


# 缺陷 5：stretch_policy=none 仍按距离缩放骨骼。
def test_stretch_policy_none_never_scales_bones():
    chain = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 2.0], "constraints": {}}]
    result = stretch_policy(chain, [0.0, 0.0, 10.0], "none")

    assert result["mode"] == "none"
    assert result["scales"] == [1.0]
    assert _dist(result["end"], [0.0, 0.0, 2.0]) <= 1e-12


# 缺陷 6：权重混合不归一，weight=0.5 时整体缩短。
def test_weight_blend_is_normalized_midpoint():
    pose_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    pose_b = [[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]

    blended = weight_blend(pose_a, pose_b, 0.5)

    assert len(blended) == 2
    assert _dist(blended[0], [0.0, 0.0, 0.0]) <= 1e-12
    assert _dist(blended[1], [1.0, 0.0, 2.0]) <= 1e-12


# 缺陷 7：链组装不检测环，父子互指时无限递归。
def test_chain_from_plan_detects_cycle():
    plan = [
        {"name": "root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}},
        {"name": "a", "parent": "b", "head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
        {"name": "b", "parent": "a", "head": [0.0, 0.0, 2.0], "tail": [0.0, 0.0, 3.0], "constraints": {}},
    ]
    with pytest.raises(IkError):
        chain_from_plan(plan)


# 缺陷 8：正向运动学用世界坐标累加，骨骼缩放下结果偏大。
def test_forward_kinematics_multiplies_scale():
    chain = [
        {"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
        {
            "head": [0.0, 0.0, 0.5],
            "tail": [0.0, 0.0, 1.0],
            "constraints": {},
            "scale": 0.5,
        },
    ]

    positions = forward_kinematics(chain, [10.0, 0.0, 0.0])

    expected = [[10.0, 0.0, 0.0], [10.0, 0.0, 2.0], [10.0, 0.0, 2.5]]
    assert len(positions) == 3
    for got, want in zip(positions, expected):
        assert _dist(got, want) <= 1e-9


# 缺陷 9：批量求解共用可变状态，角色之间互相污染。
def test_solve_batch_does_not_share_mutable_state_between_rigs():
    rig_a = [
        {"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}},
        {"head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
    ]
    rig_b = [
        {"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}},
        {"head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "constraints": {}},
    ]
    rigs = [rig_a, rig_b]
    targets = [[0.5, 0.0, 1.5], [-0.5, 0.0, 1.5]]
    snapshot = copy.deepcopy(rigs)

    batch = solve_batch(rigs, targets, {"iterations": 4000, "tolerance": TOL})

    assert batch["ok"] is True
    assert rigs == snapshot  # 入参不得被原地修改
    assert len(batch["results"]) == 2
    assert batch["results"][0] is not batch["results"][1]

    solo_a = solve_ccd(rig_a, targets[0], 4000, TOL)
    solo_b = solve_ccd(rig_b, targets[1], 4000, TOL)
    for got, want in zip(batch["results"][0], solo_a):
        assert _dist(got, want) <= 1e-9
    for got, want in zip(batch["results"][1], solo_b):
        assert _dist(got, want) <= 1e-9


# 缺陷 10：批量遇无解 rig 只跳过，输出里留下未收敛姿态。
def test_solve_batch_rolls_back_and_drops_unconverged_poses():
    rig_ok = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}}]
    rig_bad = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}}]
    targets = [[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]]

    batch = solve_batch(
        [rig_ok, rig_bad], targets, {"iterations": 20, "tolerance": TOL}
    )

    assert batch["ok"] is False
    assert batch["results"] == []
    assert batch["failed"] == 1


# 组合 1：已到位时迭代 0 次。
def test_ccd_already_at_target_returns_original_directions():
    chain = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 2.0], "constraints": {}}]

    directions = solve_ccd(chain, [0.0, 0.0, 2.0], 1, TOL)

    assert len(directions) == 1
    assert _dist(directions[0], [0.0, 0.0, 1.0]) <= 1e-12


# 组合 2：空链返回空。
def test_ccd_empty_chain_returns_empty():
    assert solve_ccd([], [0.0, 0.0, 1.0], 10, TOL) == []


# 组合 3：weight=0 返回 pose_a。
def test_weight_blend_zero_returns_pose_a():
    pose_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    pose_b = [[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]

    blended = weight_blend(pose_a, pose_b, 0.0)

    assert len(blended) == 2
    for got, want in zip(blended, pose_a):
        assert _dist(got, want) <= 1e-12


# 组合 4：iterations=0 抛 IkError。
def test_ccd_zero_iterations_raises():
    chain = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}}]
    with pytest.raises(IkError):
        solve_ccd(chain, [0.0, 0.0, 1.0], 0, TOL)


# 组合 5：批量全成功时顺序与输入一致。
def test_solve_batch_keeps_input_order_on_full_success():
    rig_a = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}}]
    rig_b = [{"head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0], "constraints": {}}]
    targets = [[0.6, 0.0, 0.8], [-0.6, 0.0, 0.8]]

    batch = solve_batch(
        [rig_a, rig_b], targets, {"iterations": 4000, "tolerance": TOL}
    )

    assert batch["ok"] is True
    assert batch["failed"] == -1
    assert len(batch["results"]) == 2
    assert _dist(_end(rig_a, batch["results"][0]), targets[0]) <= TOL + 1e-9
    assert _dist(_end(rig_b, batch["results"][1]), targets[1]) <= TOL + 1e-9
