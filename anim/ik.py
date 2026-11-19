"""角色骨骼反向运动学（IK）求解骨架。

坐标与单位约定
--------------
* 世界坐标系为右手系，Y 轴向上，单位为任意一致的长度单位。
* 所有角度一律使用**度**（degree）；所有长度与坐标单位一致。
* 所有距离容差默认 ``1e-6``。
* 骨骼一律使用内联字典，形如::

      {"head": [x, y, z], "tail": [x, y, z], "constraints": {...}}

  ``constraints`` 为逐轴限位 ``{"x": [min, max], "y": [...], "z": [...]}``，
  缺省的轴表示不限制。``forward_kinematics`` 的骨骼还可带 ``scale``（默认 1.0）。

初始状态下下列 8 个公开函数全部为未实现的桩（``NotImplementedError``），
``tests/test_ik.py`` 中的 15 条测试据此全红；实现时不得改动测试。
"""

import math


TOLERANCE = 1e-6

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class IkError(Exception):
    """IK 输入非法、求解失败或批量回滚时抛出的异常。"""


# ---------------------------------------------------------------------------
# 私有三维向量辅助
# ---------------------------------------------------------------------------

def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _scale(s, v):
    return [s * c for c in v]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dist(a, b):
    return _norm(_sub(a, b))


def _normalize(v):
    n = _norm(v)
    if n <= TOLERANCE:
        raise IkError("零向量无法单位化")
    return [c / n for c in v]


def _arbitrary_perp(v):
    """返回与 ``v`` 垂直的单位向量（v 本身为单位向量时使用）。"""
    axis = min(range(3), key=lambda i: abs(v[i]))
    w = [0.0, 0.0, 0.0]
    w[axis] = 1.0
    return _normalize(_cross(v, w))


def _rotate_points(points, pivot, axis, angle):
    """绕 ``pivot``、按单位轴 ``axis`` 旋转角 ``angle``（弧度），返回新点列表。"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    rotated = []
    for point in points:
        v = _sub(point, pivot)
        cr = _cross(axis, v)
        dt = _dot(axis, v)
        one_minus = 1.0 - ca
        rotated.append(
            _add(
                pivot,
                [
                    v[i] * ca + cr[i] * sa + axis[i] * dt * one_minus
                    for i in range(3)
                ],
            )
        )
    return rotated


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def solve_ccd(chain, target, iterations, tolerance):
    """用循环坐标下降（CCD）求解骨骼链，使末端尽量到达 ``target``。

    参数
    ----
    chain : list[dict]
        从根到末端排列的骨骼链（见模块 docstring）；各节首尾相接。
    target : list[float]
        目标点 ``[x, y, z]``，世界坐标。
    iterations : int
        最大迭代轮数，必须 ``>= 1``；否则抛 :class:`IkError`。
    tolerance : float
        收敛容差（世界单位），约定值为 ``1e-6``；末端误差
        ``<= tolerance`` 时立即停止，不得跑满固定次数。

    契约
    ----
    * 每轮从末端骨骼向根骨骼依次旋转：把当前末端绕该骨骼的 ``head``
      转到目标方向，并同步更新其后的所有关节。
    * 骨骼 ``constraints`` 是逐轴关节限位（约束该节尾端的世界坐标，
      单位与坐标一致）；钳制必须在**每轮迭代内立即进行**：把受约束轴夹到
      合法区间，同时保持该节骨骼长度不变（尾端落到以 ``head`` 为球心、
      该节长度为半径的球面上），再平移其后所有关节；不得留到求解结束后
      再钳制。
    * 返回每节骨骼的新朝向，长度与 ``chain`` 相同；每项为从 ``head`` 指向
      ``tail`` 的单位向量 ``[x, y, z]``。
    * 空链返回 ``[]``。
    * 迭代耗尽后末端误差仍大于 ``tolerance``（例如目标不可达）时抛
      :class:`IkError`。
    """
    if not chain:
        return []
    if not isinstance(iterations, int) or iterations < 1:
        raise IkError("iterations 必须为 >= 1 的整数")

    target = [float(c) for c in target]
    n = len(chain)
    lengths = [_dist(bone["head"], bone["tail"]) for bone in chain]

    # 仅在局部副本上工作，绝不原地修改入参链。
    joints = [[float(c) for c in chain[0]["head"]]]
    for bone in chain:
        joints.append([float(c) for c in bone["tail"]])

    def end_error():
        return _dist(joints[-1], target)

    def align_bone(i):
        """把末端绕 joints[i] 旋转到目标方向（最小转角），同步其后关节。"""
        if lengths[i] <= TOLERANCE:
            return
        pivot = joints[i]
        to_end = _sub(joints[-1], pivot)
        to_target = _sub(target, pivot)
        ne = _norm(to_end)
        nt = _norm(to_target)
        if ne <= TOLERANCE or nt <= TOLERANCE:
            return
        a = _scale(1.0 / ne, to_end)
        b = _scale(1.0 / nt, to_target)
        cos_angle = max(-1.0, min(1.0, _dot(a, b)))
        # 跳过阈值必须远小于收敛容差对应的角量，否则末端逼近目标时的
        # 微小旋转会被整体跳过，误差卡死在容差之外。
        if cos_angle >= 1.0 - 1e-15:
            return  # 已同向
        axis = _cross(a, b)
        axis_norm = _norm(axis)
        if axis_norm <= 1e-15:
            # 反向平行：任选一个垂直轴转 180 度。
            axis = _arbitrary_perp(a)
            angle = math.pi
        else:
            axis = _scale(1.0 / axis_norm, axis)
            angle = math.acos(cos_angle)
        joints[i + 1:] = _rotate_points(joints[i + 1:], pivot, axis, angle)

    def clamp_bone(i):
        """逐轴钳制第 i 节尾端，球面投影保长，再平移其后所有关节。"""
        constraints = chain[i].get("constraints") or {}
        if not constraints:
            return
        head = joints[i]
        old_tail = joints[i + 1]
        new_tail = list(old_tail)
        for axis_name, bounds in constraints.items():
            k = _AXIS_INDEX[axis_name]
            low, high = float(bounds[0]), float(bounds[1])
            if new_tail[k] < low:
                new_tail[k] = low
            elif new_tail[k] > high:
                new_tail[k] = high
        radial = _sub(new_tail, head)
        radial_norm = _norm(radial)
        if radial_norm <= TOLERANCE:
            return  # 退化构型：放弃本次钳制，避免除零
        projected = _add(head, _scale(lengths[i] / radial_norm, radial))
        delta = _sub(projected, old_tail)
        for k in range(i + 1, n + 1):
            joints[k] = _add(joints[k], delta)

    # 已到位：零次迭代，返回原始朝向。
    if end_error() <= tolerance:
        return [
            _normalize(_sub(joints[i + 1], joints[i])) for i in range(n)
        ]

    for _ in range(iterations):
        # 每轮从末端向根依次处理；旋转后立即在同一轮内钳制。
        for i in range(n - 1, -1, -1):
            align_bone(i)
            clamp_bone(i)
        if end_error() <= tolerance:
            break

    if end_error() > tolerance:
        raise IkError("CCD 在给定迭代次数内未收敛到容差内")

    return [_normalize(_sub(joints[i + 1], joints[i])) for i in range(n)]


def apply_constraints(joints, constraints):
    """按逐轴限位钳制关节坐标，返回新列表且不修改入参。

    参数
    ----
    joints : list[list[float]]
        关节世界坐标列表 ``[[x, y, z], ...]``。
    constraints : dict
        ``{"x": [min, max], "y": [...], "z": [...]}``；缺省的轴不限制。
        单位与坐标一致（不是角度）。

    契约
    ----
    * 返回长度与 ``joints`` 相同的坐标列表，每个坐标的每个受限轴都落在
      闭区间 ``[min, max]`` 内（含端点）。
    * 只钳制、不旋转、不缩放，也不得原地修改 ``joints``。
    """
    result = []
    for joint in joints:
        clamped = [float(c) for c in joint]
        for axis_name, bounds in (constraints or {}).items():
            k = _AXIS_INDEX[axis_name]
            low, high = float(bounds[0]), float(bounds[1])
            if clamped[k] < low:
                clamped[k] = low
            elif clamped[k] > high:
                clamped[k] = high
        result.append(clamped)
    return result


def limit_twist(direction, axis, max_deg):
    """把方向限制在绕 ``axis`` 的半锥内，锥半角为 ``max_deg`` 度。

    参数
    ----
    direction : list[float]
        待限制方向 ``[x, y, z]``，允许非单位向量，但不得为零向量。
    axis : list[float]
        锥轴方向 ``[x, y, z]``，允许非单位向量，但不得为零向量。
    max_deg : float
        允许的最大夹角，单位度，取值 ``[0, 180]``。

    契约
    ----
    * 若 ``direction`` 与 ``axis`` 的夹角 ``<= max_deg``，返回 ``direction``
      的单位化结果，方向不变。
    * 否则保持绕 ``axis`` 的方位角（垂直分量的方向）不变，仅把夹角压到
      ``max_deg``，返回单位向量。
    * 必须按旋转/投影处理，不得用点积符号判别钝角：夹角超过 90 度时
      不得把方向判反到轴的另一侧。
    """
    d = _normalize(direction)
    a = _normalize(axis)
    cos_angle = max(-1.0, min(1.0, _dot(d, a)))
    angle = math.degrees(math.acos(cos_angle))
    if angle <= max_deg:
        return d

    # 分解为平行分量与垂直分量；垂直分量承载方位角，钝角下方向依然正确。
    parallel = _scale(_dot(d, a), a)
    perp = _sub(d, parallel)
    perp_norm = _norm(perp)
    if perp_norm <= TOLERANCE:
        # 恰好反向平行时方位角未定义，任选一个垂直方向兜底。
        perp_hat = _arbitrary_perp(a)
    else:
        perp_hat = _scale(1.0 / perp_norm, perp)

    rad = math.radians(max_deg)
    return _add(
        _scale(math.cos(rad), a),
        _scale(math.sin(rad), perp_hat),
    )


def stretch_policy(chain, target, mode):
    """决定骨骼链对目标的拉伸策略。

    参数
    ----
    chain : list[dict]
        骨骼链（见模块 docstring）；空链抛 :class:`IkError`。
    target : list[float]
        目标点 ``[x, y, z]``。
    mode : str
        ``"none"``：不拉伸；``"uniform"``：各节等比缩放；
        ``"last"``：只拉伸末节。其他取值抛 :class:`IkError`。

    契约
    ----
    * 设 ``base = chain[0]["head"]``、``total`` 为链总长、``dist`` 为
      ``base`` 到 ``target`` 的距离，需要拉伸的长度为 ``max(0, dist - total)``。
    * ``mode="none"`` 时无论目标多远，``scales`` 必须全为 ``1.0``，
      ``end`` 保持当前末端 ``chain[-1]["tail"]``。
    * ``dist <= total`` 时无需拉伸：``scales`` 全为 ``1.0``，``end`` 为当前末端。
    * ``mode="uniform"``：``scales = [dist / total] * len(chain)``。
    * ``mode="last"``：前 ``n-1`` 节为 ``1.0``，末节为
      ``(len_last + dist - total) / len_last``。
    * 返回 ``{"mode": mode, "scales": [...], "end": [x, y, z]}``，
      ``scales`` 长度等于 ``len(chain)``。
    """
    if not chain:
        raise IkError("空链无法计算拉伸策略")
    if mode not in ("none", "uniform", "last"):
        raise IkError("未知拉伸模式: %r" % (mode,))

    base = chain[0]["head"]
    lengths = [_dist(bone["head"], bone["tail"]) for bone in chain]
    total = sum(lengths)
    dist = _dist(base, target)
    n = len(chain)
    current_end = [float(c) for c in chain[-1]["tail"]]

    if mode == "none" or dist <= total:
        return {
            "mode": mode,
            "scales": [1.0] * n,
            "end": current_end,
        }

    if total <= TOLERANCE:
        raise IkError("零长度骨骼链无法拉伸")

    if mode == "uniform":
        scales = [dist / total] * n
    else:
        last_len = lengths[-1]
        if last_len <= TOLERANCE:
            raise IkError("末节骨骼长度为零，无法只拉伸末节")
        scales = [1.0] * (n - 1) + [
            (last_len + dist - total) / last_len
        ]

    # 拉伸后链总长恰为 dist，末端沿 base->target 方向到达目标点。
    return {
        "mode": mode,
        "scales": scales,
        "end": [float(c) for c in target],
    }


def weight_blend(pose_a, pose_b, weight):
    """按权重线性混合两个姿态。

    参数
    ----
    pose_a, pose_b : list[list[float]]
        关节世界坐标列表，两者长度必须一致，否则抛 :class:`IkError`。
    weight : float
        混合系数，必须位于 ``[0, 1]``；越界抛 :class:`IkError`。

    契约
    ----
    * 返回 ``[(1 - weight) * a + weight * b]`` 的逐关节结果。
    * 必须归一化（两个系数和为 1），不得出现 ``weight=0.5`` 时姿态整体缩短。
    * ``weight=0`` 返回 ``pose_a`` 的拷贝，``weight=1`` 返回 ``pose_b`` 的拷贝。
    """
    if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
        raise IkError("weight 必须位于 [0, 1]")
    if len(pose_a) != len(pose_b):
        raise IkError("两个姿态的关节数量必须一致")

    wa = 1.0 - weight
    return [
        [wa * float(a[k]) + weight * float(b[k]) for k in range(len(a))]
        for a, b in zip(pose_a, pose_b)
    ]


def chain_from_plan(plan):
    """按父子关系把 ``plan`` 组装成从根到末端的骨骼链。

    参数
    ----
    plan : list[dict]
        ``[{"name": str, "parent": str | None, "head": [...],
        "tail": [...], "constraints": {...}}, ...]``。

    契约
    ----
    * 必须有且只有一个 ``parent`` 为 ``None`` 的根；每个 ``parent`` 必须存在；
      父子关系必须构成单链（每个节点至多一个子节点）；不允许环；不允许
      从根不可达的节点。违反任一条件抛 :class:`IkError`。
    * 环检测不得无限递归。
    * 返回骨骼字典列表（根在前），每项保留 ``head``、``tail``、
      ``constraints``，若 ``plan`` 中带 ``scale`` 也一并保留。
    """
    by_name = {}
    for node in plan:
        name = node["name"]
        if name in by_name:
            raise IkError("节点重名: %r" % (name,))
        by_name[name] = node

    roots = [node for node in plan if node["parent"] is None]
    if len(roots) != 1:
        raise IkError("骨骼计划必须有且只有一个根节点")

    # 校验父节点存在且每个节点至多一个子节点（单链）。
    child_count = {name: 0 for name in by_name}
    for node in plan:
        parent = node["parent"]
        if parent is not None:
            if parent not in by_name:
                raise IkError("节点 %r 的父节点 %r 不存在" % (node["name"], parent))
            child_count[parent] += 1
            if child_count[parent] > 1:
                raise IkError("节点 %r 拥有多个子节点，不构成单链" % (parent,))

    # 迭代式从根向下走，visited 同时承担环检测，杜绝无限递归。
    ordered = []
    visited = set()
    current = roots[0]
    while current is not None:
        if current["name"] in visited:
            raise IkError("父子关系中存在环")
        visited.add(current["name"])
        ordered.append(current)
        nxt = None
        for node in plan:
            if node["parent"] == current["name"]:
                nxt = node
                break
        current = nxt

    if len(visited) != len(plan):
        raise IkError("存在从根节点不可达的节点（可能含环或孤儿分支）")

    chain = []
    for node in ordered:
        constraints = node.get("constraints") or {}
        bone = {
            "head": [float(c) for c in node["head"]],
            "tail": [float(c) for c in node["tail"]],
            "constraints": {
                axis: [float(c) for c in bounds]
                for axis, bounds in constraints.items()
            },
        }
        if "scale" in node:
            bone["scale"] = node["scale"]
        chain.append(bone)
    return chain


def forward_kinematics(chain, root):
    """逐节累乘计算正向运动学的关节世界坐标。

    参数
    ----
    chain : list[dict]
        骨骼链；每节的 ``head`` / ``tail`` 是相对父节尾端的局部向量，
        可带 ``scale``（默认 ``1.0``），表示该节局部缩放。
    root : list[float]
        根关节世界坐标 ``[x, y, z]``。

    契约
    ----
    * 每节的缩放与父级累乘（``cum *= scale``）后作用于该节局部向量。
    * 返回长度 ``len(chain) + 1`` 的关节世界坐标列表：第 0 项为 ``root``，
      第 ``i + 1`` 项为第 ``i`` 节尾端的世界坐标，末项即末端位置。
    * 空链返回 ``[root]``。
    * 不得把局部坐标当世界坐标直接累加（骨骼缩放时结果会偏大）。
    """
    positions = [[float(c) for c in root]]
    cumulative = 1.0
    for bone in chain:
        cumulative *= bone.get("scale", 1.0)
        # tail 是该节尾端在父节局部系下的位置；先乘累计缩放再进入世界系。
        local_tail = bone["tail"]
        parent_pos = positions[-1]
        positions.append(
            [
                parent_pos[i] + cumulative * float(local_tail[i])
                for i in range(3)
            ]
        )
    return positions


def solve_batch(rigs, targets, config):
    """批量求解多个角色，语义为全有或全无。

    参数
    ----
    rigs : list[list[dict]]
        骨骼链列表；``targets`` 为等长的目标点列表。
    targets : list[list[float]]
        每个角色的目标点。
    config : dict
        ``{"iterations": int, "tolerance": float}``，透传给每条链的
        :func:`solve_ccd`。

    契约
    ----
    * 全部成功时返回 ``{"ok": True, "results": [...], "failed": -1}``；
      ``results`` 与 ``rigs`` 同序，每项等于对相应链单独调用
      :func:`solve_ccd` 的结果，且各结果之间不得共享可变状态。
    * 任一 ``rig`` 抛 :class:`IkError`（无解或不收敛）时整体回滚，返回
      ``{"ok": False, "results": [], "failed": <失败下标>}``；不得跳过失败项
      后把未收敛姿态留在输出里。
    * 不得原地修改 ``rigs`` / ``targets``。
    """
    if len(rigs) != len(targets):
        raise IkError("rigs 与 targets 长度必须一致")

    iterations = config["iterations"]
    tolerance = config["tolerance"]
    results = []

    for index, (rig, target) in enumerate(zip(rigs, targets)):
        # 深拷贝入参（不使用 copy 模块），求解与缓存均不触碰调用方数据。
        rig_copy = []
        for bone in rig:
            bone_copy = dict(bone)
            bone_copy["head"] = [float(c) for c in bone["head"]]
            bone_copy["tail"] = [float(c) for c in bone["tail"]]
            constraints = bone.get("constraints") or {}
            bone_copy["constraints"] = {
                axis: [float(c) for c in bounds]
                for axis, bounds in constraints.items()
            }
            rig_copy.append(bone_copy)
        target_copy = [float(c) for c in target]

        try:
            directions = solve_ccd(
                rig_copy, target_copy, iterations, tolerance
            )
        except IkError:
            # 全有或全无：丢弃此前所有结果，整体回滚。
            return {"ok": False, "results": [], "failed": index}
        results.append(directions)

    return {"ok": True, "results": results, "failed": -1}
