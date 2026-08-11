# Arena Hero 策略架构

策略采用单向决策流：

```text
Core + Worker + Vanguard + Ranger 的独立视野
                       ↓
             全局观察与持久地图记忆
                       ↓
      TacticalMap（敌情 / 风险 / 资源 / 地形 / 拥堵）
                       ↓
 Worker        Vanguard        Ranger          Core
 经济避险        截击封路        火力调度        撤退生产
                       ↓
             ActionIntent 全局仲裁
                       ↓
                   SDK 发射
```

## 核心边界

- 视野属于全队地图，不属于 Worker 模块。Core、Worker、Vanguard、Ranger 按官方视距和障碍遮挡分别计算贡献，再取并集。
- `WorldModel` 保存当前纯值快照和持久地图事实；`TacticalMap` 是单 Tick 内唯一的团队战术解释。
- 所有角色收到同一 Tick、同一事实版本的 `TacticalMap`；服务通道和最终预订只生成不可变操作叠层。角色只能选择不同风险容忍度，不得各自重建另一套敌情真相。
- 经济、战斗、治疗、远征等模块只生成脱离 SDK Controller 的 `ActionIntent`；只有 `BalancedTactic` 最终修改当前 `Turn`。
- 迷雾中的敌军轨迹、敌核和资源都是带时间与置信度的记忆，不得冒充当前占用。
- `TacticMemory` 不得保存 `Turn`、Core/Unit Controller 或其他 SDK 活对象。

## 全局地图内容

`TacticalMap` 统一维护：

- 永久地形：已知障碍、已知可通行格。
- 视野：当前可见格、最后可见 Tick、每格的友军观察者、每个友军的独立覆盖。
- 资源：当前是否可见、最后确认 Tick、当前分配的 Worker。
- 敌情：类型、最后位置、连续轨迹、合法一步落点、移动走廊、置信度和观察来源。
- 威胁：当前攻击区、下一 Tick 攻击区、迷雾移动走廊、事件热区和线性衰减风险。
- 友军：当前位置、预计移动、交付通道、服务区、占用和仲裁预订。

由任意兵种发现的资源都会进入 Worker 全局匹配；由任意兵种发现的敌军都会进入同一威胁场，并可触发 Worker 避险、守军响应和 Core 压力评估。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `world.py` | 同步事件、独立视野贡献和长期地图事实，生成 `WorldModel` |
| `projection.py` | 从世界事实一次性生成不可变 `TacticalMap` |
| `planning.py` | A*、BFS、风险加权路径和信息增益 |
| `worker.py` / `resource_allocator.py` | 从全局资源和风险图生成采集、返航、探索与避险意图 |
| `combat.py` / `defense.py` | 从全局敌情生成射线、齐射、截击、封路和阵位意图 |
| `service.py` / `recovery.py` | Core 通道、FIFO 交付和治疗仲裁 |
| `core_safety.py` / `production.py` | 从整体压力和预算生成迁移、治疗及生产意图 |
| `resolver.py` | 资源预算、容量、移动依赖、保护区和单格独占的全局仲裁 |
| `trace.py` | schema 22 决策追踪，包括全局地图、满仓驻守、动态射击位和最终预订 |
| `persistence.py` | schema 11 检查点；永久地图、资源、热区和短期敌情可跨后台重启恢复 |

## 决策优先级

1. Core 保命与持续撤退。
2. 静止 Core 上的载货 Worker 交付。
3. 单位致命避险或最后攻击。
4. 对 Core 或友军构成威胁的合法攻击。
5. 紧急治疗与治疗筹资。
6. 载货返航、采集和本土防守。
7. 远征、探索与和平巡逻。
8. 无合法任务时显式 `WAIT`。

## 验证

```powershell
D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe -m unittest discover -s tests -v
D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe -m compileall -q balanced_tactic.py replay_log.py arena_tactic tests
D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe -m pip check
```

规则基线为 Gameplay v0.14、官方 Python SDK v0.2.9。
