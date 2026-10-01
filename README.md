# 虚拟电池情景求解服务（FastAPI + Vue）

纯模拟系统，**不连接任何真实电力设备**。每份情景含 8～48 个整数时段，对
虚拟电池做逐段调度，给出购电计划与证据。

## 物理模型

每段只能选择一个动作（不允许同段又充又放）：

```
soc[t+1] = soc[t] + charge[t] - discharge[t]
0 <= soc <= capacity (capacity <= 50)
0 <= charge <= max_charge
0 <= discharge <= min(max_charge, max(0, load - pv))   # 放电不得超过净负载
residual[t] = (load - pv) - discharge + charge
purchase[t] = max(0, residual[t])                       # 只能买电
curtail[t]  = max(0, -residual[t])                      # 多余光伏弃用，不能售电
```

终点要求 `soc[n] >= terminal_min_soc`。购电与弃光不可能同段并存
（`residual` 不可能同时为正和为负），从模型上保证“不售电”。

### 总购电额度（可选）

`purchase_budget` 为可省略的非负整数。省略时直接得到无约束最优计划；
填写后要求 **所有时段购电量之和 ≤ 额度**，约束在 DP 内部生效
（**不是**先求无约束最优再事后筛除）。一个有用的模型恒等式：

```
Σ purchase − Σ curtail = Σ(load − pv) + soc[n] − soc[0]
```

终点电量固定时，多买的电量必然同时对应多弃掉的光伏（纯浪费），因此
费用最优计划本身就达到最小总购电量：额度只起到“可行性闸门”作用——
低于最小总购电量即无解。无可行解分两类，均 HTTP 200 且不生成局部计划：

- **物理本就无解**（终点电量不可达）：`required_min_purchase` 为 `null`；
- **仅额度不足**：`required_min_purchase` 给出满足原物理条件所需的
  **最小总购电量**，把额度提到该值即可恢复可行。

## 优化目标（严格字典序）

1. 最小化总购电费用 `Σ price[t] * purchase[t]`；
2. 并列时最小化总购电量 `Σ purchase[t]`；
3. 仍并列时取充放电序列 `(c0, d0, c1, d1, …)` 字典序最小者
   （等价于“尽早少充、动作尽量小”的充放计划）。

求解用整数状态上的**精确后向动态规划**（容量 ≤ 50，48 段约几十毫秒），
前向按最小动作贪心重建并列最优序列。带额度时先算无约束 DP 判定可行性与
最小总购电量；额度起约束时用“累计购电量”稀疏 DP，每状态仅保留
（后缀购电量, 后缀费用）的帕累托前沿，再做同样的前向重建。

## 运行

```bash
pip install -r requirements.txt          # 或用 uv/venv
uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

情景数据默认存 `data/scenarios.json`（可用 `VBAT_DB` 环境变量覆盖）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/scenarios` | 创建情景，修订号从 1 开始 |
| GET | `/api/scenarios` | 列表（各情景最新修订） |
| GET | `/api/scenarios/{id}?revision=N` | 取某修订（缺省取最新；全部历史修订保留） |
| PUT | `/api/scenarios/{id}` | **乐观锁保存**，请求体含 `expected_revision` |
| DELETE | `/api/scenarios/{id}` | 删除 |
| POST | `/api/scenarios/{id}/solve?revision=N` | 求解某修订（缺省最新），响应回显所解修订号 |
| POST | `/api/solve` | 对未保存草稿直接求解，不落库 |

### 修订竞争（并发安全）

保存必须带客户端所基于的修订号：

```jsonc
// PUT /api/scenarios/{id}
{ "expected_revision": 2, "scenario": { ... } }
```

服务端在锁内比较最新修订号：不一致即返回 `409`，并回带当前修订号，
**拒绝覆盖**，新修订不会写入：

```json
{ "detail": "expected revision 2, current revision is 3", "current_revision": 3 }
```

### 求解响应与证据

每段返回 `load/pv/price/soc_start/charge/discharge/soc_end/net_load/
purchase/curtail/cost`（带额度时另附 `remaining_budget`），并附
`total_cost/total_purchase/final_soc`，无解时附 `purchase_budget` 与
`required_min_purchase`。前端对每段重新核验电量守恒、无同段充放、放电 ≤ 净负载、
`purchase = max(0, net - d + c)`、`curtail = max(0, -(net - d + c))`、
费用 = 购电量 × 单价，并核验 `剩余额度 = 总额度 − 累计购电`（两者互相复算）、
总购电量不超过额度；无可行解时返回 `feasible=false` 与原因（HTTP 仍为
200，参数校验错误才是 422），且不返回任何局部“当前计划”。

## 前端“旧计划不得成为新情景计划”的保证

- 求解结果记录它对应的**输入签名**（全部参数，含 `purchase_budget`，
  的 JSON 摘要）以及修订号；
- 一旦编辑任何参数（含修改额度），或保存产生新修订号，旧结果立即变为
  “历史结果（已失效）”，灰色展示且明确标注不得作为当前计划，“当前计划”
  区置空；
- 新情景加载时清空全部旧求解响应；
- 保存成功后提示重新求解；求解指定修订时响应回显修订号，
  r1 的计划永远不会被误当成 r2 的计划。

## 测试

```bash
pytest
```

- `tests/test_solver.py`：短时段（容量 ≤5、1～4 段，300 个随机种子 +
  固定边界用例）下，DP 与**穷举全部合法动作树**的参考实现逐字段对拍；
  含字典序并列、费用优先于购电量、不可行判定，以及 48 段满规模不变量检查。
  另有 150 个种子对每个整数额度 `0..上界` 全量对拍，覆盖额度边界
  （恰好等于最小购电量可行、低一单位仅因额度无解）、费用/用量裁决、
  并列序列，以及两类无解的区分与最小总购电量。
- `tests/test_api.py`：参数校验（含额度须为非负整数）、修订历史、
  `409` 竞争拒绝、8 线程并发保存（恰有一个成功、其余全部 409）、
  带重试的串行链、求解证据逐段核验（含剩余额度与累计购电互算）、
  指定修订求解不串号、改额度即产生新修订且旧结果失效、草稿求解不落库、
  **无该字段的旧存档仍可读取并按无额度求解**、静态页可达。

## 目录

```
app/solver.py    求解器（DP + 穷举参考实现）
app/models.py    Pydantic 模型与校验（8–48 段、容量 ≤50 等）
app/storage.py   线程安全情景库与乐观修订
app/main.py      FastAPI 路由
web/             Vue 3 情景页面（本地 vendor，无外网依赖）
tests/           对拍与 API/并发测试
```
