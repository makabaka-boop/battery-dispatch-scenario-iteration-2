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

### 可选购电额度

`purchase_budget` 为可选非负整数，省略（`null`/缺省）时求解与原来完全
一致；填写后约束

```
Σ purchase[t] <= purchase_budget      # 所有时段购电量之和
```

额度是**优化内建约束**：额度 DP 把累计购电量作为状态维（稀疏 Pareto 标签
DP over (soc, 已购电量)），在受约束的完整动作树上直接求字典序最优；
**不是**先求无约束最优计划、再事后裁剪。逐段结果附 `remaining_budget`
（剩余额度 = 额度 − 截至该段累计购电量，可与累计购电互相复算，全程非负）。

两种无解严格区分，且都不生成局部“当前计划”：

- **物理无解**：任何额度下都无法达到终点电量等物理条件，
  `minimum_purchase = null`；
- **仅因额度不足无解**：物理可行计划存在，但最小总购电量超过额度，
  `minimum_purchase` 给出满足原物理条件所需的**最小总购电量**，
  把额度提高到该值即可求解。

## 优化目标（严格字典序）

1. 最小化总购电费用 `Σ price[t] * purchase[t]`；
2. 并列时最小化总购电量 `Σ purchase[t]`；
3. 仍并列时取充放电序列 `(c0, d0, c1, d1, …)` 字典序最小者
   （等价于“尽早少充、动作尽量小”的充放计划）。

求解用整数状态上的**精确后向动态规划**（容量 ≤ 50，48 段约几十毫秒），
前向按最小动作贪心重建并列最优序列。填写购电额度时，改用在 (电量状态,
累计购电量) 上的稀疏 Pareto 标签 DP（购电、费用双维剪枝），仍严格按
费用 → 购电量 → 序列三级裁决。

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

`purchase_budget` 是情景输入的一部分（省略即 `null`，与显式 `0` 不同），
随修订一起保存、参与乐观锁与输入签名；读取早于该字段的旧存档时自动
按“不设额度”补全，旧数据可正常加载与求解。

### 求解响应与证据

每段返回 `load/pv/price/soc_start/charge/discharge/soc_end/net_load/
purchase/curtail/cost/remaining_budget`，并附
`total_cost/total_purchase/final_soc/purchase_budget/minimum_purchase`。
前端对每段重新核验电量守恒、无同段充放、放电 ≤ 净负载、
`purchase = max(0, net - d + c)`、`curtail = max(0, -(net - d + c))`、
费用 = 购电量 × 单价，并在设了额度时核验
`remaining_budget = purchase_budget − 累计购电量` 且全程非负；
无可行解时返回 `feasible=false` 与原因（物理无解 `minimum_purchase=null`，
额度不足时给出 `minimum_purchase`；HTTP 仍为 200，参数校验错误才是 422）。

## 前端“旧计划不得成为新情景计划”的保证

- 求解结果记录它对应的**输入签名**（全部参数——含 `purchase_budget`——
  的 JSON 摘要）以及修订号；修改额度与修改其他参数一样会改变签名；
- 一旦编辑任何参数（含额度），或保存产生新修订号，旧结果立即变为
  “历史结果（已失效）”，灰色展示且明确标注不得作为当前计划，
  “当前计划”区置空；
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
  购电额度另做：预算边界（`q−1 / q / q+1` 等，300 种子）与随机额度
  （200 种子）的全树对拍、费用与用量目标冲突（同用量不同费用）、并列
  裁决、物理无解与额度不足两种无解的区分及“无局部计划”、剩余额度复算。
- `tests/test_api.py`：参数校验（含负额度 422、省略为 `null`）、修订历史、
  `409` 竞争拒绝、8 线程并发保存（恰有一个成功、其余全部 409）、
  带重试的串行链、求解证据逐段核验（含剩余额度）、**改额度产生新修订并使
  旧结果过期**、指定修订求解不串号、缺字段旧存档可读取、草稿求解不落库、
  静态页可达。

## 目录

```
app/solver.py    求解器（DP + 穷举参考实现）
app/models.py    Pydantic 模型与校验（8–48 段、容量 ≤50 等）
app/storage.py   线程安全情景库与乐观修订
app/main.py      FastAPI 路由
web/             Vue 3 情景页面（本地 vendor，无外网依赖）
tests/           对拍与 API/并发测试
```
