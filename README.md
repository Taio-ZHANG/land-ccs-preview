# 蓝和 CCS 预览 · LAND CCS Preview

使用 Python、Tkinter、NumPy 和 Matplotlib 在本机读取蓝和 LAND `.ccs` 电池测试文件，查看曲线并导出 CSV。

A local desktop viewer and CSV exporter for supported LAND CCS battery-test files.

## 功能

- 预览电压、电流、容量、循环充放电容量及库仑效率。
- 查看指定循环的容量/比容量—电压曲线。
- 按循环范围合并导出 CSV，每圈为一组 X/Y 列，便于后续绘图。
- 在支持的文件布局中读取活性物质质量；不能可靠确认时允许手动输入。

## 安装与运行

推荐 Windows 和 Python 3.14，安装 Python 时保留 Tcl/Tk 支持。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe land_ccs_preview.py
```

也可将文件路径作为启动参数传入。若默认 Python 已有依赖，可双击 `启动蓝和预览.bat`。

## 数据解释与适用范围

这是独立开发的文件预览工具，与蓝和厂商无隶属关系，也不包含厂商软件或原始实验数据。
CCS 属于二进制格式，解析逻辑基于已验证的文件布局，不保证覆盖所有仪器、固件和 CCS 变体。
在用于定量分析前，请对照仪器原始软件核对时间、容量、质量、循环编号和单位。
比容量取决于活性物质质量；无法识别、缺失或两处质量元数据不一致时，需人工核对输入。
输入文件按读取处理；导出的 CSV 写到用户选择的位置。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest test_capacity_export -v
```

现有回归测试使用合成曲线和临时文件，检查循环范围、X/Y 列、空值补齐、比容量与导出异常处理。
测试需要可用的桌面/Tk 环境。它验证导出行为，不等于已验证所有真实 CCS 格式。

## 公开快照与许可

首个公开快照标记为 0.1.0，来源为 2026-11-06 当前本机版本；排除了历史备份、交接文档中的本机信息和实验 CSV。
代码采用 [MIT](LICENSE)。NumPy、Matplotlib、Python/Tk 遵守各自许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
