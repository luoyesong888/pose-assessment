# KinetiQ

KinetiQ 是一个基于 Streamlit 的运动康复评估应用。它先用 MediaPipe 在本地提取姿态关键点，再把结构化结果交给 DeepSeek 生成面向治疗师的康复报告。

## 功能

- 正面与侧面照片上传
- 姿态关键点识别与标记骨架图生成
- ACL 风险分层与动力链分析
- 肌群功能推断
- DeepSeek 个性化康复建议
- 患者档案管理与历史记录
- HTML 和 PDF 报告导出

## 项目结构

- `main.py`：Streamlit 前端界面
- `app_pipeline.py`：完整评估流程
- `pose.py`：MediaPipe 姿态识别和标记图片生成
- `analysis.py`：姿态指标、ACL 风险、动力链总结
- `clinical_knowledge.py`：肌群映射和报告模板
- `deepseek_client.py`：DeepSeek API 调用
- `records_store.py`：患者档案存储
- `report_export.py`：HTML 和 PDF 导出

## 配置

1. 创建并激活虚拟环境。
2. 安装依赖。
3. 将 `pose_landmarker.task` 放在项目根目录。
4. 如果要使用 DeepSeek 生成报告，设置 `DEEPSEEK_API_KEY`。

## 启动

```bash
streamlit run main.py
```

## 说明

- 如果没有设置 `DEEPSEEK_API_KEY`，程序会自动使用本地报告模板。
- 患者档案和上传图片会保存在本地 `data/` 目录。
- 这个项目用于筛查和工作流辅助，不是医学诊断工具。
