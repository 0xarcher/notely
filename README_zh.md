# Notely

[English](README.md) | 简体中文

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/ASR-FunASR%20%7C%20Whisper-orange.svg" alt="ASR">
  <img src="https://img.shields.io/badge/OCR-PaddleOCR-red.svg" alt="OCR">
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic%20%7C%20智谱-purple.svg" alt="LLM">
</p>

<p align="center">
  <em>将视频/音频课程自动转换为结构化 Markdown 笔记</em>
</p>

---

**Notely** 是一个 Python SDK，使用 ASR、OCR 和 LLM 技术，自动将课程视频、音频录音和演示文稿转换为高质量的 Markdown 笔记。

## 核心特性

- 🎯 **高质量语音识别** - FunASR（中文 CER < 3%）、Whisper（多语言）
- 📊 **智能 OCR** - PaddleOCR + 关键帧去重
- 🤖 **多 LLM 支持** - OpenAI、智谱、Anthropic、Moonshot、DeepSeek
- 🧠 **三层增强架构** - 理解层 → 组织层 → 格式化层
- ✂️ **语义分块** - 智能文本分段（2000 tokens，1000 重叠）
- 📐 **LaTeX 公式支持** - 数学符号渲染
- 🌍 **语言自动检测** - 自动识别转录文本语言
- ⚡ **并发处理** - 并行处理多个文本块，提升效率
- ✨ **精美输出** - 结构化 Markdown，自动格式化
- 🔧 **灵活配置** - 零配置启动，支持深度定制

---

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/0xarcher/notely.git
cd notely

# 安装依赖（推荐使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# 或使用 pip
pip install -e ".[all]"

# 安装 FFmpeg（必需）
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### 2. 运行示例

```python
from notely import Notely, NotelyConfig, EnhancerConfig, LLMConfig

# 方法 1：从配置对象创建
config = NotelyConfig(
    enhancer=EnhancerConfig(
        llm=LLMConfig(
            api_key="sk-xxx",
            model="gpt-4o",
        )
    )
)
notely = Notely(config)

# 处理视频课程（异步）
import asyncio

result = asyncio.run(notely.process("lecture.mp4"))

# 保存笔记
result.save("notes.md")
```

```python
# 方法 2：从字典创建（更简单）
notely = Notely.from_dict({
    "llm": {
        "api_key": "sk-xxx",
        "model": "gpt-4o",
    }
})

result = asyncio.run(notely.process("lecture.mp4"))
result.save("notes.md")
```

```python
# 方法 3：从 YAML 文件创建（推荐用于复杂配置）
# 先创建 config.yaml，然后：
notely = Notely.from_yaml("config.yaml")
result = asyncio.run(notely.process("lecture.mp4"))
```

### 3. 使用流程

<p align="center">
  <img src="docs/images/usage-flow.png" alt="使用流程" width="600">
</p>

**输出示例：**

```markdown
# 机器学习基础

> 📌 课程信息：45 分钟 | 讲师：张教授

## 📌 课程概述

本节课介绍机器学习的基本概念...

## 📚 核心知识点

### 什么是机器学习

**机器学习**是一种让计算机从数据中学习的技术...

### 机器学习的类型

| 类型 | 特点 | 应用场景 |
|------|------|---------|
| **监督学习** | 有标签数据 | 分类、回归 |
| **无监督学习** | 无标签数据 | 聚类、降维 |
| **强化学习** | 环境反馈 | 游戏、机器人 |

## 💡 关键要点

1. 机器学习是 AI 的核心技术
2. 算法选择取决于数据类型和任务
3. **特征工程**对模型性能至关重要
```

---

## 详细使用指南

### 初始化配置

#### 方式 1：基本使用

```python
import os
from notely import Notely

# 显式传入 API Key
notely = Notely(api_key="sk-xxx")

# 或从环境变量读取
notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))
```

#### 方式 2：切换 LLM 提供商

```python
import os

# 使用智谱 AI
notely = Notely(
    api_key=os.getenv("ZHIPU_API_KEY"),
    provider="zhipu",
    model="glm-4",
)

# 使用 Anthropic
notely = Notely(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    provider="anthropic",
    model="claude-3-opus-20240229",
)

# 使用 Moonshot
notely = Notely(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    provider="moonshot",
    model="moonshot-v1-8k",
)
```

#### 方式 3：使用自定义 OpenAI 兼容端点

```python
notely = Notely(
    api_key="sk-xxx",
    provider="custom",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

#### 方式 4：完整配置

```python
import os

notely = Notely(
    # LLM 配置
    api_key=os.getenv("OPENAI_API_KEY"),
    provider="openai",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",  # 可选
    temperature=0.3,  # 降低以提高一致性（默认：0.3）
    max_tokens=4096,

    # ASR 配置
    asr_backend="funasr",  # 中文推荐 funasr，多语言用 whisper
    asr_device="cuda",     # 有 GPU 用 cuda，否则用 cpu
    asr_model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",

    # OCR 配置
    ocr_backend="paddleocr",
    ocr_lang="ch",  # 中文用 ch，英文用 en

    # 增强配置（新增）
    chunk_size=2000,        # 最大文本块大小（tokens）（默认：2000）
    chunk_overlap=1000,     # 文本块重叠大小（tokens）（默认：1000）
    language=None,          # 输出语言：'zh'、'en' 或 None 自动检测

    # 处理配置
    key_frame_interval_seconds=5.0,  # 关键帧提取间隔
    min_frame_similarity=0.85,       # 帧去重相似度阈值

    # 其他配置
    template="academic",  # 笔记模板：academic, technical, meeting
    verbose=True,         # 显示详细日志
)
```

### 支持的 LLM 提供商

| 提供商 | Provider 值 | 推荐模型 |
|--------|------------|---------|
| OpenAI | `openai` | gpt-4o, gpt-4-turbo |
| 智谱 AI | `zhipu` | glm-4, glm-4-plus |
| Anthropic | `anthropic` | claude-3-opus, claude-3-sonnet |
| Moonshot | `moonshot` | moonshot-v1-8k, moonshot-v1-32k |
| DeepSeek | `deepseek` | deepseek-chat |
| 自定义 | `custom` | 任何 OpenAI 兼容 API |

### 处理不同输入格式

#### 处理视频

```python
# 基本用法
result = notely.process(
    video_path="lecture.mp4",
    title="课程标题",
)

# 带 PDF 课件
result = notely.process(
    video_path="lecture.mp4",
    pdf_paths=["slides.pdf", "handout.pdf"],
    title="深度学习基础",
    instructor="李教授",
    date="2026-03-03",
)
```

#### 处理音频

```python
# 方式 1：使用 process_audio
result = notely.process_audio(
    audio_path="podcast.mp3",
    title="技术播客第 42 期",
)

# 方式 2：使用 process
result = notely.process(
    audio_path="recording.wav",
    title="会议录音",
)
```

#### 处理 PDF

```python
result = notely.process_pdf(
    pdf_path="presentation.pdf",
    title="产品发布会",
)
```

### 自定义笔记模板

```python
from notely.prompts import NoteTemplate

# 使用内置模板
notely = Notely(template="academic")  # 学术风格
notely = Notely(template="technical") # 技术风格
notely = Notely(template="meeting")   # 会议记录

# 自定义模板
template = NoteTemplate(
    name="meeting",
    language="zh",
    style="casual",
    include_timestamps=True,
    include_transcript=False,
    custom_sections=["待办事项", "决策记录"],
)

result = notely.process(
    video_path="meeting.mp4",
    template=template,
)
```

### 访问处理结果

```python
result = notely.process("lecture.mp4")

# 获取 Markdown 内容
print(result.markdown)

# 获取转录文本
print(result.transcript.full_text)
print(f"时长: {result.transcript.duration:.1f} 秒")
print(f"片段数: {len(result.transcript.segments)}")

# 获取 OCR 结果
for ocr_result in result.ocr_results:
    print(ocr_result.full_text)

# 获取元数据
print(result.metadata)

# 保存到文件
result.save("output/notes.md")
```

---

## 工作原理

### 处理流程

<p align="center">
  <img src="docs/images/pipeline.png" alt="处理流程" width="700">
</p>

### 架构概览

Notely 使用三层增强流水线将原始转录文本转换为结构化笔记：

**1. 理解层（Comprehension）** - 从转录文本块中提取语义信息
   - 每个文本块摘要最少 300 字
   - 保留所有技术细节、公式和示例
   - 并发处理提升效率

**2. 组织层（Structuring）** - 将理解结果组织成连贯的章节
   - 每个主要章节最少 200 字
   - 按主题组织（非时间顺序）
   - 跨文本块概念合并

**3. 格式化层（Formatting）** - 美化 Markdown 并支持 LaTeX 公式
   - 数学符号渲染
   - 统一标题层级
   - Emoji 图标增强视觉效果

<p align="center">
  <img src="docs/images/architecture.png" alt="架构概览" width="800">
</p>

**关键步骤：**

1. **输入处理** - 从视频提取音频和关键帧
2. **ASR 转录** - 语音转文字，带时间戳（中文用 FunASR，多语言用 Whisper）
3. **OCR 识别** - 使用 PaddleOCR 提取幻灯片/画面中的文字
4. **语义分块** - 将转录文本切分为 2000 token 的块，1000 token 重叠
5. **理解层处理** - 从每个文本块提取语义信息（并行处理）
6. **组织层处理** - 将所有文本块按主题组织成连贯章节
7. **格式化输出** - 美化 Markdown 并支持 LaTeX 公式

---

## 常见问题

### 1. 如何选择 ASR 后端？

- **中文内容**：推荐 `funasr`（准确率更高，CER < 3%）
- **多语言内容**：使用 `whisper`（支持 99+ 语言）
- **无 GPU**：使用 `whisper` + `asr_device="cpu"`

```python
# 中文课程
notely = Notely(asr_backend="funasr", asr_device="cuda")

# 英文课程
notely = Notely(asr_backend="whisper", asr_device="cpu")
```

### 2. 如何降低成本？

- 使用更便宜的模型：`gpt-4o-mini`、`glm-4-flash`
- 调整 `max_tokens` 限制输出长度
- 使用国产大模型（智谱、Moonshot、DeepSeek）

```python
notely = Notely(
    provider="zhipu",
    model="glm-4-flash",  # 更便宜
    max_tokens=2048,      # 限制输出
)
```

### 3. 处理速度慢怎么办？

- 使用 GPU 加速：`asr_device="cuda"`
- 增大关键帧间隔：`key_frame_interval_seconds=10.0`
- 提高帧相似度阈值：`min_frame_similarity=0.90`

```python
notely = Notely(
    asr_device="cuda",
    key_frame_interval_seconds=10.0,
    min_frame_similarity=0.90,
)
```

### 4. 如何处理长视频？

Notely 会自动处理长视频，但建议：
- 确保有足够的内存和磁盘空间
- 使用 GPU 加速
- 考虑分段处理（手动切分视频）

### 5. 支持哪些视频格式？

支持 FFmpeg 支持的所有格式：
- 视频：mp4, avi, mov, mkv, flv, wmv, webm
- 音频：mp3, wav, m4a, flac, aac, ogg

---

## 项目结构

```
notely/
├── src/notely/
│   ├── __init__.py          # 主入口
│   ├── core.py              # 核心逻辑
│   ├── asr/                 # ASR 后端
│   │   ├── funasr.py        # FunASR
│   │   └── whisper.py       # Whisper
│   ├── ocr/                 # OCR 后端
│   │   └── paddle.py        # PaddleOCR
│   ├── llm/                 # LLM 后端
│   │   └── openai.py        # OpenAI 兼容
│   ├── prompts/             # 笔记模板
│   │   ├── templates/       # 模板文件
│   │   └── loader.py        # 模板加载器
│   ├── formatter/           # Markdown 格式化
│   └── utils/               # 工具函数
├── examples/                # 示例代码
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── pyproject.toml
```

---

## 开发指南

### 安装开发环境

```bash
# 克隆仓库
git clone https://github.com/0xarcher/notely.git
cd notely

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装所有依赖
uv sync --all-extras

# 安装 FFmpeg
brew install ffmpeg  # macOS
```

### 代码规范

```bash
# 格式化代码
uv run ruff format .

# 检查代码
uv run ruff check .

# 自动修复
uv run ruff check --fix .
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_core.py

# 生成覆盖率报告
uv run pytest --cov=notely --cov-report=html
```

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

**快速开始：**

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 致谢

Notely 基于以下优秀的开源项目构建：

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里巴巴 ASR 工具包
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度 OCR 工具包
- [Whisper](https://github.com/openai/whisper) - OpenAI 语音识别模型
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF 文本提取

---

## 联系方式

- GitHub: [@0xarcher](https://github.com/0xarcher)
- Email: coder.archer@gmail.com
- Issues: [GitHub Issues](https://github.com/0xarcher/notely/issues)

---

<p align="center">
  <strong>Made with ❤️ by Archer</strong>
</p>
