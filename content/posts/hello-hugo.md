---
title: "用 Hugo + GitHub Pages 搭建个人博客"
date: 2026-08-07T23:50:00+08:00
draft: false
categories: ["教程"]
tags: ["Hugo", "GitHub Pages", "博客"]
summary: "从零开始，用 Hugo 静态站点生成器 + GitHub Pages 搭建一个免费、快速、可版本控制的个人技术博客。"
ShowToc: true
TocOpen: true
---

## 为什么选择 Hugo？

市面上博客框架很多（Hexo、Jekyll、Astro……），但 Hugo 有几个明显优势：

| 特性 | Hugo | Hexo | Jekyll |
|------|------|------|--------|
| 构建速度 | ⚡ 极快（Go 编译） | 中等（Node.js） | 慢（Ruby） |
| 依赖 | 单个二进制文件 | npm 生态 | bundler |
| 主题数量 | 1000+ | 300+ | 200+ |
| 学习成本 | 低 | 低 | 中 |

**一句话总结**：Hugo 是构建速度最快、依赖最少的静态站点生成器。

---

## 前置准备

你需要：

1. **一个 GitHub 账号**（你已经有了）
2. **安装 Hugo**（本地可选，部署在 CI 上自动构建也行）
3. **基本的命令行操作能力**

### 安装 Hugo

```bash
# macOS
brew install hugo

# Linux (Debian/Ubuntu)
sudo apt install hugo

# Windows
scoop install hugo
```

验证安装：

```bash
hugo version
```

---

## 第一步：创建站点

```bash
hugo new site my-blog
cd my-blog
```

目录结构如下：

```
my-blog/
├── archetypes/     # 文章模板
├── content/        # Markdown 内容
├── layouts/        # 布局模板
├── static/         # 静态资源（图片等）
├── themes/         # 主题
├── config.toml     # 站点配置
└── data/           # 数据文件
```

---

## 第二步：安装主题

推荐 **PaperMod**——简洁、快速、功能完备：

```bash
git init
git submodule add https://github.com/adityatelange/hugo-PaperMod themes/PaperMod
```

在 `config.toml` 中启用：

```toml
theme = "PaperMod"
```

> **为什么要用 `git submodule`？**
> 因为主题本身是一个独立的 Git 仓库。用 submodule 可以在不动主题代码的前提下保持版本追踪，也方便 CI 克隆时自动拉取。

---

## 第三步：写第一篇文章

```bash
hugo new posts/hello-hugo.md
```

这会在 `content/posts/hello-hugo.md` 生成一个带 Front Matter 的模板文件：

```markdown
+++
title: 'Hello Hugo'
date: 2026-08-07T23:50:00+08:00
draft: true
+++

在这里写正文……
```

### Front Matter 是什么？

Front Matter 是文章开头的元数据块，用 `+++`（TOML）或 `---`（YAML）包裹。常用字段：

| 字段 | 作用 | 示例 |
|------|------|------|
| `title` | 文章标题 | `"我的第一篇博客"` |
| `date` | 发布时间 | `2026-08-07` |
| `draft` | 是否草稿 | `true` / `false` |
| `tags` | 标签 | `["Hugo", "教程"]` |
| `categories` | 分类 | `["Web开发"]` |
| `summary` | 摘要 | `"一篇简介"` |

> ⚠️ `draft: true` 的文章不会被部署！发布前记得改成 `false`。

---

## 第四步：Markdown 写作基础

博客内容用 Markdown 编写，以下是常用语法：

### 标题

```markdown
# 一级标题
## 二级标题
### 三级标题
```

### 文本样式

```markdown
**粗体**  *斜体*  ~~删除线~~  `行内代码`
```

效果：**粗体** *斜体* ~~删除线~~ `行内代码`

### 列表

```markdown
- 无序列表项 1
- 无序列表项 2

1. 有序列表项 1
2. 有序列表项 2
```

### 代码块

使用三个反引号 + 语言名实现语法高亮：

````markdown
```python
def hello():
    print("Hello, Hugo!")
```
````

效果：

```python
def hello():
    print("Hello, Hugo!")
```

### 表格

```markdown
| 名称   | 类型   | 说明       |
|--------|--------|------------|
| Hugo   | 工具   | SSG 框架   |
| TOML   | 配置   | 简洁可读   |
```

### 引用与提示

```markdown
> 这是一段引用文字。

> [!TIP]
> Hugo 支持 GitHub 风格的提示框（需主题适配）。
```

### 链接与图片

```markdown
[链接文字](https://gohugo.io)

![图片描述](/images/screenshot.png)
```

> 💡 图片放在 `static/images/` 目录下，引用时路径为 `/images/xxx.png`。

---

## 第五步：本地预览

```bash
hugo server -D
```

`-D` 参数表示包含草稿文章。浏览器打开 `http://localhost:1313` 即可实时预览。

修改任何文件都会自动热刷新。

---

## 第六步：部署到 GitHub Pages

### 6.1 创建仓库

仓库名必须为 **`<你的用户名>.github.io`**。

例如：`yukki3421.github.io`

### 6.2 添加 GitHub Actions 自动部署

在仓库中创建 `.github/workflows/hugo.yml`：

```yaml
name: Deploy Hugo to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

defaults:
  run:
    shell: bash

steps:
  - name: Checkout
    uses: actions/checkout@v4
    with:
      fetch-depth: 0
      submodules: true

  - name: Setup Hugo
    uses: peaceiris/actions-hugo@v3
    with:
      hugo-version: latest
      extended: true

  - name: Setup Pages
    id: pages
    uses: actions/configure-pages@v5

  - name: Build with Hugo
    env:
      HUGO_CACHEDIR: ${{ runner.temp }}/hugo_cache
      HUGO_ENVIRONMENT: production
      TZ: Asia/Shanghai
    run: hugo --minify --baseURL "${{ steps.pages.outputs.base_url }}/"

  - name: Upload artifact
    uses: actions/upload-pages-artifact@v3
    with:
      path: ./public

  - name: Deploy to GitHub Pages
    id: deployment
    uses: actions/deploy-pages@v4
```

### 6.3 配置 GitHub Pages

进入仓库 **Settings → Pages → Build and deployment**，Source 选择 **GitHub Actions**。

### 6.4 推送代码

```bash
git add .
git commit -m "Initial Hugo site"
git push origin main
```

推送后 GitHub Actions 会自动构建，几分钟后访问：

```
https://<你的用户名>.github.io
```

---

## 日常使用流程

博客搭好之后，日常写文章只需要三步：

```bash
# 1. 新建文章
hugo new posts/my-new-post.md

# 2. 写内容（编辑 Markdown 文件，把 draft 改成 false）

# 3. 推送部署
git add .
git commit -m "新文章：xxx"
git push
```

CI 会自动完成构建 + 部署。

---

## 常见问题

### Q: 文章发布后没显示？

检查 `draft` 是否设为 `false`，`date` 是否是未来时间（Hugo 默认不发布未来日期的文章）。

### Q: 如何自定义域名？

在仓库根目录添加 `static/CNAME` 文件，写入你的域名，然后在域名 DNS 添加 CNAME 记录指向 `<用户名>.github.io`。

### Q: 主题怎么更新？

```bash
git submodule update --remote --merge
```

### Q: 如何添加评论系统？

PaperMod 支持 Utterances / Giscus，在 `config.toml` 或 `layouts/partials/` 中配置即可。两者都基于 GitHub Issues/Discussions，无需后端。

---

## 总结

| 步骤 | 命令/操作 |
|------|-----------|
| 创建站点 | `hugo new site my-blog` |
| 安装主题 | `git submodule add <主题仓库>` |
| 写文章 | `hugo new posts/xxx.md` |
| 本地预览 | `hugo server -D` |
| 部署 | `git push`（CI 自动构建） |

Hugo + GitHub Pages 的组合完全免费、版本可控、构建极快。开始写你的第一篇技术博客吧！
