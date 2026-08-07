# yukki3421.github.io

基于 **Hugo + PaperMod** 主题的个人技术博客，通过 GitHub Actions 自动部署到 GitHub Pages。

👉 访问地址：https://yukki3421.github.io

## 本地开发

```bash
# 安装 Hugo Extended
# https://gohugo.io/installation/

# 克隆（含主题 submodule）
git clone --recurse-submodules https://github.com/yukki3421/yukki3421.github.io.git
cd yukki3421.github.io

# 本地预览（含草稿）
hugo server -D

# 浏览器打开 http://localhost:1313
```

## 写新文章

```bash
hugo new posts/your-post-name.md
# 编辑 Markdown 文件，完成后把 draft 改成 false
git add .
git commit -m "新文章：xxx"
git push   # CI 自动构建部署
```

## 目录结构

```
content/posts/     # 博客文章（Markdown）
static/            # 图片等静态资源
themes/PaperMod/   # 主题（git submodule）
config.toml        # 站点配置
```

## 技术栈

- [Hugo](https://gohugo.io/) — 静态站点生成器
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) — 主题
- GitHub Actions — CI/CD
- GitHub Pages — 免费托管
