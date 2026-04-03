import markdown
import os
from playwright.sync_api import sync_playwright

def convert_md_to_pdf(md_filepath, pdf_filepath):
    """
    使用 Playwright 将 Markdown 文件转换为 PDF 文件
    """
    if not os.path.exists(md_filepath):
        print(f"错误: 找不到文件 {md_filepath}")
        return

    # 1. 读取 Markdown 文件内容
    with open(md_filepath, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 2. 将 Markdown 转换为 HTML
    html_content = markdown.markdown(
        md_text, 
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )

    # 3. 定义 CSS 样式 (GitHub 风格)
    css_style = """
    body { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
        line-height: 1.6; 
        color: #333; 
        max-width: 100%;
        margin: 0;
    }
    h1, h2, h3, h4, h5 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    code { 
        background-color: #f6f8fa; 
        padding: 0.2em 0.4em; 
        border-radius: 3px; 
        font-family: monospace;
    }
    pre { 
        background-color: #f6f8fa; 
        padding: 16px; 
        border-radius: 6px; 
        overflow-x: auto; 
    }
    pre code { background-color: transparent; padding: 0; }
    blockquote { 
        border-left: 0.25em solid #dfe2e5; 
        color: #6a737d; 
        padding: 0 1em; 
        margin: 0;
    }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
    table th, table td { 
        border: 1px solid #dfe2e5; 
        padding: 6px 13px; 
    }
    table tr:nth-child(2n) { background-color: #f6f8fa; }
    img { max-width: 100%; box-sizing: content-box; }
    """

    # 4. 拼接完整的 HTML
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Document</title>
        <style>{css_style}</style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # 5. 将 HTML 写入临时文件 (关键步骤：这样可以正确加载本地相对路径的图片)
    temp_html_path = "temp_render.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    # 6. 使用 Playwright 渲染并生成 PDF
    print(f"正在转换: {md_filepath} -> {pdf_filepath} ...")
    try:
        with sync_playwright() as p:
            # 启动无头 Chromium
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # 将本地路径转换为 file:/// 格式的 URL 供浏览器加载
            absolute_path = os.path.abspath(temp_html_path).replace("\\", "/")
            file_url = f"file:///{absolute_path}"
            
            # 加载 HTML
            page.goto(file_url)
            
            # 导出为 PDF
            page.pdf(
                path=pdf_filepath, 
                format="A4", 
                margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"},
                print_background=True # 保留代码块和表格的背景色
            )
            browser.close()
            print("✅ 转换完成！")
    except Exception as e:
        print(f"转换失败: {e}")
    finally:
        # 7. 清理临时 HTML 文件
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)

# ================= 运行示例 =================
if __name__ == "__main__":
    input_md = "D:\\Workspace\\MatMoE\\MatAgent\\logs\\reports\\MatReport_20260327_144824.md"  # 这里改成你要转换的 MD 文件名
    output_pdf = "深层物理常识的自主触发.pdf"
    
    convert_md_to_pdf(input_md, output_pdf)
