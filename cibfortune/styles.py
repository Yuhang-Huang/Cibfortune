EDITABLE_TABLE_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR 识别结果预览</title>
        <style>
            body {{
                font-family: "Microsoft YaHei", Arial, sans-serif;
                padding: 40px;
                background-color: #f4f4f4;
                display: flex;
                justify-content: center;
            }}
            
            /* 表格容器样式 */
            .table-container {{
                background-color: white;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow-x: auto; /* 防止表格过宽溢出 */
            }}

            /* 核心表格样式 */
            table.ocr-result-table {{
                border-collapse: collapse; /* 合并边框，必须有 */
                margin: 0 auto;
                /* 如果你想覆盖原始的 width，可以在这里加 !important，否则保留原始宽度 */
            }}

            /* 单元格样式 */
            table.ocr-result-table td, table.ocr-result-table th {{
                border: 1px solid #333; /* 实线边框 */
                padding: 8px 12px;
                text-align: center;
                vertical-align: middle;
                font-size: 14px;
                min-width: 60px; /* 最小宽度防止太挤 */
            }}

            /* 针对可编辑区域 (contenteditable="true") 的样式优化 */
            [contenteditable="true"] {{
                background-color: #eef7ff; /*以此颜色标识可编辑区域 */
                color: #0056b3;
                cursor: text;
                transition: background-color 0.2s;
            }}

            [contenteditable="true"]:focus {{
                background-color: #fff;
                outline: 2px solid #2196F3; /* 聚焦时的高亮边框 */
                box-shadow: 0 0 5px rgba(33, 150, 243, 0.5);
            }}
            
            /* 表头/标签列的样式 (不可编辑部分) */
            td:not([contenteditable="true"]) {{
                background-color: #fafafa;
                font-weight: bold;
                color: #555;
            }}
        </style>
    </head>
    <body>

        <div class="table-container">
            <h3>{title}</h3>
            {html_content}
        </div>

    </body>
    </html>
    """

TOUCH_CSS = """
:root {
    --radius-lg: 22px;
    --radius-md: 14px;
    --surface: #ffffff;
    --surface-muted: #f5f7fb;
    --surface-border: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.12);
}
body {
    background: linear-gradient(135deg, #eef2ff 0%, #f9fafc 55%, #ffffff 100%);
    color: var(--text-primary);
}
.gradio-container {
    max-width: 1650px !important;
    margin: 0 auto;
    padding: 20px 24px 48px;
    font-size: 16px;
    color: var(--text-primary);
}
.gradio-container .gr-markdown {
    color: var(--text-primary);
}
#unified-header {
    background: linear-gradient(130deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(37, 99, 235, 0.18);
    padding: 24px 28px;
    border-radius: 28px;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}
#unified-header h1 {
    margin: 0 0 6px;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
#unified-header p {
    margin: 0;
    color: var(--text-secondary);
}
#unified-mode-bar {
    background: var(--surface);
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.06);
    padding: 20px 22px;
    gap: 18px;
    margin-bottom: 20px;
    border: 1px solid var(--surface-border);
}
#unified-mode-bar .gradio-button,
#unified-mode-bar button {
    font-size: 16px !important;
    padding: 12px 18px !important;
    border-radius: 14px !important;
}
#unified-mode-bar textarea,
#unified-mode-bar input[type="text"] {
    background: var(--surface);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 14px;
    color: var(--text-primary);
    box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.04);
}
#unified-mode-bar textarea:focus,
#unified-mode-bar input[type="text"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}
.gradio-container .tabs {
    background: transparent;
    border: none;
}
.gradio-container .tabitem {
    border-radius: var(--radius-md);
    background: #f8fafc;
    border: 1px solid transparent;
    color: var(--text-secondary);
}
.gradio-container .tabitem.selected {
    border-color: rgba(37, 99, 235, 0.25);
    color: var(--text-primary);
    background: #ffffff;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.08);
}
#unified-input-panel,
#unified-chat-panel,
#unified-batch-panel,
#unified-compare-panel {
    background: var(--surface);
    border-radius: 24px;
    padding: 22px 24px;
    box-shadow: 0 22px 44px rgba(15, 23, 42, 0.06);
    border: 1px solid var(--surface-border);
}
#unified-input-panel .gradio-slider > label,
#unified-input-panel .gradio-dropdown > label {
    color: var(--text-secondary);
}
#unified-chat-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}
#unified-chatbot > .wrap {
    background: #f8fafc;
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    padding: 8px 10px;
}
#unified-chatbot .message {
    border-radius: 16px !important;
    padding: 12px 14px !important;
    line-height: 1.6;
    font-size: 15px;
    color: var(--text-primary);
}
#unified-chatbot .message.user {
    background: linear-gradient(138deg, rgba(37, 99, 235, 0.16), rgba(96, 165, 250, 0.12));
    border: 1px solid rgba(37, 99, 235, 0.22);
    color: var(--text-primary);
    align-self: flex-end;
}
#unified-chatbot .message.bot {
    background: #ffffff;
    border: 1px solid rgba(203, 213, 225, 0.9);
    color: var(--text-primary);
    align-self: flex-start;
}
#unified-query textarea {
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: var(--surface);
    color: var(--text-primary);
    box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.05);
}
#unified-query textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}
#unified-stats .stats-text {
    background: var(--accent-soft);
    border-radius: 16px;
    border: 1px solid rgba(37, 99, 235, 0.2);
    color: var(--text-primary);
    font-weight: 500;
    padding: 12px 14px;
    line-height: 1.6;
    margin-bottom: 12px;
    word-break: break-word;
}
.gradio-container .gradio-button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border: none;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 18px 30px rgba(37, 99, 235, 0.22);
}
.gradio-container .gradio-button.primary:hover {
    filter: brightness(1.03);
}
.gradio-container .gradio-button.secondary {
    background: rgba(37, 99, 235, 0.1);
    border: 1px solid rgba(37, 99, 235, 0.18);
    color: var(--text-primary);
}
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--surface);
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: var(--text-primary);
    border-radius: 16px;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}
#unified-batch-panel textarea,
#unified-compare-panel textarea {
    min-height: 320px;
}
.gradio-container .dropdown span.label,
.gradio-container .slider > label,
.gradio-container .dropdown label {
    color: var(--text-secondary);
}
.gradio-container .gradio-dropdown .wrap select,
.gradio-container .gradio-dropdown .wrap button {
    background: var(--surface);
    color: var(--text-primary);
    border-color: rgba(148, 163, 184, 0.4);
}
.gradio-container .gradio-dropdown .wrap select:focus,
.gradio-container .gradio-dropdown .wrap button:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}
/*
Bigger markdown preview area for unified stats (OCR/table preview)
*/
#unified-stats {
    max-height: 560px;
    overflow: auto;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 12px 14px;
    border-radius: 14px;
    background: #ffffff;
}
#unified-stats table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 14px;
}
#unified-stats th,
#unified-stats td {
    border: 1px solid #e5e7eb;
    padding: 8px 10px;
    text-align: left;
    vertical-align: top;
    font-size: 14px;
    line-height: 1.55;
}
#unified-stats thead th {
    background: #f8fafc;
    font-weight: 600;
}
#unified-stats code {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 1px 4px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
/* 字段表格样式 */
.gradio-container .dataframe {
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    overflow: hidden;
}
.gradio-container .dataframe table {
    width: 100%;
    border-collapse: collapse;
}
.gradio-container .dataframe th {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(59, 130, 246, 0.08));
    color: var(--text-primary);
    font-weight: 600;
    padding: 10px 12px;
    border-bottom: 2px solid rgba(37, 99, 235, 0.2);
}
.gradio-container .dataframe td {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
.gradio-container .dataframe tr:hover {
    background: rgba(37, 99, 235, 0.04);
}
.gradio-container .dataframe input[type="text"] {
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 6px;
    padding: 4px 8px;
    width: 100%;
}
.gradio-container .dataframe input[type="text"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
}
"""

BILL_FINAL_RESULT_TABLE_TEMPLATE = """
    <style>
    /* 可调整大小的表格容器 */
    .ocr-result-table-container {{
        position: relative;
        display: inline-block;
        min-width: 500px;
        min-height: 300px;
        max-width: 95vw;
        max-height: 90vh;
        width: 100%;
        height: 600px;
        resize: both;
        overflow: auto;  /* 允许滚动，确保表格不超出容器 */
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }}
    /* 调整大小手柄样式 */
    .ocr-result-table-container::-webkit-resizer {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 0 8px 0;
        width: 20px;
        height: 20px;
    }}
    /* 调整大小提示 */
    .ocr-result-table-container::before {{
        content: '↘ 拖拽调整大小';
        position: absolute;
        top: 5px;
        right: 5px;
        font-size: 11px;
        color: #667eea;
        background: rgba(255, 255, 255, 0.9);
        padding: 2px 6px;
        border-radius: 4px;
        pointer-events: none;
        opacity: 0.7;
        z-index: 5;
        transition: opacity 0.3s ease;
    }}
    .ocr-result-table-container:hover::before {{
        opacity: 1;
    }}
    /* 调整大小时的边框高亮 */
    .ocr-result-table-container:active {{
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}
    .ocr-result-table {{
        width: auto;  /* 表格宽度根据内容自适应 */
        min-width: 100%;  /* 最小宽度为容器宽度 */
        max-width: 100%;  /* 最大宽度不超过容器 */
        border-collapse: collapse;
        margin: 0;
        font-size: 14px;
        table-layout: auto;  /* 使用auto，让列宽根据内容自动调整 */
        box-shadow: none;
        border-radius: 8px;
        overflow: visible;  /* 允许内容溢出，不裁剪 */
        background-color: #ffffff;
    }}
    .ocr-result-table th,
    .ocr-result-table td {{
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        text-align: left;
        vertical-align: top;
        word-break: break-word;
        word-wrap: break-word;
        transition: all 0.2s ease;
        line-height: 1.6;
        height: auto !important;  /* 行高根据内容自动调整，覆盖HTML中的固定height */
        min-height: auto !important;
        overflow: visible;  /* 允许内容显示，不裁剪 */
        width: auto;  /* 列宽根据内容自动调整 */
        max-width: none;  /* 不限制最大宽度 */
    }}
    /* 字段名列：根据内容自适应宽度 */
    .ocr-result-table td:not([contenteditable="true"]) {{
        background-color: #f8f9fa;
        font-weight: 600;
        color: #374151;
        width: auto;  /* 宽度根据内容自适应 */
        min-width: 120px;  /* 最小宽度 */
        max-width: 300px;  /* 最大宽度限制，避免过宽 */
        white-space: nowrap;  /* 字段名不换行 */
        font-size: 14px;
        border-right: 2px solid #d1d5db;
        height: auto !important;
        overflow: visible;
    }}
    /* 值列：根据内容自适应宽度 */
    .ocr-result-table td[contenteditable="true"] {{
        background-color: #ffffff;
        cursor: text;
        min-height: 20px;
        height: auto !important;  /* 行高根据内容自动调整 */
        position: relative;
        width: auto;  /* 宽度根据内容自适应 */
        min-width: 200px;  /* 最小宽度 */
        max-width: none;  /* 不限制最大宽度，允许长文本 */
        overflow: visible;  /* 允许内容显示 */
        word-break: break-word;  /* 长文本自动换行 */
    }}
    /* 根据文本长度动态调整样式（保持列宽比例） */
    .ocr-result-table td[contenteditable="true"][data-length="short"] {{
        font-size: 15px;
        padding: 10px 14px;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="medium"] {{
        font-size: 14px;
        padding: 12px 16px;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="long"] {{
        font-size: 13px;
        padding: 14px 18px;
        line-height: 1.7;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="very-long"] {{
        font-size: 12px;
        padding: 16px 20px;
        line-height: 1.8;
        height: auto !important;
    }}
    .ocr-result-table th {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-color: #5568d3;
    }}
    .ocr-result-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .ocr-result-table tr:nth-child(odd) {{
        background-color: #ffffff;
    }}
    .ocr-result-table tr:hover {{
        background-color: #f0f4ff;
    }}
    .ocr-result-table tr:hover td:not([contenteditable="true"]) {{
        background-color: #e5e7eb;
    }}
    .ocr-result-table td[contenteditable="true"]:hover {{
        background-color: #f8f9ff;
        box-shadow: inset 0 0 0 1px #667eea;
    }}
    .ocr-result-table td[contenteditable="true"]:focus {{
        outline: none;
        background-color: #eef5ff;
        box-shadow: inset 0 0 0 2px #667eea, 0 0 0 3px rgba(102, 126, 234, 0.1);
        border-radius: 4px;
    }}
    .ocr-result-table td[contenteditable="true"]:empty:before {{
        content: "点击编辑...";
        color: #999;
        font-style: italic;
    }}
    .ocr-result-table td[contenteditable="true"]:empty:focus:before {{
        content: "";
    }}
    /* 优化长文本显示 */
    .ocr-result-table td[contenteditable="true"] {{
        overflow-wrap: break-word;
        hyphens: auto;
    }}
    /* 响应式设计 */
    @media (max-width: 768px) {{
        .ocr-result-table-container {{
            min-width: 300px;
            min-height: 200px;
        }}
        .ocr-result-table {{
            font-size: 12px;
            table-layout: fixed;
        }}
        .ocr-result-table th,
        .ocr-result-table td {{
            padding: 8px 12px;
        }}
        .ocr-result-table td:not([contenteditable="true"]) {{
            width: 30%;
            font-size: 12px;
        }}
        .ocr-result-table td[contenteditable="true"] {{
            width: 70%;
        }}
    }}
    </style>
    <script>
    (function() {{
        // 移除所有固定的height和width属性，让行高和列宽根据内容自动调整
        function removeFixedHeights() {{
            var table = document.querySelector('.ocr-result-table');
            if (table) {{
                // 移除table的width属性
                if (table.hasAttribute('width')) {{
                    table.removeAttribute('width');
                }}
                if (table.style.width) {{
                    table.style.width = '';
                }}
                
                // 移除tr的height和width属性
                var rows = table.querySelectorAll('tr');
                rows.forEach(function(row) {{
                    if (row.hasAttribute('height')) {{
                        row.removeAttribute('height');
                    }}
                    if (row.hasAttribute('width')) {{
                        row.removeAttribute('width');
                    }}
                }});
                
                // 移除td和th的height和width属性
                var cells = table.querySelectorAll('td, th');
                cells.forEach(function(cell) {{
                    if (cell.hasAttribute('height')) {{
                        cell.removeAttribute('height');
                    }}
                    if (cell.hasAttribute('width')) {{
                        cell.removeAttribute('width');
                    }}
                    // 移除内联样式中的height和width
                    if (cell.style.height) {{
                        cell.style.height = '';
                    }}
                    if (cell.style.width) {{
                        cell.style.width = '';
                    }}
                }});
                
                // 移除colgroup中的width属性
                var colgroups = table.querySelectorAll('colgroup');
                colgroups.forEach(function(colgroup) {{
                    var cols = colgroup.querySelectorAll('col');
                    cols.forEach(function(col) {{
                        if (col.hasAttribute('width')) {{
                            col.removeAttribute('width');
                        }}
                        if (col.style.width) {{
                            col.style.width = '';
                        }}
                    }});
                }});
            }}
        }}
        
        // 根据文本长度动态设置data-length属性
        function updateCellLength() {{
            var cells = document.querySelectorAll('.ocr-result-table td[contenteditable="true"]');
            cells.forEach(function(cell) {{
                var text = cell.textContent || cell.innerText || '';
                var length = text.length;
                cell.removeAttribute('data-length');
                if (length > 0) {{
                    if (length <= 20) {{
                        cell.setAttribute('data-length', 'short');
                    }} else if (length <= 50) {{
                        cell.setAttribute('data-length', 'medium');
                    }} else if (length <= 100) {{
                        cell.setAttribute('data-length', 'long');
                    }} else {{
                        cell.setAttribute('data-length', 'very-long');
                    }}
                }}
            }});
        }}
        
        // 页面加载后执行
        setTimeout(function() {{
            removeFixedHeights();
            updateCellLength();
        }}, 100);
        
        // 监听内容变化
        var observer = new MutationObserver(function(mutations) {{
            removeFixedHeights();
            updateCellLength();
        }});
        
        setTimeout(function() {{
            var table = document.querySelector('.ocr-result-table');
            if (table) {{
                observer.observe(table, {{
                    childList: true,
                    subtree: true,
                    characterData: true,
                    attributes: true,
                    attributeFilter: ['height', 'style']
                }});
            }}
        }}, 200);
    }})();
    </script>
    <div class="ocr-result-table-container">
        {html_content}
    </div>
    <script>
    (function() {{
        var updateTimeout = null;
        
        function updateEditedContent() {{
            // 清除之前的定时器
            if (updateTimeout) {{
                clearTimeout(updateTimeout);
            }}
            
            // 延迟更新，避免频繁触发
            updateTimeout = setTimeout(function() {{
                var table = document.querySelector('.ocr-result-table');
                if (!table) return;
                
                // 获取完整的HTML（包括样式）
                var fullHtml = document.querySelector('#bill-ocr-result-html, [id*="bill-ocr-result-html"]');
                var htmlContent = '';
                
                if (fullHtml) {{
                    // 获取包含表格的完整HTML
                    var container = fullHtml.querySelector('.ocr-result-table') || fullHtml;
                    htmlContent = container.innerHTML;
                }} else {{
                    // 如果没有找到容器，直接获取表格的outerHTML
                    htmlContent = table.outerHTML;
                }}
                
                // 查找隐藏的Textbox - 使用多种方法
                var hiddenInput = null;
                
                // 方法1: 直接通过ID查找
                hiddenInput = document.getElementById('bill-ocr-result-html-edited');
                
                // 方法2: 通过ID包含关键字查找
                if (!hiddenInput) {{
                    var inputs = document.querySelectorAll('input, textarea');
                    for (var i = 0; i < inputs.length; i++) {{
                        if (inputs[i].id && inputs[i].id.includes('bill-ocr-result-html-edited')) {{
                            hiddenInput = inputs[i];
                            break;
                        }}
                    }}
                }}
                
                // 方法3: 通过name属性查找
                if (!hiddenInput) {{
                    hiddenInput = document.querySelector('input[name*="bill-ocr-result-html-edited"], textarea[name*="bill-ocr-result-html-edited"]');
                }}
                
                // 方法4: 通过data属性或class查找
                if (!hiddenInput) {{
                    var allInputs = document.querySelectorAll('input[type="text"], textarea');
                    for (var i = 0; i < allInputs.length; i++) {{
                        var input = allInputs[i];
                        // 检查是否在Gradio的隐藏组件区域
                        if (input.style.display === 'none' || input.hidden || input.offsetParent === null) {{
                            // 尝试设置值，看是否能找到正确的输入框
                            var testValue = input.value;
                            input.value = 'TEST_' + Date.now();
                            if (input.value === 'TEST_' + Date.now()) {{
                                input.value = testValue; // 恢复原值
                                // 这可能是我们要找的输入框，但需要更精确的匹配
                            }}
                        }}
                    }}
                }}
                
                if (hiddenInput) {{
                    // 获取完整的HTML内容（包括样式）
                    var styleTag = document.querySelector('style');
                    var styleContent = styleTag ? styleTag.outerHTML : '';
                    var fullContent = styleContent + '\\n' + table.outerHTML;
                    
                    hiddenInput.value = fullContent;
                    
                    // 触发多种事件，确保Gradio捕获到变化
                    var events = ['input', 'change', 'blur', 'keyup'];
                    events.forEach(function(eventType) {{
                        var event = new Event(eventType, {{ bubbles: true, cancelable: true }});
                        hiddenInput.dispatchEvent(event);
                    }});
                    
                    // 也尝试直接设置属性
                    if (hiddenInput.setAttribute) {{
                        hiddenInput.setAttribute('value', fullContent);
                    }}
                    
                    console.log('[DEBUG] 已更新隐藏Textbox，内容长度:', fullContent.length);
                }} else {{
                    console.warn('[DEBUG] 未找到隐藏的Textbox组件');
                    // 如果找不到，尝试通过window对象存储
                    if (window.gradioEditedContent === undefined) {{
                        window.gradioEditedContent = {{}};
                    }}
                    window.gradioEditedContent['bill-ocr-result-html-edited'] = htmlContent;
                }}
            }}, 300);
        }}
        
        // 监听所有可编辑单元格的输入事件
        function attachListeners() {{
            var editableCells = document.querySelectorAll('.ocr-result-table td[contenteditable="true"]');
            editableCells.forEach(function(cell) {{
                // 移除旧的监听器（如果存在）
                var newCell = cell.cloneNode(true);
                cell.parentNode.replaceChild(newCell, cell);
                
                // 添加新的监听器
                newCell.addEventListener('input', updateEditedContent);
                newCell.addEventListener('blur', updateEditedContent);
                newCell.addEventListener('keyup', updateEditedContent);
                newCell.addEventListener('paste', function() {{
                    setTimeout(updateEditedContent, 100);
                }});
            }});
            
            // 初始更新
            updateEditedContent();
        }}
        
        // 延迟执行，确保DOM已加载
        setTimeout(attachListeners, 500);
        
        // 使用MutationObserver监听表格变化（动态添加的单元格）
        var observer = new MutationObserver(function(mutations) {{
            var shouldReattach = false;
            mutations.forEach(function(mutation) {{
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {{
                    shouldReattach = true;
                }}
            }});
            if (shouldReattach) {{
                setTimeout(attachListeners, 100);
            }}
        }});
        
        setTimeout(function() {{
            var table = document.querySelector('.ocr-result-table');
            if (table) {{
                observer.observe(table, {{
                    childList: true,
                    subtree: true,
                    characterData: true
                }});
            }}
        }}, 500);
        
        // 页面卸载前保存
        window.addEventListener('beforeunload', updateEditedContent);
        
        // 监听导出按钮点击事件，在导出前强制更新内容
        function setupExportButton() {{
            var exportBtn = document.getElementById('bill-ocr-export-btn') || 
                            document.querySelector('button[id*="bill-ocr-export-btn"]') ||
                            document.querySelector('button:contains("导出结果")');
            
            if (exportBtn) {{
                exportBtn.addEventListener('click', function(e) {{
                    console.log('[DEBUG] 导出按钮被点击，强制更新内容...');
                    // 立即更新内容，不延迟
                    var table = document.querySelector('.ocr-result-table');
                    if (table) {{
                        var styleTag = document.querySelector('style');
                        var styleContent = styleTag ? styleTag.outerHTML : '';
                        // 获取编辑后的表格HTML（包含所有用户编辑的内容）
                        var tableHtml = table.outerHTML;
                        var fullContent = styleContent + '\\n' + tableHtml;
                        
                        console.log('[DEBUG] 获取到的表格HTML长度:', tableHtml.length);
                        console.log('[DEBUG] 表格内容预览:', tableHtml.substring(0, 200));
                        
                        // 查找隐藏的Textbox - 使用多种方法
                        var hiddenInput = null;
                        
                        // 方法1: 直接通过ID查找
                        hiddenInput = document.getElementById('bill-ocr-result-html-edited');
                        
                        // 方法2: 通过ID包含关键字查找
                        if (!hiddenInput) {{
                            var inputs = document.querySelectorAll('input, textarea');
                            for (var i = 0; i < inputs.length; i++) {{
                                if (inputs[i].id && inputs[i].id.includes('bill-ocr-result-html-edited')) {{
                                    hiddenInput = inputs[i];
                                    break;
                                }}
                            }}
                        }}
                        
                        // 方法3: 查找所有隐藏的输入框
                        if (!hiddenInput) {{
                            var allInputs = document.querySelectorAll('input[type="text"], textarea');
                            for (var i = 0; i < allInputs.length; i++) {{
                                var input = allInputs[i];
                                // 检查是否是隐藏的组件
                                if ((input.style.display === 'none' || input.hidden || input.offsetParent === null) &&
                                    input.id && input.id.includes('bill')) {{
                                    hiddenInput = input;
                                    break;
                                }}
                            }}
                        }}
                        
                        if (hiddenInput) {{
                            console.log('[DEBUG] 找到隐藏Textbox，ID:', hiddenInput.id);
                            hiddenInput.value = fullContent;
                            
                            // 触发所有可能的事件，确保Gradio捕获到变化
                            var events = ['input', 'change', 'blur', 'keyup', 'focus'];
                            events.forEach(function(eventType) {{
                                try {{
                                    var event = new Event(eventType, {{ bubbles: true, cancelable: true }});
                                    hiddenInput.dispatchEvent(event);
                                }} catch(err) {{
                                    console.error('触发事件失败:', eventType, err);
                                }}
                            }});
                            
                            // 也尝试直接设置属性
                            if (hiddenInput.setAttribute) {{
                                hiddenInput.setAttribute('value', fullContent);
                            }}
                            
                            console.log('[DEBUG] 导出前已强制更新，内容长度:', fullContent.length);
                            console.log('[DEBUG] Textbox当前值长度:', hiddenInput.value.length);
                        }} else {{
                            console.error('[DEBUG] 导出前未找到隐藏Textbox，尝试所有输入框...');
                            var allInputs = document.querySelectorAll('input, textarea');
                            console.log('[DEBUG] 找到', allInputs.length, '个输入框');
                            for (var i = 0; i < Math.min(allInputs.length, 10); i++) {{
                                console.log('  输入框', i, ':', allInputs[i].id, allInputs[i].name, allInputs[i].className);
                            }}
                        }}
                    }} else {{
                        console.error('[DEBUG] 未找到表格元素');
                    }}
                }}, true); // 使用捕获阶段，确保先执行
            }} else {{
                // 如果按钮还没加载，延迟重试
                setTimeout(setupExportButton, 500);
            }}
        }}
        
        // 延迟设置导出按钮监听器
        setTimeout(setupExportButton, 1000);
    }})();
    </script>
    """

CARD_FINAL_RESULT_TABLE_TEMPLATE = """
    <style>
    /* 可调整大小的表格容器 */
    .ocr-result-table-container {{
        position: relative;
        display: inline-block;
        min-width: 500px;
        min-height: 300px;
        max-width: 95vw;
        max-height: 90vh;
        width: 100%;
        height: 600px;
        resize: both;
        overflow: auto;  /* 允许滚动，确保表格不超出容器 */
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }}
    /* 调整大小手柄样式 */
    .ocr-result-table-container::-webkit-resizer {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 0 8px 0;
        width: 20px;
        height: 20px;
    }}
    /* 调整大小提示 */
    .ocr-result-table-container::before {{
        content: '↘ 拖拽调整大小';
        position: absolute;
        top: 5px;
        right: 5px;
        font-size: 11px;
        color: #667eea;
        background: rgba(255, 255, 255, 0.9);
        padding: 2px 6px;
        border-radius: 4px;
        pointer-events: none;
        opacity: 0.7;
        z-index: 5;
        transition: opacity 0.3s ease;
    }}
    .ocr-result-table-container:hover::before {{
        opacity: 1;
    }}
    /* 调整大小时的边框高亮 */
    .ocr-result-table-container:active {{
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}
    .ocr-result-table {{
        width: auto;  /* 表格宽度根据内容自适应 */
        min-width: 100%;  /* 最小宽度为容器宽度 */
        max-width: 100%;  /* 最大宽度不超过容器 */
        border-collapse: collapse;
        margin: 0;
        font-size: 14px;
        table-layout: auto;  /* 使用auto，让列宽根据内容自动调整 */
        box-shadow: none;
        border-radius: 8px;
        overflow: visible;  /* 允许内容溢出，不裁剪 */
        background-color: #ffffff;
    }}
    .ocr-result-table th,
    .ocr-result-table td {{
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        text-align: left;
        vertical-align: top;
        word-break: break-word;
        word-wrap: break-word;
        transition: all 0.2s ease;
        line-height: 1.6;
        height: auto !important;  /* 行高根据内容自动调整，覆盖HTML中的固定height */
        min-height: auto !important;
        overflow: visible;  /* 允许内容显示，不裁剪 */
        width: auto;  /* 列宽根据内容自动调整 */
        max-width: none;  /* 不限制最大宽度 */
    }}
    /* 字段名列：根据内容自适应宽度 */
    .ocr-result-table td:not([contenteditable="true"]) {{
        background-color: #f8f9fa;
        font-weight: 600;
        color: #374151;
        width: auto;  /* 宽度根据内容自适应 */
        min-width: 120px;  /* 最小宽度 */
        max-width: 300px;  /* 最大宽度限制，避免过宽 */
        white-space: nowrap;  /* 字段名不换行 */
        font-size: 14px;
        border-right: 2px solid #d1d5db;
        height: auto !important;
        overflow: visible;
    }}
    /* 值列：根据内容自适应宽度 */
    .ocr-result-table td[contenteditable="true"] {{
        background-color: #ffffff;
        cursor: text;
        min-height: 20px;
        height: auto !important;  /* 行高根据内容自动调整 */
        position: relative;
        width: auto;  /* 宽度根据内容自适应 */
        min-width: 200px;  /* 最小宽度 */
        max-width: none;  /* 不限制最大宽度，允许长文本 */
        overflow: visible;  /* 允许内容显示 */
        word-break: break-word;  /* 长文本自动换行 */
    }}
    /* 根据文本长度动态调整样式（保持列宽比例） */
    .ocr-result-table td[contenteditable="true"][data-length="short"] {{
        font-size: 15px;
        padding: 10px 14px;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="medium"] {{
        font-size: 14px;
        padding: 12px 16px;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="long"] {{
        font-size: 13px;
        padding: 14px 18px;
        line-height: 1.7;
        height: auto !important;
    }}
    .ocr-result-table td[contenteditable="true"][data-length="very-long"] {{
        font-size: 12px;
        padding: 16px 20px;
        line-height: 1.8;
        height: auto !important;
    }}
    .ocr-result-table th {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-color: #5568d3;
    }}
    .ocr-result-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .ocr-result-table tr:nth-child(odd) {{
        background-color: #ffffff;
    }}
    .ocr-result-table tr:hover {{
        background-color: #f0f4ff;
    }}
    .ocr-result-table tr:hover td:not([contenteditable="true"]) {{
        background-color: #e5e7eb;
    }}
    .ocr-result-table td[contenteditable="true"]:hover {{
        background-color: #f8f9ff;
        box-shadow: inset 0 0 0 1px #667eea;
    }}
    .ocr-result-table td[contenteditable="true"]:focus {{
        outline: none;
        background-color: #eef5ff;
        box-shadow: inset 0 0 0 2px #667eea, 0 0 0 3px rgba(102, 126, 234, 0.1);
        border-radius: 4px;
    }}
    .ocr-result-table td[contenteditable="true"]:empty:before {{
        content: "点击编辑...";
        color: #999;
        font-style: italic;
    }}
    .ocr-result-table td[contenteditable="true"]:empty:focus:before {{
        content: "";
    }}
    /* 优化长文本显示 */
    .ocr-result-table td[contenteditable="true"] {{
        overflow-wrap: break-word;
        hyphens: auto;
    }}
    /* 响应式设计 */
    @media (max-width: 768px) {{
        .ocr-result-table-container {{
            min-width: 300px;
            min-height: 200px;
        }}
        .ocr-result-table {{
            font-size: 12px;
            table-layout: fixed;
        }}
        .ocr-result-table th,
        .ocr-result-table td {{
            padding: 8px 12px;
        }}
        .ocr-result-table td:not([contenteditable="true"]) {{
            width: 30%;
            font-size: 12px;
        }}
        .ocr-result-table td[contenteditable="true"] {{
            width: 70%;
        }}
    }}
    </style>
    <script>
    (function() {{
        // 移除所有固定的height和width属性，让行高和列宽根据内容自动调整
        function removeFixedHeights() {{
            var table = document.querySelector('.ocr-result-table');
            if (table) {{
                // 移除table的width属性
                if (table.hasAttribute('width')) {{
                    table.removeAttribute('width');
                }}
                if (table.style.width) {{
                    table.style.width = '';
                }}
                
                // 移除tr的height和width属性
                var rows = table.querySelectorAll('tr');
                rows.forEach(function(row) {{
                    if (row.hasAttribute('height')) {{
                        row.removeAttribute('height');
                    }}
                    if (row.hasAttribute('width')) {{
                        row.removeAttribute('width');
                    }}
                }});
                
                // 移除td和th的height和width属性
                var cells = table.querySelectorAll('td, th');
                cells.forEach(function(cell) {{
                    if (cell.hasAttribute('height')) {{
                        cell.removeAttribute('height');
                    }}
                    if (cell.hasAttribute('width')) {{
                        cell.removeAttribute('width');
                    }}
                    // 移除内联样式中的height和width
                    if (cell.style.height) {{
                        cell.style.height = '';
                    }}
                    if (cell.style.width) {{
                        cell.style.width = '';
                    }}
                }});
                
                // 移除colgroup中的width属性
                var colgroups = table.querySelectorAll('colgroup');
                colgroups.forEach(function(colgroup) {{
                    var cols = colgroup.querySelectorAll('col');
                    cols.forEach(function(col) {{
                        if (col.hasAttribute('width')) {{
                            col.removeAttribute('width');
                        }}
                        if (col.style.width) {{
                            col.style.width = '';
                        }}
                    }});
                }});
            }}
        }}
        
        // 根据文本长度动态设置data-length属性
        function updateCellLength() {{
            var cells = document.querySelectorAll('.ocr-result-table td[contenteditable="true"]');
            cells.forEach(function(cell) {{
                var text = cell.textContent || cell.innerText || '';
                var length = text.length;
                cell.removeAttribute('data-length');
                if (length > 0) {{
                    if (length <= 20) {{
                        cell.setAttribute('data-length', 'short');
                    }} else if (length <= 50) {{
                        cell.setAttribute('data-length', 'medium');
                    }} else if (length <= 100) {{
                        cell.setAttribute('data-length', 'long');
                    }} else {{
                        cell.setAttribute('data-length', 'very-long');
                    }}
                }}
            }});
        }}
        
        // 页面加载后执行
        setTimeout(function() {{
            removeFixedHeights();
            updateCellLength();
        }}, 100);
        
        // 监听内容变化
        var observer = new MutationObserver(function(mutations) {{
            removeFixedHeights();
            updateCellLength();
        }});
        
        setTimeout(function() {{
            var table = document.querySelector('.ocr-result-table');
            if (table) {{
                observer.observe(table, {{
                    childList: true,
                    subtree: true,
                    characterData: true,
                    attributes: true,
                    attributeFilter: ['height', 'style']
                }});
            }}
        }}, 200);
    }})();
    </script>
    <div class="ocr-result-table-container">
        {html_content}
    </div>
    """