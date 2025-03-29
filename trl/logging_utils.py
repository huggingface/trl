import html
from typing import Dict, List, Any
from pygments.formatters import HtmlFormatter

formatter = HtmlFormatter(nowrap=True, style="monokai")

def build_html_table(table: Dict[str, List[Any]]):
    # Modal and overlay styles + JavaScript for fullscreen functionality
    modal_styles = """
    <style>
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
        }
        .modal-content {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 95vw;
            max-width: 1200px;
            height: 95vh;
            z-index: 1001;
            display: flex;
            flex-direction: column;
        }
        .modal-inner {
            flex: 1;
            overflow: auto;
            padding: 1em;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            display: flex;
        }
        /* Override inline constraints when content is shown in modal */
        .modal-inner .content-wrapper {
            max-height: none !important;
            height: auto !important;
            overflow-y: visible !important;
        }
        .modal-inner .content-wrapper pre {
            height: auto !important;
        }
        .modal-inner > .content-wrapper {
            flex: 1;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .modal-inner pre {
            flex: 1;
            margin: 0;
            height: 100%;
        }
        .close-modal {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 24px;
            cursor: pointer;
            z-index: 1002;
        }
        .cell-content {
            cursor: pointer;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .cell-content:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        /* Content formatting */
        .content-wrapper {
            width: 100%;
            min-width: 0;
            word-wrap: break-word;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .content-wrapper p {
            margin: 0.5em 0;
            white-space: pre-wrap;
            width: 100%;
        }
        .content-wrapper pre {
            margin: 0;
            white-space: pre;
            overflow-x: auto;
            width: 100%;
            flex: 1;
            min-height: 0;
        }
        /* Ensure code blocks in table cells don't overflow */
        td .content-wrapper {
            overflow-x: auto;
            width: 100%;
            min-width: 0;
            height: 300px;
            display: flex;
            flex-direction: column;
        }
        td pre {
            margin: 0;
            width: 100%;
            flex: 1;
            min-height: 0;
        }
    </style>
    """

    modal_script = """
    <script>
        function showModal(content) {
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content highlight';
            
            const modalInner = document.createElement('div');
            modalInner.className = 'modal-inner';
            modalInner.innerHTML = content;
            
            const closeBtn = document.createElement('span');
            closeBtn.className = 'close-modal';
            closeBtn.innerHTML = '×';
            closeBtn.onclick = function() {
                document.body.removeChild(overlay);
            };
            
            modalContent.appendChild(modalInner);
            modalContent.appendChild(closeBtn);
            overlay.appendChild(modalContent);
            
            overlay.onclick = function(e) {
                if (e.target === overlay) {
                    document.body.removeChild(overlay);
                }
            };
            
            document.body.appendChild(overlay);
            overlay.style.display = 'block';
        }
    </script>
    """

    # Build HTML table with fixed layout and pygments styling
    html_str = modal_styles + modal_script + """
    <table style="width:100%; border-collapse: collapse; table-layout: fixed;" class="highlight">
      <thead>
        <tr>
          <th style="border: 1px solid black; padding: 6px; width:33.33%;">Prompt</th>
          <th style="border: 1px solid black; padding: 6px; width:33.33%;">Response</th>
          <th style="border: 1px solid black; padding: 6px; width:33.33%;">Ground Truth</th>
        </tr>
      </thead>
      <tbody>
    """

    for prompt, response, correct in zip(table["prompt"], table["completion"], table["answer"]):
        # --- Prompt with syntax highlighting ---
        prompt_html = (
            "<div class='cell-content' onclick='showModal(this.innerHTML)'>"
            "<div class='content-wrapper' style='height:100%; overflow-y:auto;'>"
            f"<pre style='margin:0; height:100%;'>{prompt}</pre>"
            "</div>"
            "</div>"
        )

        # --- Response (simple HTML formatting) ---
        response_html = (
            "<div class='cell-content' onclick='showModal(this.innerHTML)'>"
            "<div class='content-wrapper' style='height:100%; overflow-y:auto;'>"
            f"{response}"
            "</div>"
            "</div>"
        )

        # --- Correct Answer ---
        correct_escaped = html.escape(str(correct))
        correct_html = (
            "<div class='cell-content' onclick='showModal(this.innerHTML)'>"
            "<div class='content-wrapper' style='height:100%; overflow-y:auto;'>"
            f"<pre style='margin:0; height:100%;'>{correct_escaped}</pre>"
            "</div>"
            "</div>"
        )

        html_str += f"""
        <tr>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{prompt_html}</td>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{response_html}</td>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{correct_html}</td>
        </tr>
        """

    html_str += """
      </tbody>
    </table>
    """

    # Embed the Pygments CSS for syntax highlighting
    css = formatter.get_style_defs('.highlight')
    style_block = f"<style>{css}</style>"

    return style_block + html_str