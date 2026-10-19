from datetime import datetime
from html import escape
from pathlib import Path


class BaseTemplate:
    def __init__(
        self,
        dataset_name: str,
        title: str,
        full_width: bool = False,
    ):
        self.dataset_name = dataset_name
        self.title = title
        self.full_width = True

    def content(self) -> str:
        raise NotImplementedError("Subclasses must implement content().")

    def render(self) -> str:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        container_max_width = "none" if self.full_width else "1200px"

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{escape(self.dataset_name)} - {escape(self.title)}</title>

            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    margin: 0;
                    background: #0f172a;
                    color: #e2e8f0;
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                }}

                .container {{
                    max-width: {container_max_width};
                    width: 100%;
                    margin: 0 auto;
                    padding: 32px 24px;
                }}

                .header {{
                    border-bottom: 1px solid #334155;
                    padding-bottom: 28px;
                    margin-bottom: 32px;
                    text-align: center;
                }}

                .brand {{
                    color: #38bdf8;
                    font-size: 14px;
                    font-weight: bold;
                    letter-spacing: 1px;
                    margin-bottom: 20px;
                }}

                .dataset {{
                    font-size: 34px;
                    font-weight: bold;
                    color: #f8fafc;
                    margin-bottom: 8px;
                    overflow-wrap: anywhere;
                }}

                h1 {{
                    font-size: 21px;
                    font-weight: 500;
                    color: #94a3b8;
                    margin: 0 0 12px;
                }}

                .generated-at {{
                    color: #64748b;
                    font-size: 12px;
                }}

                .grid {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 12px;
                }}

                .card {{
                    min-width: 0;
                    background: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 10px;
                    padding: 16px;
                }}

                .card-title {{
                    font-size: 13px;
                    font-weight: bold;
                    color: #38bdf8;
                    margin-bottom: 8px;
                }}

                pre {{
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    margin: 0;
                    font-size: 12px;
                    font-family: Consolas, monospace;
                }}

                .table-section {{
                    margin-top: 32px;
                }}

                .table-section h2 {{
                    font-size: 18px;
                    margin: 0 0 12px;
                    color: #f1f5f9;
                }}

                .table-wrapper {{
                    width: 100%;
                    overflow-x: auto;
                    border: 1px solid #334155;
                    border-radius: 10px;
                }}

                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                    text-align: center;
                }}

                .data-table th,
                .data-table td {{
                    padding: 10px 14px;
                    border-bottom: 1px solid #334155;
                    white-space: nowrap;
                    text-align: center;
                    vertical-align: middle;
                }}

                .data-table thead {{
                    background: #1e293b;
                    color: #38bdf8;
                }}

                .data-table tbody tr:nth-child(even) {{
                    background: #162235;
                }}

                .data-table tbody tr:hover {{
                    background: #243449;
                }}

                .data-table tbody tr:last-child td,
                .data-table tbody tr:last-child th {{
                    border-bottom: none;
                }}

                .data-table tbody th {{
                    color: #a78bfa;
                    font-weight: 600;
                }}

                .footer {{
                    border-top: 1px solid #334155;
                    margin-top: 48px;
                    padding-top: 16px;
                    color: #64748b;
                    font-size: 12px;
                    text-align: center;
                }}

                @media (max-width: 650px) {{
                    .container {{
                        padding: 20px 12px;
                    }}

                    .dataset {{
                        font-size: 26px;
                    }}

                    .grid {{
                        grid-template-columns: 1fr;
                    }}

                    .card-full {{
                        grid-column: auto !important;
                    }}
                }}
            </style>
        </head>

        <body>
            <main class="container">
                <header class="header">
                    <div class="dataset">
                        {escape(self.dataset_name)}
                    </div>

                    <h1>{escape(self.title)}</h1>

                    <div class="generated-at">
                        Generated on {generated_at}
                    </div>
                </header>

                {self.content()}

                <footer class="footer">
                    OpenLearnLab · Learn. Experiment. Understand. Build.
                </footer>
            </main>
        </body>
        </html>
        """

    def save(self, output: Path, suffix_file_name="") -> Path:
        output = Path(output)

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if suffix_file_name:
            filename += f"-{suffix_file_name}"
        filename += ".html"

        file_path = output / filename

        file_path.write_text(
            self.render(),
            encoding="utf-8",
        )

        return file_path
