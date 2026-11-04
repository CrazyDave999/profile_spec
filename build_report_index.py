"""Keep all dated HTML reports reachable without moving historical evidence."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path

REPORTS = Path(__file__).resolve().parent/"reports"


class Title(HTMLParser):
    def __init__(self):
        super().__init__(); self.inside=False; self.value=""

    def handle_starttag(self, tag, attrs):
        if tag == 'title':self.inside=True

    def handle_endtag(self, tag):
        if tag == 'title':self.inside=False

    def handle_data(self, data):
        if self.inside:self.value+=data


def write_index():
    files = sorted(REPORTS.glob('20??-??-??_*.html'),key=lambda p:(p.name[:10],3 if p.name.endswith('_parallel_sd.html') else 2 if p.name.endswith('_jetspec_profile.html') else 1,p.name),reverse=True)
    files += [REPORTS/p for p in ['2026-11-04/parallel_sd_checkpoints/report.html','2026-11-04/sglang_sd_research/report.html','2026-11-03/report.html']]
    links=[]
    for p in files:
        if not p.exists():continue
        title=Title();title.feed(p.read_text());label=title.value or str(p.relative_to(REPORTS))
        links.append(f'<li><a href="{escape(str(p.relative_to(REPORTS)))}">{escape(label)}</a><br><small>{escape(str(p.relative_to(REPORTS)))}</small></li>')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parallel SD · 实验报告</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:50px auto;padding:24px;color:#203247}a{color:#14778d}li{margin:18px 0}small{color:#657384}</style><h1>Parallel SD · 实验报告</h1><p>新报告统一使用 reports/YYYY-MM-DD_topic.html，图表与小型证据放在同名目录。大 profiler 文件留在本地 profiles/；历史报告与原始快照保留原路径。</p><ul>'''+''.join(links)+'''</ul><p>克隆或下载仓库后用浏览器打开 HTML。GitHub 源码页面本身不会执行 HTML。</p></html>'''
    (REPORTS/'index.html').write_text(content)


if __name__=='__main__':write_index()
