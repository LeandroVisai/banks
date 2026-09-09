from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def render_html(
    nav_groups: list,
    figures_serialized: dict,
    market_table: list[dict],
    clp_stat: dict,
    *,
    template_dir: Path,
    template_name: str,
    title: str,
    title_short: str,
    subtitle: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template(template_name)
    return tpl.render(
        title=title,
        title_short=title_short,
        subtitle=subtitle,
        nav_groups=nav_groups,
        figures_json=json.dumps(figures_serialized, ensure_ascii=False),
        market_table=market_table,
        clp_stat=clp_stat,
    )
