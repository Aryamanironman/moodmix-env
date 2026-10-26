"""Terminal rendering for the MoodMix control surface."""

from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

MOOD_STYLE = {
    "energetic": "bold bright_red",
    "happy": "bold yellow",
    "relaxed": "bold cyan",
    "emotional": "bold magenta",
    "focused": "bold blue",
    "party": "bold bright_magenta",
}

EQ_GLYPH = {0: ("−", "red"), 1: ("·", "dim"), 2: ("+", "cyan")}
DRUM_LABEL = {1: "K", 2: "S", 3: "H", 4: "C"}
BLOCKS = "▁▂▃▄▅▆▇█"


def _eq_block(states: list[int]) -> Table:
    table = Table.grid(padding=(0, 1))
    for _ in ("LOW", "MID", "HIGH"):
        table.add_column(justify="center")
    table.add_row(*[Text(label, style="dim") for label in ("LOW", "MID", "HIGH")])
    cells = []
    for state in states:
        glyph, style = EQ_GLYPH[state]
        cells.append(Text(glyph, style=style, justify="center"))
    table.add_row(*cells)
    return table


def _fader_text(value: float, width: int = 19) -> Text:
    pos = int(round(value * (width - 1)))
    text = Text()
    text.append("A ", style="bold green")
    for i in range(width):
        if i == pos:
            text.append("●", style="bold white")
        elif i < pos:
            text.append("━", style="green")
        else:
            text.append("─", style="dim")
    text.append(" B", style="bold cyan")
    return text


def _sparkline(values: list[float], width: int = 32) -> Text:
    if not values:
        return Text("")
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = max(hi - lo, 1e-6)
    text = Text()
    for value in recent:
        idx = int((value - lo) / span * (len(BLOCKS) - 1))
        color = "green" if value >= 0 else "red"
        text.append(BLOCKS[max(0, idx)], style=color)
    return text


def _deck_panel(info: dict, key: str, title: str, style: str) -> Panel:
    deck = info[key]
    eq = info["eq_a"] if key == "deck_a" else info["eq_b"]
    gain = info["gain_a"] if key == "deck_a" else info["gain_b"]
    body = Table.grid(padding=(0, 1))
    body.add_row(Text(deck["id"], style="bold"), Text(f"{gain:.2f}", style="dim"))
    body.add_row(
        Text(f"{deck['bpm']} BPM", style="bold"),
        Text(f"{deck['key']}  {deck['mood']}", style=style),
    )
    body.add_row(Text(f"[{deck['section']}]  beat {deck['position']:.0f}",
                      style="dim"), Text(""))
    body.add_row(_eq_block(eq))
    return Panel(body, title=title, border_style=style, padding=(0, 1))


def _mixer_panel(info: dict) -> Panel:
    body = Table.grid(padding=(0, 1))
    body.add_row(_fader_text(info["crossfader"]))
    sync = Text("SYNC", style="bold green" if info["sync"] else "dim")
    filt = ("OFF", "LPF", "HPF")[info["filter"]]
    body.add_row(sync, Text(f"FILTER {filt}", style="cyan" if info["filter"] else "dim"))
    pads = Text()
    drum_name = info["drum"]
    for idx, name in ((1, "kick"), (2, "snare"), (3, "hat"), (4, "clap")):
        active = drum_name == name
        pads.append(f" {DRUM_LABEL[idx]} ",
                    style="black on bright_yellow" if active else "dim")
        pads.append(" ")
    body.add_row(Text("PADS", style="dim"), pads)
    body.add_row(Text(f"step {info['step']:02d}/{info['max_steps']}", style="dim"),
                 Text(f"bar {info['bar']:02d}", style="dim"))
    return Panel(body, title="MIXER", border_style="white", padding=(0, 1))


def dashboard(info: dict, history: list[float], policy: str) -> Group:
    style = MOOD_STYLE.get(info["mood"], "white")
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text.assemble(("MOODMIX ", "bold white"), ("DJ SET", "bold bright_magenta"),
                      ("   mood ", "dim"), (info["mood"].upper(), style)),
        Text(f"{policy} policy   step {info['step']:02d}/{info['max_steps']}",
             style="dim"),
    )

    decks = Table.grid(expand=True)
    decks.add_column(ratio=3)
    decks.add_column(ratio=4)
    decks.add_column(ratio=3)
    decks.add_row(
        _deck_panel(info, "deck_a", "DECK A", "green"),
        _mixer_panel(info),
        _deck_panel(info, "deck_b", "DECK B", "cyan"),
    )

    components = info.get("reward_components", {})
    order = ("beat_alignment", "tempo", "eq_craft", "bass_swap",
             "filter_craft", "pad_craft", "phrase", "busy_hands", "jitter",
             "harmonic", "spectral", "transition_shape", "mood",
             "drum", "smoothness")
    parts = Text()
    for name in order:
        if name in components:
            value = components[name]
            colour = "green" if value >= 0 else "red"
            parts.append(f"{name.replace('_', ' ')} ", style="dim")
            parts.append(f"{value:+.2f}  ", style=colour)
    total = sum(info.get("episode_components", {}).values())
    top = Table.grid(expand=True)
    top.add_column(ratio=3)
    top.add_column(ratio=2)
    top.add_row(
        Text.assemble(("REWARD ", "dim"), (f"{info.get('reward', 0.0):+.3f}", "bold"),
                      ("   episode ", "dim"), (f"{total:+.2f}", "bold")),
        _sparkline(history),
    )
    bottom = Table.grid(expand=True)
    bottom.add_column()
    bottom.add_row(parts)
    bottom.add_row(Text(info.get("last_event", ""), style="italic dim"))

    return Group(
        Panel(header, border_style=style, padding=(0, 1)),
        decks,
        Panel(Group(top, bottom), border_style="dim", padding=(0, 1)),
    )
