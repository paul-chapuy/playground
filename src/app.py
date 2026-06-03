from __future__ import annotations

import panel as pn

from access import PAGES, ROLE_LABELS, can_access, get_user, user_options

pn.extension(sizing_mode="stretch_width")

PRIMARY = "#175c62"


def hello_page(title: str) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(f"## {title}"),
        pn.pane.Markdown(f"welcome to the {title} page"),
    )


def build_app() -> pn.template.FastListTemplate:
    selected_user = pn.widgets.Select(
        name="user",
        options=user_options(),
        value="111",
    )
    selected_page = pn.widgets.RadioButtonGroup(
        name="Page",
        options={title: page for page, title in PAGES.items()},
        value="analytics",
        button_type="primary",
    )
    content = pn.Column(sizing_mode="stretch_both")
    access_note = pn.pane.Alert(alert_type="light", margin=(12, 0, 0, 0))

    def render_page(*_: object) -> None:
        user = get_user(selected_user.value)
        page = selected_page.value
        title = PAGES[page]

        access_note.object = (
            f"Signed in as **{user.name}** with role **{ROLE_LABELS[user.role]}**."
        )

        if can_access(user, page):
            content[:] = [hello_page(title)]
            return

        content[:] = [
            pn.Column(
                pn.pane.Markdown("## Access Denied"),
                pn.pane.Markdown(f"{ROLE_LABELS[user.role]} cannot open {title}."),
            )
        ]

    selected_user.param.watch(render_page, "value")
    selected_page.param.watch(render_page, "value")

    template = pn.template.FastListTemplate(
        title="Panel Access Playground",
        accent_base_color=PRIMARY,
        header_background=PRIMARY,
        sidebar=[
            selected_user,
            access_note,
            pn.pane.Markdown("### Pages"),
            selected_page,
        ],
        main=[content],
        theme_toggle=False,
    )

    render_page()
    return template


build_app().servable()
