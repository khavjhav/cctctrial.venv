from reactpy import component, html


@component
def Sidebar():
    return html.div(
        {"class": "box", "style": "width: 250px;"},
        [
            html.img(
                {
                    "style": "width: 30%; display: block; margin-left: auto; margin-right: auto;",
                    "src": "/static/logo.png",
                }
            ),
            html.hr(),
            html.form(
                [
                    html.input(
                        {"class": "button is-light is-fullwidth", "value": "Dashboard"},
                    ),
                    html.br(),
                    html.input(
                        {
                            "class": "button is-light is-fullwidth",
                            "value": "Camera List",
                        },
                    ),
                ]
                # {
                #     "class": "button is-light is-fullwidth",
                #     "value": "Dashboard",
                #     "onclick": "window.location.href='/page1.html';",
                # }
            ),
        ],
    )
