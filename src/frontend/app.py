from reactpy import component, html, run
from components.sidebar import Sidebar

# from components.printbutton import PrintButton

css_url = "https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"
bulma_css = html.link({"rel": "stylesheet", "href": css_url})


@component
def Layout():
    return html.div(
        {"class": "section"},
        [
            html.div(
                {"class": "container"},
                [html.div({"class": "columns"}, [Sidebar()])],
            )
        ],
    )


@component
def App():
    return html.div(bulma_css, Layout())


run(App, host="0.0.0.0", port=8000, debug=True)
