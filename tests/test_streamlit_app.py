from streamlit.testing.v1 import AppTest


def _run_app() -> AppTest:
    app = AppTest.from_file("app/streamlit/app.py", default_timeout=90)
    app.run()
    assert not app.exception
    return app


def test_streamlit_chat_coach_and_crisis_flows():
    app = _run_app()
    app.text_area[0].set_value(
        "The meeting made me angry because I felt talked over. "
        "I do not want to explode, but I also do not want to pretend it was fine."
    )
    app.text_input[0].set_value("Office")
    app.text_input[1].set_value("meeting")
    labels = [button.label for button in app.button]
    app = app.button[labels.index("Send to JournalPulse")].click().run()
    assert not app.exception

    pending = app.session_state["pending_entry"]
    assert pending["prediction"].emotion == "anger"
    assert pending["coach_transcript"][0]["role"] == "user"
    assert all(resource["resource_type"] != "support" for resource in pending["resources"])

    app = _run_app()
    app.text_area[0].set_value("I do not feel safe tonight and I need help.")
    labels = [button.label for button in app.button]
    app = app.button[labels.index("Send to JournalPulse")].click().run()
    assert not app.exception

    pending = app.session_state["pending_entry"]
    assert pending["prediction"].is_crisis is True
    assert [resource["id"] for resource in pending["resources"]] == [
        "support_988",
        "support_befrienders",
        "site_nimh_need_help",
    ]
