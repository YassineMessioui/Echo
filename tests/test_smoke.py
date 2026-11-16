import echo


def test_main_runs(capsys):
    echo.main()
    assert "echo" in capsys.readouterr().out
