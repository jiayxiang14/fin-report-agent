"""所有带`{ticker}`路径/查询参数的路由共用同一个`TickerPath`/`TickerQuery`
格式校验（见`app/api/ticker_path.py`）：非法输入应该在路由层就被FastAPI拦成
422，根本不会走到service层去拼URL/缓存文件名。这里只挑几条有代表性的路由
覆盖，不需要每个路由都测一遍（校验逻辑是共享的，不是各自实现的）。

带斜杠的路径穿越（如`../../etc/passwd`）作为`{ticker}`路径参数传入时，会先
被Starlette的路由匹配挡成404——请求根本进不了我们的handler，这本身就是安全
的，只是不会经过我们的422校验层。真正会经过`TickerQuery`校验层的路径穿越
场景是查询参数（`thematic-flow`的`ticker`是可选查询参数，不受路径路由匹配
规则约束），下面单独测。
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_path_traversal_as_path_param_never_reaches_handler():
    response = client.get("/api/financials/..%2F..%2Fetc%2Fpasswd")

    assert response.status_code == 404


def test_path_traversal_as_query_param_is_rejected_with_422():
    response = client.get("/api/thematic-flow?ticker=..%2F..%2Fetc%2Fpasswd")

    assert response.status_code == 422


def test_ticker_with_digits_is_rejected_with_422():
    response = client.get("/api/company-profile/AAPL1")

    assert response.status_code == 422


def test_ticker_too_long_is_rejected_with_422():
    response = client.get("/api/sector-position/TOOLONGTICKER")

    assert response.status_code == 422


def test_class_share_ticker_with_dot_is_accepted():
    response = client.get("/api/financials/BRK.B")

    assert response.status_code != 422
