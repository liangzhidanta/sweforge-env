"""Mac-side authenticated §8 client: vendored RemoteEnvironmentBackend + Bearer header.

AutoDL 的 RemoteEnvironmentBackend（vendored, 不可编辑）不发送认证头。Mac 侧
server 可选 `--token` 加 Bearer 认证后, AutoDL 客户端需要同样加一行请求头;
本模块给出 Mac 侧参考实现（子类复写 client, 重试/幂等逻辑不变）, 本地联调
与测试直接用它。

AutoDL 侧要做的同样改动（remote.py::_request 的 headers 加一行）:
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
"""

from __future__ import annotations

import httpx

from sweforge.environment.remote import DEFAULT_BASE_URL, RemoteEnvironmentBackend

__all__ = ["AuthenticatedRemoteEnvironmentBackend", "DEFAULT_BASE_URL"]


def _bearer(token: str):
    def _add_bearer(request: httpx.Request) -> httpx.Request:
        request.headers["Authorization"] = f"Bearer {token}"
        return request

    return _add_bearer


class AuthenticatedRemoteEnvironmentBackend(RemoteEnvironmentBackend):
    """§8 客户端, 每个请求带 `Authorization: Bearer <token>`。"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        super().__init__(base_url=base_url, timeout=timeout, max_retries=max_retries)
        if token:
            self._client = httpx.Client(timeout=timeout, auth=_bearer(token))
