FROM python:3.11-slim
LABEL org.opencompass.code-eval.protocol="1"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --chmod=0444 runner.py /opt/runner.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "/opt/runner.py"]
