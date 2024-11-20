FROM python:3.12-slim

ENV OPENAI_API_KEY=""
ENV OPENAI_API_BASE="https://api.openai.com/v1"

WORKDIR /proof_of_value
COPY . /proof_of_valu

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /proof_of_value
USER appuser

ENTRYPOINT ["streamlit", "run", "vd_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--config", "value_dashboard/config/config_demo.toml"]
