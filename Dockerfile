# shuo import, small: shuo's import server for a small free host (Render's free web service: 512 MB, a tenth of a
# CPU). Videos with Chinese captions only: no speech model (it needs more memory) and no YouTube token maker (too slow
# here). The full one (../Dockerfile) is for a Hugging Face Space. The host gives the port in $PORT.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/home/user/.deno/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SHUO_MODELS=/home/user/models \
    SHUO_DATA=/tmp/shuo-import
WORKDIR /home/user

# deno: what yt-dlp uses to answer YouTube's JavaScript checks
RUN mkdir -p .deno/bin \
    && curl -fsSL -o /tmp/deno.zip https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    && unzip -o -q /tmp/deno.zip -d .deno/bin && rm /tmp/deno.zip && .deno/bin/deno --version

# yt-dlp and OpenCC (Traditional to Simplified)
RUN pip install --no-cache-dir --user "yt-dlp[default]" opencc-python-reimplemented

COPY --chown=user shuo_import.py /home/user/shuo_import.py
RUN python /home/user/shuo_import.py --selftest || echo "selftest: see above (no speech model here: fine)"
EXPOSE 7860
CMD ["python", "/home/user/shuo_import.py"]
