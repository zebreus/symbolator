## Symbolator
FROM ubuntu:latest AS build-symbolator

# Build from forked source because upstream is broken for the latest python3 versions
RUN apt-get update && apt-get install --no-install-recommends --yes \
    git \
    pip \
    python3-dev \
    patchelf \
    python3-gi-cairo \
    python3-gi \
    build-essential \
    libpango1.0-dev && \
    apt-get clean && apt-get autoremove

# Install latest pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install --upgrade nuitka

WORKDIR /build

ADD . /src

# Install symbolator
RUN python3 -m pip install --upgrade /src

# Use nuitka to compile a static binary so we dont need python in the final image
RUN python3 -m nuitka --onefile /usr/local/bin/symbolator --include-module=gi.overrides.Pango --include-module=gi._gi_cairo

FROM ubuntu:latest AS main

RUN apt-get update && apt-get install --no-install-recommends --yes \
    libpango1.0 && \
    apt-get clean && apt-get autoremove

COPY --from=build-symbolator /build/symbolator.bin /usr/bin/symbolator

ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--", "/usr/bin/symbolator"]

WORKDIR /src

