FROM freqtradeorg/freqtrade:stable

RUN pip install --no-cache-dir \
    joblib==1.5.3 \
    lightgbm==4.6.0

COPY strategies/ /freqtrade/strategies/
COPY ml/ /freqtrade/ml/
COPY scripts/ /freqtrade/scripts/
COPY core/ /freqtrade/core/
COPY research/ /freqtrade/research/
COPY tests/ /freqtrade/tests/

WORKDIR /freqtrade
