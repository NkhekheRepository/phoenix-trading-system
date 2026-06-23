FROM freqtradeorg/freqtrade:stable

RUN pip install --no-cache-dir \
    joblib==1.5.3

COPY strategies/ /freqtrade/strategies/
COPY ml/ /freqtrade/ml/
COPY scripts/ /freqtrade/scripts/

WORKDIR /freqtrade
