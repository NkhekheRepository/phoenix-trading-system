FROM freqtradeorg/freqtrade:stable

RUN pip install --no-cache-dir \
    lightgbm==4.6.0 \
    scikit-learn==1.9.0 \
    joblib==1.5.3

COPY strategies/ /freqtrade/strategies/
COPY ml/ /freqtrade/ml/
COPY scripts/ /freqtrade/scripts/

WORKDIR /freqtrade
