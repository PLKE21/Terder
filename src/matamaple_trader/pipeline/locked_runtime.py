from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from matamaple_trader.data.development_builder import build_development_dataset
from matamaple_trader.domain import Bar, MarketSnapshot, PipelineContext, Signal, SignalResult
from matamaple_trader.features import FeatureEngine
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.models import LightGBMModel, LogisticBaseline, TrainingGuard, XGBoostModel
from matamaple_trader.models.calibration import BinaryProbabilityCalibrator, CalibrationMethod
from matamaple_trader.regime import RegimeEngine
from matamaple_trader.validation.candidate_lock import CandidateLockRegistry
from matamaple_trader.validation.purged import PurgedKFold, PurgedSplitConfig

from .core import QuantSignalPipeline


class LockedQuantPipeline:
    """Shared research/shadow quantitative path for one immutable candidate.

    The adapter supplies only raw bars/tick data. Feature generation, locked-model
    inference, development-only calibration, regime classification, cost inputs,
    candidate direction and final signal/levels/grade all live in this object.
    No execution API exists here.
    """

    def __init__(self, *, lock: dict, feature_engine: FeatureEngine, model, calibrator: BinaryProbabilityCalibrator,
                 regime_engine: RegimeEngine, feature_columns: tuple[str, ...], normal_spread_points: float | None) -> None:
        self.lock = lock
        self.feature_engine = feature_engine
        self.model = model
        self.calibrator = calibrator
        self.regime_engine = regime_engine
        self.feature_columns = feature_columns
        self.normal_spread_points = normal_spread_points
        self.signal_pipeline = QuantSignalPipeline()

    @staticmethod
    def _factory(lock: dict, contract):
        name = lock['model_name']
        if name == 'logistic_regression':
            return LogisticBaseline(contract)
        if name == 'xgboost':
            return XGBoostModel(contract, n_estimators=int(lock['booster_n_estimators']), max_depth=4, learning_rate=0.05)
        if name == 'lightgbm':
            return LightGBMModel(contract, n_estimators=int(lock['booster_n_estimators']), max_depth=-1, learning_rate=0.05)
        raise RuntimeError('unsupported locked model')

    @staticmethod
    def _positive_probability(output) -> np.ndarray:
        if output.probability is None or output.classes is None:
            raise RuntimeError('locked binary candidate must expose probability/classes')
        classes = np.asarray(output.classes)
        if set(classes.tolist()) != {0, 1}:
            raise RuntimeError(f'locked binary candidate classes must be {{0,1}}, got {classes.tolist()}')
        return np.asarray(output.probability)[:, int(np.where(classes == 1)[0][0])]

    @classmethod
    def from_artifacts(cls, *, parquet_path: str | Path, holdout_registry_path: str | Path,
                       candidate_lock_path: str | Path) -> 'LockedQuantPipeline':
        lock = CandidateLockRegistry(candidate_lock_path).load()
        if lock['label_name'] != 'n_bar_direction' or float(lock['neutral_threshold']) != 0.0:
            raise RuntimeError('runtime v1 supports locked binary n_bar_direction only')
        contract = n_bar_direction_contract(int(lock['horizon_bars']), 0.0)
        if contract.label_version != lock['label_version']:
            raise RuntimeError('candidate lock label version mismatch')

        feature_engine = FeatureEngine()
        if feature_engine.config.feature_version != lock['feature_version']:
            raise RuntimeError('candidate lock feature version mismatch')
        development = build_development_dataset(
            parquet_path, holdout_registry_path=holdout_registry_path,
            label_contract=contract, feature_engine=feature_engine,
        )
        X = development.frame[list(development.feature_columns)].replace([np.inf, -np.inf], np.nan)
        y = development.frame[development.label_column]
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].astype(int).reset_index(drop=True)
        if set(y.unique().tolist()) != {0, 1}:
            raise RuntimeError('development partition must contain both classes')

        guard = TrainingGuard(holdout_registry_path)
        guard.assert_allowed()
        n = len(X)
        event_end = np.minimum(np.arange(n) + contract.horizon_bars, n - 1)
        oof = np.full(n, np.nan, dtype=float)
        splitter = PurgedKFold(PurgedSplitConfig(
            n_splits=int(lock['purged_splits']), embargo_bars=int(lock['embargo_bars'])
        ))
        for train_idx, test_idx in splitter.split(X, event_end_index=event_end):
            if set(y.iloc[train_idx].unique().tolist()) != {0, 1}:
                raise RuntimeError('single-class development fold blocks locked calibration')
            fold_model = cls._factory(lock, contract).fit(X.iloc[train_idx], y.iloc[train_idx], guard=guard)
            oof[test_idx] = cls._positive_probability(fold_model.predict_output(X.iloc[test_idx]))
        if not np.isfinite(oof).all():
            raise RuntimeError('development OOF calibration coverage incomplete')

        calibrator = BinaryProbabilityCalibrator(CalibrationMethod(str(lock['calibration_method'])))
        calibrator.fit(oof, y.to_numpy(), source='oof')
        model = cls._factory(lock, contract).fit(X, y, guard=guard)

        dev_features = development.frame.loc[mask].reset_index(drop=True)
        regime = RegimeEngine().fit(
            dev_features['ema_spread_pct'].to_numpy(), dev_features['volatility'].to_numpy(), source='development'
        )
        spread = None
        if 'spread' in development.frame.columns:
            vals = pd.to_numeric(development.frame['spread'], errors='coerce').dropna()
            if len(vals):
                spread = float(vals.median())
        return cls(lock=lock, feature_engine=feature_engine, model=model, calibrator=calibrator,
                   regime_engine=regime, feature_columns=tuple(development.feature_columns), normal_spread_points=spread)

    @staticmethod
    def _bars_frame(bars: tuple[Bar, ...]) -> pd.DataFrame:
        return pd.DataFrame([{
            'timestamp': b.timestamp, 'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
            'tick_volume': b.tick_volume, 'spread': b.spread_points,
        } for b in bars])

    def evaluate(self, market: MarketSnapshot, context: PipelineContext) -> SignalResult:
        if context.pipeline_version != self.lock['pipeline_version']:
            raise RuntimeError('pipeline version does not match locked candidate')
        if not market.bars:
            return self.signal_pipeline.evaluate(market, context)
        frame = self.feature_engine.transform(self._bars_frame(market.bars))
        row = frame.iloc[[-1]]
        X = row[list(self.feature_columns)].replace([np.inf, -np.inf], np.nan)
        if not X.notna().all(axis=None):
            return self.signal_pipeline.evaluate(market, context)

        output = self.model.predict_output(X)
        raw_p = float(self._positive_probability(output)[0])
        calibrated_buy = float(self.calibrator.transform([raw_p])[0])
        buy = calibrated_buy >= 0.5
        candidate = Signal.BUY if buy else Signal.SELL
        confidence = calibrated_buy if buy else 1.0 - calibrated_buy
        raw_score_arr = np.asarray(output.raw_score).reshape(-1)
        raw_score = float(raw_score_arr[0]) if len(raw_score_arr) else None

        r = row.iloc[0]
        regime = self.regime_engine.classify(
            trend_strength=float(r['ema_spread_pct']), volatility=float(r['volatility'])
        ).value
        current_spread = float(market.bars[-1].spread_points)
        normal_spread = self.normal_spread_points
        if (normal_spread is None or normal_spread <= 0) and current_spread > 0:
            normal_spread = current_spread
        point = market.metadata.get('point')
        point_value = float(point) if isinstance(point, (int, float)) and float(point) > 0 else None
        estimated_cost = current_spread * point_value if point_value is not None else None
        expected_move = float(r['atr']) if np.isfinite(r['atr']) else None

        # MTF alignment is deliberately fail-safe until a closed-HTF adapter is configured.
        owned_metadata = {
            'candidate_signal': candidate,
            'raw_score': raw_score,
            'calibrated_probability': confidence,
            'regime': regime,
            'atr': float(r['atr']),
            'spread_points': current_spread,
            'normal_spread_points': normal_spread,
            'expected_move': expected_move,
            'estimated_cost': estimated_cost,
            'mtf_aligned': False,
            'trend_alignment': False,
            'volatility_ok': regime != 'HIGH_VOLATILITY',
            'structure_score': 0.0,
            'model_agreement': 1.0,
        }
        internal_market = replace(market, metadata=owned_metadata)
        return self.signal_pipeline.evaluate(internal_market, context)
