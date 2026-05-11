'use strict';

/**
 * FuelNet Pro — Evaluation Script
 * Loads the saved model and prints full test-set metrics.
 *
 * Usage: node src/model/evaluate.js
 * Or:    npm run evaluate
 */

require('dotenv').config();

const tf = require('@tensorflow/tfjs');
const { loadDataset } = require('../data/loader');
const { loadModel }   = require('./model');
const { denormalizeMpg } = require('../data/preprocessor');
const logger = require('../utils/logger');

async function evaluate() {
  logger.info('═══════════════════════════════════════');
  logger.info('  FuelNet Pro — Evaluation             ');
  logger.info('═══════════════════════════════════════');

  const model = await loadModel();
  if (!model) { logger.error('No model found. Run: npm run train'); process.exit(1); }

  const { test } = await loadDataset(0.2);

  const predsTensor  = model.predict(test.xs);
  const predsNorm    = predsTensor.dataSync();
  const actualsNorm  = test.ys.dataSync();

  // Denormalize to MPG
  const preds   = Array.from(predsNorm).map(denormalizeMpg);
  const actuals = Array.from(actualsNorm).map(denormalizeMpg);

  const n = actuals.length;
  let mse = 0, mae = 0, ssTot = 0, ssRes = 0;
  const meanActual = actuals.reduce((a, b) => a + b, 0) / n;

  for (let i = 0; i < n; i++) {
    const diff = preds[i] - actuals[i];
    mse   += diff * diff;
    mae   += Math.abs(diff);
    ssTot += (actuals[i] - meanActual) ** 2;
    ssRes += diff * diff;
  }

  mse /= n;
  mae /= n;
  const rmse = Math.sqrt(mse);
  const r2   = 1 - ssRes / ssTot;
  const mape = actuals.reduce((s, a, i) => s + Math.abs((a - preds[i]) / a), 0) / n * 100;

  logger.info('Evaluation results on held-out test set:');
  logger.info(`  Samples : ${n}`);
  logger.info(`  MSE     : ${mse.toFixed(4)}`);
  logger.info(`  RMSE    : ${rmse.toFixed(4)} MPG`);
  logger.info(`  MAE     : ${mae.toFixed(4)} MPG`);
  logger.info(`  MAPE    : ${mape.toFixed(2)} %`);
  logger.info(`  R²      : ${r2.toFixed(4)}`);

  // Percentile error distribution
  const errors = actuals.map((a, i) => Math.abs(a - preds[i])).sort((a, b) => a - b);
  const p50 = errors[Math.floor(n * 0.50)];
  const p90 = errors[Math.floor(n * 0.90)];
  const p95 = errors[Math.floor(n * 0.95)];
  logger.info(`  P50 abs error: ${p50.toFixed(2)} MPG`);
  logger.info(`  P90 abs error: ${p90.toFixed(2)} MPG`);
  logger.info(`  P95 abs error: ${p95.toFixed(2)} MPG`);

  test.xs.dispose(); test.ys.dispose(); predsTensor.dispose();
  process.exit(0);
}

evaluate().catch(err => { logger.error(err); process.exit(1); });
