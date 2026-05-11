'use strict';

/**
 * FuelNet Pro — Training Pipeline
 *
 * Usage: node src/model/train.js
 * Or:    npm run train
 */

require('dotenv').config();

const tf      =  require('@tensorflow/tfjs');
const path    = require('path');
const fs      = require('fs');
const { loadDataset }   = require('../data/loader');
const { buildModel, saveModel } = require('./model');
const logger  = require('../utils/logger');

const EPOCHS       = parseInt(process.env.TRAINING_EPOCHS  || '100');
const BATCH_SIZE   = parseInt(process.env.BATCH_SIZE       || '32');
const LEARNING_RATE = parseFloat(process.env.LEARNING_RATE || '0.001');
const HISTORY_PATH = path.resolve('./models/training_history.json');

async function train() {
  logger.info('═══════════════════════════════════════');
  logger.info('  FuelNet Pro — Training Pipeline      ');
  logger.info('═══════════════════════════════════════');
  logger.info(`Config: epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}`);

  // ── Load data ───────────────────────────────────────────────────────────
  const { train: trainData, test: testData } = await loadDataset(0.2);

  // ── Build model ─────────────────────────────────────────────────────────
  const model = buildModel({
    units:        [64, 128, 64],
    learningRate: LEARNING_RATE,
    dropoutRate:  0.2,
    inputSize:    7
  });

  model.summary();

  // ── Training callbacks ──────────────────────────────────────────────────
  const history = { loss: [], val_loss: [], mae: [], val_mae: [] };

  let bestValLoss = Infinity;
  let bestEpoch   = 0;

  const callbacks = {
    onEpochEnd: async (epoch, logs) => {
      const e = epoch + 1;
      history.loss.push(parseFloat(logs.loss.toFixed(6)));
      history.val_loss.push(parseFloat(logs.val_loss.toFixed(6)));
      history.mae.push(parseFloat(logs.mae.toFixed(6)));
      history.val_mae.push(parseFloat(logs.val_mae.toFixed(6)));

      if (logs.val_loss < bestValLoss) {
        bestValLoss = logs.val_loss;
        bestEpoch   = e;
        logger.info(`[epoch ${String(e).padStart(3)}] loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} ✓ best`);
        await saveModel(model); // save checkpoint
      } else if (e % 10 === 0) {
        logger.info(`[epoch ${String(e).padStart(3)}] loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} mae=${logs.mae.toFixed(4)}`);
      }
    }
  };

  // ── Fit ─────────────────────────────────────────────────────────────────
  logger.info('Starting training...');
  const t0 = Date.now();

  await model.fit(trainData.xs, trainData.ys, {
    epochs:          EPOCHS,
    batchSize:       BATCH_SIZE,
    validationSplit: 0.15,
    shuffle:         true,
    callbacks
  });

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  logger.info(`Training complete in ${elapsed}s — best val_loss=${bestValLoss.toFixed(4)} at epoch ${bestEpoch}`);

  // ── Evaluate on held-out test set ───────────────────────────────────────
  logger.info('Evaluating on test set...');
  const evalResult = model.evaluate(testData.xs, testData.ys, { batchSize: BATCH_SIZE });
  const testLoss = evalResult[0].dataSync()[0];
  const testMAE  = evalResult[1].dataSync()[0];

  // Compute R²
  const preds  = model.predict(testData.xs).dataSync();
  const actuals = testData.ys.dataSync();
  const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length;
  const ssTot = actuals.reduce((s, v) => s + (v - meanActual) ** 2, 0);
  const ssRes = actuals.reduce((s, v, i) => s + (v - preds[i]) ** 2, 0);
  const r2    = 1 - ssRes / ssTot;

  logger.info(`Test results: MSE=${testLoss.toFixed(4)} MAE=${testMAE.toFixed(4)} R²=${r2.toFixed(4)}`);

  // ── Save history ─────────────────────────────────────────────────────────
  const modelsDir = path.dirname(HISTORY_PATH);
  if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });
  fs.writeFileSync(HISTORY_PATH, JSON.stringify({ history, metrics: { testLoss, testMAE, r2 } }, null, 2));
  logger.info(`Training history saved → ${HISTORY_PATH}`);

  // ── Cleanup ──────────────────────────────────────────────────────────────
  trainData.xs.dispose(); trainData.ys.dispose();
  testData.xs.dispose();  testData.ys.dispose();

  logger.info('Done. Model ready for inference.');
  process.exit(0);
}

train().catch(err => { logger.error('Training failed', { error: err.message, stack: err.stack }); process.exit(1); });
