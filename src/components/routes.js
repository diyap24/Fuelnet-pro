'use strict';

/**
 * FuelNet Pro — API Routes
 *
 * POST /api/predict        — single prediction
 * POST /api/predict/batch  — batch predictions (max 100)
 * GET  /api/health         — health check
 * GET  /api/model/info     — model metadata
 * GET  /api/history        — training history
 */

const express = require('express');
const path    = require('path');
const fs      = require('fs');
const router  = express.Router();

const predictionService = require('./predictionService');
const logger            = require('../utils/logger');

// ── Health check ──────────────────────────────────────────────────────────
router.get('/health', (req, res) => {
  res.json({
    status:    'ok',
    model:     predictionService.ready ? 'loaded' : 'not_loaded',
    timestamp: new Date().toISOString(),
    uptime_s:  Math.floor(process.uptime())
  });
});

// ── Model info ────────────────────────────────────────────────────────────
router.get('/model/info', (req, res) => {
  res.json({
    name:         'FuelNet Pro',
    version:      '2.4.0',
    architecture: '7→64→128→64→1',
    input_features: [
      { name: 'displacement', unit: 'L',    range: [1.0, 7.0] },
      { name: 'horsepower',   unit: 'hp',   range: [60, 500]  },
      { name: 'weight',       unit: 'lbs',  range: [1500, 6500] },
      { name: 'cylinders',    unit: 'count',range: [3, 12]    },
      { name: 'modelYear',    unit: 'year', range: [1970, 2024] },
      { name: 'drivetrain',   unit: 'enum', values: { 0:'FWD', 1:'RWD', 2:'AWD', 3:'4WD' } },
      { name: 'transmission', unit: 'enum', values: { 0:'Auto', 1:'Manual', 2:'CVT' } }
    ],
    metrics: { r2: 0.936, rmse_mpg: 1.84, mae_mpg: 1.31 },
    ready: predictionService.ready
  });
});

// ── Training history ──────────────────────────────────────────────────────
router.get('/history', (req, res) => {
  const histPath = path.resolve('./models/training_history.json');
  if (!fs.existsSync(histPath)) {
    return res.status(404).json({ error: 'Training history not found. Run: npm run train' });
  }
  try {
    const data = JSON.parse(fs.readFileSync(histPath, 'utf8'));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: 'Failed to parse training history' });
  }
});

// ── Single prediction ─────────────────────────────────────────────────────
router.post('/predict', async (req, res, next) => {
  try {
    const result = await predictionService.predictOne(req.body);
    res.json({ success: true, data: result });
  } catch (err) {
    if (err.statusCode === 400) {
      return res.status(400).json({ success: false, error: err.message, details: err.details });
    }
    next(err);
  }
});

// ── Batch prediction ──────────────────────────────────────────────────────
router.post('/predict/batch', async (req, res, next) => {
  const { vehicles } = req.body;
  if (!Array.isArray(vehicles) || vehicles.length === 0) {
    return res.status(400).json({ success: false, error: 'Body must include a non-empty "vehicles" array' });
  }
  if (vehicles.length > 100) {
    return res.status(400).json({ success: false, error: 'Batch size exceeds maximum of 100' });
  }
  try {
    const results = await predictionService.predictBatch(vehicles);
    res.json({ success: true, count: results.length, data: results });
  } catch (err) {
    if (err.statusCode === 400) {
      return res.status(400).json({ success: false, error: err.message, details: err.details });
    }
    next(err);
  }
});

module.exports = router;
