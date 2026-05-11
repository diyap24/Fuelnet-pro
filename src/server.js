'use strict';

/**
 * FuelNet Pro — Express Server
 * Entry point for the REST API.
 */

require('dotenv').config();

const express    = require('express');
const cors       = require('cors');
const helmet     = require('helmet');
const morgan     = require('morgan');
const path       = require('path');

const routes             = require('./components/routes');
const predictionService  = require('./components/predictionService');
const logger             = require('./utils/logger');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(morgan('combined', {
  stream: { write: msg => logger.http(msg.trim()) }
}));

// ── Static frontend ───────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, '../public')));

// ── API routes ────────────────────────────────────────────────────────────
app.use('/api', routes);

// ── Catch-all → index.html ────────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// ── Global error handler ──────────────────────────────────────────────────
app.use((err, req, res, next) => {
  logger.error('Unhandled error', { message: err.message, stack: err.stack });
  res.status(500).json({ success: false, error: 'Internal server error' });
});

// ── Boot ──────────────────────────────────────────────────────────────────
async function start() {
  try {
    logger.info('Initializing FuelNet Pro...');
    await predictionService.initialize();
    app.listen(PORT, () => {
      logger.info(`Server running → http://localhost:${PORT}`);
      logger.info('API docs → http://localhost:' + PORT + '/api/model/info');
    });
  } catch (err) {
    logger.warn(`Model not loaded at startup: ${err.message}`);
    logger.warn('Run "npm run train" to train the model first.');
    // Still start the server so health check works
    app.listen(PORT, () => {
      logger.info(`Server running (no model) → http://localhost:${PORT}`);
    });
  }
}

start();

module.exports = app; // for testing
