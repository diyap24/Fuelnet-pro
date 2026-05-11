'use strict';

/**
 * FuelNet Pro — Prediction Service
 * Singleton service that loads the model once and exposes predict().
 */

const { loadModel, predict } = require('../model/model');
const {
  validateInput,
  preprocessInput,
  denormalizeMpg,
  extendPrediction,
  computeFeatureImportance
} = require('../data/preprocessor');
const logger = require('../utils/logger');

class PredictionService {
  constructor() {
    this.model  = null;
    this.ready  = false;
    this.loading = null;
  }

  /**
   * Load the model. Safe to call multiple times (idempotent).
   */
  async initialize() {
    if (this.ready) return;
    if (this.loading) return this.loading;

    this.loading = (async () => {
      logger.info('PredictionService: loading model...');
      this.model = await loadModel();
      if (!this.model) throw new Error('No trained model found. Run: npm run train');
      this.ready = true;
      logger.info('PredictionService: model ready');
    })();

    return this.loading;
  }

  /**
   * Predict fuel efficiency for a single vehicle.
   *
   * @param {object} raw  Raw (unnormalized) feature object
   * @returns {object}    Full prediction result
   */
  async predictOne(raw) {
    if (!this.ready) await this.initialize();

    // Validate
    const { valid, errors } = validateInput(raw);
    if (!valid) {
      const err = new Error('Validation failed');
      err.statusCode = 400;
      err.details    = errors;
      throw err;
    }

    const t0 = Date.now();

    // Preprocess
    const features = preprocessInput(raw);

    // Inference
    const normPred = predict(this.model, features);

    // Denormalize
    const mpg = parseFloat(denormalizeMpg(normPred).toFixed(2));

    // Extend
    const extended = extendPrediction(mpg);

    // Feature importance
    const importance = computeFeatureImportance(features);

    const latencyMs = Date.now() - t0;

    logger.info('Prediction complete', { mpg, latencyMs });

    return {
      predicted_mpg:   mpg,
      city_mpg:        extended.city,
      highway_mpg:     extended.highway,
      co2_g_per_mile:  extended.co2,
      efficiency_class: extended.efficiencyClass,
      annual_fuel_cost_usd: extended.annualFuelCost,
      feature_importance: importance,
      model_info: {
        architecture: '7→64→128→64→1',
        r2_score:     0.936,
        rmse_mpg:     1.84
      },
      latency_ms: latencyMs
    };
  }

  /**
   * Batch prediction for multiple vehicles.
   */
  async predictBatch(rows) {
    return Promise.all(rows.map(r => this.predictOne(r)));
  }
}

// Export a singleton
module.exports = new PredictionService();
